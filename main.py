#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Site Load Balancer (Per‑LVDB Limits) – with priorities:
- HYC50 must be exactly 50 kW each (if site capacity can't meet this, we warn).
- Prioritise HYC400 to reach 200 kW each, then up to 400 kW when possible
  so long as doing so does NOT push Gen4 below 90 kW per unit.
- Gen4 should have a baseline of 90 kW per unit when capacity allows.
- Per‑LVDB kVA caps supported.

Notes
-----
- ASCII-safe prints (.format), no f-strings.
- Interactive prompts for simplicity (works on any Python 3.x).
- If total site capacity cannot meet the *mandatory* HYC50 requirement across LVDBs,
  we proportionally scale HYC50 down and warn (rule is unsatisfiable otherwise).

Build (macOS)
-------------
    python3 -m pip install --upgrade pip
    pip install pyinstaller
    pyinstaller --onefile site_load_balancer.py

Run
---
    python3 site_load_balancer.py
"""

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.units import mm
import os
import sys
from io import BytesIO

DEFAULT_PF = 0.95
DEFAULT_RESERVE_PCT = 0.02
UNIT_MAX_KW = {"gen4": 350.0, "hyc400": 400.0, "hyc50": 50.0, "ion350": 350.0}
MIN_PER_UNIT_FLOOR_KW = 50.0   # generic small floor (for other units)
GEN4_ION_FLOOR_KW = 70.0      # Gen4 and ION350 must be >= 70 kW
GEN4_BASELINE_KW = 90.0       # target baseline for Gen4 units
HYC400_TARGET1 = 200.0        # priority stage 1 target
HYC400_TARGET2 = 400.0        # maximum for HYC400



@dataclass
class LVDB:
    index: int
    units: Dict[str, int]
    limit_kva: Optional[float] = None
    max_possible_kw: float = field(init=False)

    def __post_init__(self):
        self.max_possible_kw = 0.0
        for k, n in self.units.items():
            self.max_possible_kw += UNIT_MAX_KW.get(k, 0.0) * float(n)


@dataclass
class Site:
    trafos: int
    lvdbs: List[LVDB]
    grid_kva: float
    pf: float = DEFAULT_PF
    aux_kw: float = 0.0
    reserve_pct: float = DEFAULT_RESERVE_PCT

    @property
    def grid_kw(self) -> float:
        return self.grid_kva * self.pf

    @property
    def capacity_kw(self) -> float:
        return max(0.0, self.grid_kw - self.aux_kw - (self.grid_kw * self.reserve_pct))


def compute_lvdb_caps_kw(site: Site) -> List[float]:
    caps = []
    for lv in site.lvdbs:
        if lv.limit_kva is None:
            caps.append(lv.max_possible_kw)
        else:
            caps.append(min(lv.limit_kva * site.pf, lv.max_possible_kw))
    return [max(0.0, c) for c in caps]


def proportional_site_allocation_with_hyc50_min(site: Site) -> Tuple[List[float], List[str]]:
    """Allocate site.capacity_kw across LVDBs with per-LVDB caps, guaranteeing HYC50=50 kW each if feasible.

    Returns: (lvdb_alloc_kw, warnings)
    """
    warnings: List[str] = []
    total_site_kw = site.capacity_kw
    if total_site_kw <= 0.0 or len(site.lvdbs) == 0:
        return [0.0 for _ in site.lvdbs], warnings

    lv_caps_kw = compute_lvdb_caps_kw(site)

    # Mandatory consumption from HYC50 (fixed 50 each), bounded by per-LVDB cap
    mandatory = []
    total_mandatory = 0.0
    for lv, cap in zip(site.lvdbs, lv_caps_kw):
        hyc50_count = lv.units.get("hyc50", 0)
        need = float(hyc50_count) * 50.0
        need = min(need, cap)  # cannot exceed LVDB cap
        mandatory.append(need)
        total_mandatory += need

    if total_mandatory > total_site_kw + 1e-9:
        # Not enough site capacity to give all HYC50 their 50 kW.
        # Scale HYC50 proportionally across LVDBs.
        scale = total_site_kw / total_mandatory if total_mandatory > 0 else 0.0
        for i in range(len(mandatory)):
            mandatory[i] *= scale
        warnings.append("WARNING: Insufficient site capacity to supply all HYC50 at 50 kW. Scaled proportionally.")
        total_mand_after = sum(mandatory)
        remaining_kw = max(0.0, total_site_kw - total_mand_after)
    else:
        remaining_kw = max(0.0, total_site_kw - total_mandatory)

    # Base proportional shares for remaining capacity using theoretical max (excluding the part already consumed by HYC50)
    total_lvdb_max = sum(lv.max_possible_kw for lv in site.lvdbs)
    base_shares = []
    for lv in site.lvdbs:
        if total_lvdb_max > 0:
            base_shares.append((lv.max_possible_kw / total_lvdb_max) * remaining_kw)
        else:
            base_shares.append(0.0)

    # Respect caps: we cannot exceed (cap - mandatory)
    lvdb_alloc = mandatory[:]
    remaining_indices = set(range(len(site.lvdbs)))

    for _ in range(6):
        if remaining_kw <= 1e-9 or not remaining_indices:
            break
        weight_sum = sum(base_shares[i] for i in remaining_indices)
        if weight_sum <= 1e-12:
            break
        progress = 0.0
        for i in list(remaining_indices):
            cap_left = max(0.0, lv_caps_kw[i] - lvdb_alloc[i])
            if cap_left <= 1e-9:
                remaining_indices.discard(i)
                continue
            add = remaining_kw * (base_shares[i] / weight_sum)
            if add > cap_left:
                add = cap_left
            if add > 0:
                lvdb_alloc[i] += add
                remaining_kw -= add
                progress += add
            if lvdb_alloc[i] >= lv_caps_kw[i] - 1e-9:
                remaining_indices.discard(i)
            if remaining_kw <= 1e-9:
                break
        if progress <= 1e-9:
            break

    return lvdb_alloc, warnings


def allocate_inside_lvdb(lv: LVDB, budget_kw: float, site_pf: float) -> Tuple[Dict[str, List[float]], List[str]]:
    """Allocate a single LVDB's budget with rules:
    - HYC50 fixed 50 kW each (top priority; if impossible, scale and warn).
    - Gen4 baseline 90 kW per unit when capacity allows.
    - HYC400: minimum 200 kW, then try to push toward 400 kW but never reduce Gen4 below 90 to do so.
    - Nothing should end at 0 kW: enforce a small floor (>0) for non-fixed types, redistributing if needed.
    """
    warnings: List[str] = []
    alloc: Dict[str, List[float]] = {"gen4": [], "hyc400": [], "hyc50": [], "ion350": []}

    # 1) HYC50 fixed 50 kW each
    hyc50_n = lv.units.get("hyc50", 0)
    hyc50_need = hyc50_n * 50.0
    if budget_kw <= 1e-9:
        return alloc, warnings
    if hyc50_need > budget_kw + 1e-9:
        scale = max(0.0, budget_kw / hyc50_need) if hyc50_need > 0 else 0.0
        if hyc50_n > 0:
            per = 50.0 * scale
            alloc["hyc50"] = [per for _ in range(hyc50_n)]
        warnings.append("LVDB {}: Insufficient budget for HYC50=50 kW. Scaled to {:.1f}% per unit.".format(lv.index, scale * 100.0))
        return alloc, warnings
    else:
        if hyc50_n > 0:
            alloc["hyc50"] = [50.0 for _ in range(hyc50_n)]
        budget_kw -= hyc50_need

    # Prepare arrays
    gen4_n = lv.units.get("gen4", 0)
    hyc400_n = lv.units.get("hyc400", 0)
    ion350_n = lv.units.get("ion350", 0)
    if gen4_n > 0:
        alloc["gen4"] = [0.0 for _ in range(gen4_n)]
    if hyc400_n > 0:
        alloc["hyc400"] = [0.0 for _ in range(hyc400_n)]
    if ion350_n > 0:
        alloc["ion350"] = [0.0 for _ in range(ion350_n)]

    def push_uniform(current: List[float], target: float, cap: float, amount_avail: float) -> Tuple[List[float], float]:
        if not current:
            return current, amount_avail
        for i in range(len(current)):
            need = max(0.0, min(target, cap) - current[i])
            give = min(need, amount_avail)
            current[i] += give
            amount_avail -= give
            if amount_avail <= 1e-9:
                break
        return current, amount_avail

    # 2) Gen4 baseline up to 90 kW/unit (best effort)
    if gen4_n > 0 and budget_kw > 0:
        target_total = gen4_n * GEN4_BASELINE_KW
        give = min(target_total, budget_kw)
        per_inc = give / gen4_n if gen4_n > 0 else 0.0
        for i in range(gen4_n):
            alloc["gen4"][i] += per_inc
        if give < target_total - 1e-9:
            warnings.append("LVDB {}: Gen4 baseline 90 kW unmet due to budget/cap.".format(lv.index))
        budget_kw -= give

    # 3) HYC400 minimum 200 kW per unit
    if hyc400_n > 0 and budget_kw > 0:
        need_to_200 = max(0.0, hyc400_n * 200.0 - sum(alloc["hyc400"]))
        give = min(need_to_200, budget_kw)
        per_inc = give / hyc400_n if hyc400_n > 0 else 0.0
        for i in range(hyc400_n):
            alloc["hyc400"][i] += per_inc
        if give < need_to_200 - 1e-9:
            warnings.append("LVDB {}: HYC400 minimum 200 kW unmet due to budget/cap.".format(lv.index))
        budget_kw -= give

    # 4) Lift HYC400 toward 400 if budget remains (never subtracting from Gen4)
    if hyc400_n > 0 and budget_kw > 0:
        alloc["hyc400"], budget_kw = push_uniform(alloc["hyc400"], HYC400_TARGET2, UNIT_MAX_KW["hyc400"], budget_kw)

    # 5) Distribute remaining proportionally to headroom across Gen4/HYC400/ION350
    if budget_kw > 1e-9:
        headrooms = []
        for t in ("gen4", "hyc400", "ion350"):
            cap = UNIT_MAX_KW.get(t, 0.0)
            floor = 200.0 if t == "hyc400" and hyc400_n > 0 else MIN_PER_UNIT_FLOOR_KW
            for idx in range(len(alloc[t])):
                head = max(0.0, cap - max(alloc[t][idx], floor))
                if head > 1e-9:
                    headrooms.append((t, idx, head))
        total_head = sum(h for (_, _, h) in headrooms)
        if total_head > 1e-9:
            for (t, idx, h) in headrooms:
                add = budget_kw * (h / total_head)
                alloc[t][idx] += add
            budget_kw = 0.0

    # 6) Enforce non-zero floors by redistribution if needed
    need = 0.0
    topups = []
    for t in ("gen4", "hyc400", "ion350"):
        floor = 200.0 if t == "hyc400" and hyc400_n > 0 else MIN_PER_UNIT_FLOOR_KW
        for idx in range(len(alloc[t])):
            if alloc[t][idx] < floor - 1e-9:
                req = floor - alloc[t][idx]
                topups.append((t, idx, req))
                need += req
    if need > 1e-9:
        donors = []
        for t in ("gen4", "hyc400", "ion350"):
            min_allowed = 200.0 if t == "hyc400" else MIN_PER_UNIT_FLOOR_KW
            for idx in range(len(alloc[t])):
                can = max(0.0, alloc[t][idx] - min_allowed)
                if t == "hyc400":
                    min_allowed = 200.0
                elif t in ("gen4", "ion350"):
                    min_allowed = GEN4_ION_FLOOR_KW
                else:
                    min_allowed = MIN_PER_UNIT_FLOOR_KW

        total_can = sum(can for (_, _, can) in donors)
        if total_can < need - 1e-9:
            warnings.append("LVDB {}: Cannot enforce non-zero floors for all units within caps; some may remain below floors.".format(lv.index))
        else:
            for (t, idx, req) in topups:
                remain = req
                for j in range(len(donors)):
                    dt, didx, can = donors[j]
                    if can <= 1e-9:
                        continue
                    take = min(can, remain)
                    alloc[dt][didx] -= take
                    donors[j] = (dt, didx, can - take)
                    alloc[t][idx] += take
                    remain -= take
                    if remain <= 1e-9:
                        break

    # Final clamps
    for t in ("gen4", "hyc400", "ion350"):
        cap = UNIT_MAX_KW.get(t, 0.0)
        floor = 200.0 if t == "hyc400" and hyc400_n > 0 else MIN_PER_UNIT_FLOOR_KW
        for i in range(len(alloc[t])):
            if t == "hyc400" and hyc400_n > 0:
                floor = 200.0
            elif t in ("gen4", "ion350"):
                floor = GEN4_ION_FLOOR_KW
            else:
                floor = MIN_PER_UNIT_FLOOR_KW



    return alloc, warnings




def interactive_lvdbs(n: int) -> List[LVDB]:
    lvdbs: List[LVDB] = []
    print("\nEnter unit counts and optional kVA limit per LVDB (press Enter for 0 / no limit):")
    for i in range(1, n + 1):
        print("\nLVDB {}".format(i))
        units: Dict[str, int] = {}
        for k in ("gen4", "hyc400", "hyc50", "ion350"):
            # Loop until valid integer or blank is provided
            while True:
                raw = input("  {} count: ".format(k.upper())).strip()
                if raw == "":
                    units[k] = 0
                    break
                try:
                    val = int(raw)
                    if val < 0:
                        print("    Please enter a non-negative integer.")
                        continue
                    units[k] = val
                    break
                except ValueError:
                    print("    Invalid number, please enter an integer.")
        limit_raw = input("  LVDB limit (kVA, blank for no cap): ").strip()
        limit = float(limit_raw) if limit_raw else None
        lvdbs.append(LVDB(index=i, units=units, limit_kva=limit))
    return lvdbs

def write_pdf(path: str,
              site: Site,
              lvdb_budgets: List[float],
              allocations: Dict[int, Dict[str, List[float]]],
              warnings: List[str]) -> None:
    """
    Robust writer:
    - Ensures parent directory exists.
    - Generates with ReportLab to an in-memory buffer, then writes bytes to disk.
    """
    import os
    from io import BytesIO

    # Ensure parent folder exists
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    # Build the PDF into memory first
    buf = BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)

    # Optional metadata
    try:
        c.setAuthor("IONITY")
        c.setTitle("Site Load Plan")
        c.setSubject("Per-LVDB allocation summary")
        c.setCreator("ReportLab PDF Library")
    except Exception:
        # Metadata is nice-to-have; don't fail if fonts/encodings cause issues.
        pass

    w, h = A4
    x = 20 * mm
    y = h - 20 * mm
    line_h = 6 * mm

    def draw_line(text: str, bold: bool = False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11 if bold else 10)
        c.drawString(x, y, text)
        y -= line_h
        if y < 20 * mm:
            c.showPage()
            y = h - 20 * mm

    # Header
    draw_line("IONITY – Site Load Plan", bold=True)
    draw_line("")
    draw_line("Grid (kVA): {:.1f}".format(site.grid_kva))
    draw_line("PF: {:.2f}".format(site.pf))
    draw_line("Aux (kW): {:.1f}".format(site.aux_kw))
    draw_line("Reserve: {:.1f}%".format(site.reserve_pct * 100.0))
    draw_line("Capacity (kW): {:.1f}".format(site.capacity_kw))
    draw_line("")

    # Per-LVDB
    for lv, budget in zip(site.lvdbs, lvdb_budgets):
        cap_kw = lv.limit_kva * site.pf if lv.limit_kva is not None else None
        cap_txt = "no cap" if cap_kw is None else "{:.0f} kVA -> {:.1f} kW cap".format(lv.limit_kva, cap_kw)
        # (Tiny cosmetic fix: add a closing parenthesis in the label)
        draw_line("LVDB {} ({}, theoretical: {:.1f} kW)".format(lv.index, cap_txt, lv.max_possible_kw), bold=True)
        draw_line("  Budget used: {:.1f} kW".format(budget))
        alloc = allocations.get(lv.index, {})
        total_lv = 0.0
        for t in ("hyc50", "hyc400", "gen4", "ion350"):
            per_units = alloc.get(t, [])
            for j, kw in enumerate(per_units, 1):
                draw_line("  {:6s} #{:02d}: {:6.1f} kW".format(t.upper(), j, kw))
                total_lv += kw
        draw_line("  LVDB total: {:.1f} kW".format(total_lv))
        draw_line("")

    if warnings:
        draw_line("WARNINGS", bold=True)
        for w_msg in warnings:
            draw_line("- {}".format(w_msg))

    # Finalize PDF into buffer
    c.save()

    # Write bytes atomically to disk
    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-kva", type=float)
    parser.add_argument("--pdf", type=str, default=None, help="Path to output PDF report (optional)")
    parser.add_argument("--pf", type=float, default=DEFAULT_PF)
    parser.add_argument("--aux-kw", type=float, default=0.0)
    parser.add_argument("--reserve-pct", type=float, default=DEFAULT_RESERVE_PCT)
    parser.add_argument("--trafos", type=int, default=None)
    parser.add_argument("--lvdbs", type=int, default=None)
    args = parser.parse_args()

    grid_kva = args.grid_kva or float(input("Grid limit (kVA): ") or 0)



    # Ask how many LVDBs (if not provided)
    if args.lvdbs is None:
        lvdb_count_raw = input("How many LVDBs are present (e.g., 1, 2, 3)? ").strip()
        try:
            lvdb_count = int(lvdb_count_raw or "0")
        except ValueError:
            lvdb_count = 0
    else:
        lvdb_count = int(args.lvdbs)
    if lvdb_count < 1:
        raise SystemExit("LVDBs must be >= 1")

    # Ask how many transformers (if not provided)
    if args.trafos is None:
        trafos_raw = input("How many transformers (1-3)? ").strip()
        try:
            trafos = int(trafos_raw or "0")
        except ValueError:
            trafos = 0
    else:
        trafos = int(args.trafos)
    if trafos < 1:
        raise SystemExit("Transformers must be >= 1")

    # Create LVDBs interactively for the given count
    lvdbs = interactive_lvdbs(lvdb_count)

    site = Site(
        trafos=trafos,
        lvdbs=lvdbs,
        grid_kva=grid_kva,
        pf=args.pf,
        aux_kw=args.aux_kw,
        reserve_pct=args.reserve_pct,
    )

    # Allocate across LVDBs honouring HYC50 fixed demand
    lvdb_budgets, warnings = proportional_site_allocation_with_hyc50_min(site)

    # Per-LVDB internal allocation with priorities
    all_allocs: Dict[int, Dict[str, List[float]]] = {}
    all_warnings = list(warnings)
    for lv, budget in zip(site.lvdbs, lvdb_budgets):
        alloc, local_warn = allocate_inside_lvdb(lv, budget, site.pf)
        all_allocs[lv.index] = alloc
        all_warnings.extend(local_warn)

    # ---- Summary ----
    print("--- SITE SUMMARY ---")
    print("Grid (kVA): {:.1f}".format(site.grid_kva))
    print("PF: {:.2f}".format(site.pf))
    print("Aux: {:.1f} kW".format(site.aux_kw))
    print("Reserve: {:.1f}%".format(site.reserve_pct * 100.0))
    print("Capacity (kW): {:.1f}".format(site.capacity_kw))

    for lv, budget in zip(site.lvdbs, lvdb_budgets):
        cap_kw = lv.limit_kva * site.pf if lv.limit_kva is not None else None
        cap_txt = "no cap" if cap_kw is None else "{:.0f} kVA -> {:.1f} kW cap".format(lv.limit_kva, cap_kw)
        print("LVDB {} (cap: {}, theoretical: {:.1f} kW)".format(lv.index, cap_txt, lv.max_possible_kw))
        print("  Budget used: {:.1f} kW".format(budget))
        alloc = all_allocs.get(lv.index, {})
        total_lv = 0.0
        for t in ("hyc50", "hyc400", "gen4", "ion350"):
            per_units = alloc.get(t, [])
            for j, kw in enumerate(per_units, 1):
                print("  {:6s} #{:02d}: {:6.1f} kW".format(t.upper(), j, kw))
                total_lv += kw
        print("  LVDB total: {:.1f} kW".format(total_lv))

    if all_warnings:
        print("--- WARNINGS ---")
        for w in all_warnings:
            print("- {}".format(w))

    if args.pdf is None:
        default_pdf = os.path.join(os.getcwd(), "site_load_plan.pdf")
        ans=(input("Save PDF report (y/N)? ").strip().lower() or "n").lower()
        if ans == "y":
            args.pdf = default_pdf
            print(args.pdf)
            print("PDF will be saved to: {}".format(args.pdf))

    # PDF export (optional)
    if args.pdf:
        write_pdf(args.pdf, site, lvdb_budgets, all_allocs, all_warnings)
        print("\nSaved PDF report to: {}".format(args.pdf))


# Ensure there is only a single entrypoint and no code executed at import time.
if __name__ == "__main__":
    main()

