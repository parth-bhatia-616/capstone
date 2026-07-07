"""
Battery Grading System
────────────────────────
Grade A/B/C based on SoH + uncertainty.
Conservative grading when model confidence is low.
"""

from config import GRADING
from typing import Optional, List


def grade_battery(soh: float, uncertainty: float = None) -> str:
    """
    Grade a battery by SoH.
      A: SoH ≥ 80%    → Primary use
      B: 60% ≤ SoH < 80% → Second-life storage
      C: SoH < 60%    → End-of-life / recycle

    If uncertainty > 3%, conservatively downgrade by half the std.
    """
    grade_A = GRADING["grade_A"]   # 80.0
    grade_B = GRADING["grade_B"]   # 60.0

    # Conservative lower bound
    effective_soh = soh - (uncertainty * 0.5 if uncertainty and uncertainty > 3.0 else 0.0)

    if effective_soh >= grade_A:
        return "A"
    elif effective_soh >= grade_B:
        return "B"
    return "C"


def get_recommendation(soh: float, grade: str,
                        uncertainty: float = None) -> str:
    """Human-readable recommendation for a given grade."""
    msgs = {
        "A": ("✅ Excellent condition. Suitable for primary EV duty, "
              "high-demand applications, and first-life deployment."),
        "B": ("⚠️  Degraded but usable. Recommend second-life as stationary "
              "storage (solar buffer, UPS). Avoid rates > 2C."),
        "C": ("🔴 End of life. Safe disassembly, material recovery, "
              "and recycling. Do NOT use under load."),
    }
    base = msgs.get(grade, "Unknown grade.")
    if uncertainty and uncertainty > 3.0:
        base += f" (Model uncertainty ±{uncertainty:.1f}% — consider re-running with fresh data.)"
    return base


def grade_pack(cell_sohs: List[float],
               uncertainties: List[float] = None) -> dict:
    """
    Grade a 4S pack. Pack grade = weakest cell.

    Returns dict: cell_sohs, cell_grades, pack_grade, pack_soh_mean,
                  pack_soh_min, pack_recommendation
    """
    if uncertainties is None:
        uncertainties = [None] * len(cell_sohs)

    cell_grades = [grade_battery(s, u) for s, u in zip(cell_sohs, uncertainties)]
    grade_order = {"A": 0, "B": 1, "C": 2}
    worst_grade = max(cell_grades, key=lambda g: grade_order[g])

    return {
        "cell_sohs":          cell_sohs,
        "cell_grades":        cell_grades,
        "pack_grade":         worst_grade,
        "pack_soh_mean":      sum(cell_sohs) / len(cell_sohs),
        "pack_soh_min":       min(cell_sohs),
        "pack_recommendation": get_recommendation(min(cell_sohs), worst_grade),
    }


if __name__ == "__main__":
    for soh, unc in [(93.0, 0.5), (76.5, 2.0), (76.5, 8.0), (58.0, 1.5)]:
        g = grade_battery(soh, unc)
        r = get_recommendation(soh, g, unc)
        print(f"SoH={soh:.1f}% ±{unc}  →  Grade {g}  |  {r[:55]}...")

    pack = grade_pack([88.0, 79.5, 82.1, 85.3])
    print(f"\nPack: {pack['pack_grade']} | Mean {pack['pack_soh_mean']:.1f}%")