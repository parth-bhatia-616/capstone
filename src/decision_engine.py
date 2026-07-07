

import numpy as np
from typing import List, Optional, Dict
from config import BALANCING


def decide_balancing_mode(
    cell_voltages: List[float],
    soh: float,
    temperature: float,
    soh_list: List[float] = None,
    pack_current: float = None,
) -> str:

    DV_PASS   = BALANCING["dv_passive_threshold"]   # 0.010 V
    DV_ACTIVE = BALANCING["dv_active_threshold"]    # 0.020 V
    T_MAX     = BALANCING["temp_safe_max"]           # 50°C
    SOH_PROT  = BALANCING["soh_protect_threshold"]  # 60%

    v = np.array(cell_voltages, dtype=np.float64)
    delta_v = float(v.max() - v.min())


    if v.max() > 4.25 or v.min() < 2.5 or temperature > 65:
        return "EMERGENCY_STOP"


    if soh < SOH_PROT or temperature > T_MAX:
        return "PROTECT_WEAK_CELL"

    if soh_list and any(s < SOH_PROT for s in soh_list):
        return "PROTECT_WEAK_CELL"


    if delta_v < DV_PASS:
        return "NO_BALANCING"
    if delta_v < DV_ACTIVE:
        return "PASSIVE_BALANCING"
    return "ACTIVE_BALANCING"


def get_balancing_target(cell_voltages: List[float], mode: str) -> Dict:

    v = np.array(cell_voltages)
    highest = int(np.argmax(v))
    lowest  = int(np.argmin(v))
    mean_v  = float(v.mean())

    base = {"source_cell": None, "target_cell": None, "energy_mWh": 0.0}

    if mode == "NO_BALANCING" or mode == "EMERGENCY_STOP":
        return {**base, "direction": "none"}

    if mode == "PASSIVE_BALANCING":
        return {
            "source_cell": highest + 1, "target_cell": None,
            "energy_mWh": float((v[highest] - mean_v) * 200),
            "direction":  "dissipate",
        }
    if mode == "ACTIVE_BALANCING":
        return {
            "source_cell": highest + 1, "target_cell": lowest + 1,
            "energy_mWh": float((v[highest] - v[lowest]) * 500),
            "direction":  "transfer",
        }
    if mode == "PROTECT_WEAK_CELL":
        return {
            "source_cell": highest + 1, "target_cell": lowest + 1,
            "energy_mWh": 0.0, "direction": "protect",
        }
    return base


def full_decision(cell_voltages: List[float], soh: float,
                   temperature: float,
                   soh_list: List[float] = None,
                   pack_current: float = None) -> Dict:

    mode   = decide_balancing_mode(cell_voltages, soh, temperature,
                                    soh_list, pack_current)
    target = get_balancing_target(cell_voltages, mode)
    v      = np.array(cell_voltages)
    delta_v = float(v.max() - v.min())
    weakest = int(np.argmin(v)) + 1

    explain = {
        "NO_BALANCING":     "Cells balanced — no action needed.",
        "PASSIVE_BALANCING": f"Cell {target.get('source_cell')} too high. "
                             "Bleeding via resistor.",
        "ACTIVE_BALANCING":  f"Transferring energy Cell {target.get('source_cell')} "
                             f"→ Cell {target.get('target_cell')} via inductor.",
        "PROTECT_WEAK_CELL": f"Cell {weakest} weak or temp elevated. "
                             "Limiting charge/discharge rate.",
        "EMERGENCY_STOP":    "CRITICAL: Cell voltage or temperature out of bounds! "
                             "Disconnect pack immediately.",
    }

    return {
        "mode":       mode,
        "delta_v":    delta_v,
        "weakest_cell": weakest,
        "explanation": explain.get(mode, ""),
        **target,
    }


if __name__ == "__main__":
    scenarios = [
        ([3.70, 3.71, 3.70, 3.71], 90, 25, "Balanced"),
        ([3.65, 3.72, 3.68, 3.70], 85, 28, "Slight imbalance"),
        ([3.50, 3.72, 3.60, 3.71], 75, 32, "Large imbalance"),
        ([3.65, 3.72, 3.68, 3.70], 55, 28, "Low SoH"),
        ([3.65, 3.72, 3.68, 3.70], 82, 53, "High temp"),
        ([4.27, 3.70, 3.68, 3.71], 90, 25, "Overvoltage"),
    ]
    for v, soh, t, desc in scenarios:
        r = full_decision(v, soh, t)
        print(f"{desc:<20} → {r['mode']:<22} ΔV={r['delta_v']*1000:.0f}mV")