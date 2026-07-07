
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis
import joblib

warnings.filterwarnings("ignore")




def compute_ic_features(voltage: np.ndarray, soc: np.ndarray,
                        capacity_Ah: float, n_bins: int = 200) -> dict:
  
    base = {"ic_peak_height": 0.0, "ic_peak_voltage": 0.0, "ic_peak_width": 0.0,
            "ic_area": 0.0, "ic_valley_depth": 0.0, "ic_skewness": 0.0}

    if len(voltage) < 15 or capacity_Ah < 0.01:
        return base

    try:
        dv = np.diff(voltage)
        dq = np.diff(soc) * capacity_Ah

        mask = np.abs(dv) > 1e-6
        ic = np.zeros(len(dv))
        ic[mask] = np.abs(dq[mask] / dv[mask])

        # Smooth with Savitzky-Golay
        win = min(11, len(ic) - 1 if len(ic) % 2 == 0 else len(ic))
        win = win if win % 2 == 1 else win - 1
        win = max(win, 5)
        if win >= len(ic):
            return base
        ic_smooth = savgol_filter(ic, win, 3)
        ic_smooth = np.clip(ic_smooth, 0, None)

        v_mid = voltage[:-1]

        # Primary peak
        peak_idx = int(np.argmax(ic_smooth))
        peak_h = float(ic_smooth[peak_idx])
        peak_v = float(v_mid[peak_idx]) if peak_idx < len(v_mid) else 0.0

        # Full-width half-maximum
        half_max = peak_h / 2.0
        above = ic_smooth > half_max
        width = float(np.sum(above) * (v_mid[-1] - v_mid[0]) / max(len(v_mid), 1))

        return {
            "ic_peak_height": peak_h,
            "ic_peak_voltage": peak_v,
            "ic_peak_width": width,
            "ic_area": float(np.trapz(ic_smooth, v_mid)),
            "ic_valley_depth": float(np.min(ic_smooth)),
            "ic_skewness": float(skew(ic_smooth)),
        }
    except Exception:
        return base




def compute_dv_features(voltage: np.ndarray, soc: np.ndarray,
                        capacity_Ah: float) -> dict:

    base = {"dv_peak_height": 0.0, "dv_peak_soc": 0.0,
            "dv_slope_early": 0.0, "dv_slope_late": 0.0,
            "dv_valley_soc": 0.0, "dv_asymmetry": 0.0}

    if len(voltage) < 15:
        return base

    try:
        dq = np.diff(soc) * capacity_Ah
        dv = np.diff(voltage)

        mask = np.abs(dq) > 1e-8
        dv_dq = np.zeros(len(dq))
        dv_dq[mask] = np.abs(dv[mask] / dq[mask])

        win = min(11, len(dv_dq) - 1 if len(dv_dq) % 2 == 0 else len(dv_dq))
        win = win if win % 2 == 1 else win - 1
        win = max(win, 5)
        if win >= len(dv_dq):
            return base
        dv_smooth = savgol_filter(dv_dq, win, 3)
        dv_smooth = np.clip(dv_smooth, 0, None)

        soc_mid = soc[:-1]
        n = len(dv_smooth)

        peak_idx = int(np.argmax(dv_smooth))
        valley_idx = int(np.argmin(dv_smooth))

        return {
            "dv_peak_height": float(dv_smooth[peak_idx]),
            "dv_peak_soc": float(soc_mid[peak_idx]) if peak_idx < len(soc_mid) else 0.0,
            "dv_slope_early": float(np.mean(dv_smooth[:n // 4])) if n > 4 else 0.0,
            "dv_slope_late": float(np.mean(dv_smooth[3 * n // 4:])) if n > 4 else 0.0,
            "dv_valley_soc": float(soc_mid[valley_idx]) if valley_idx < len(soc_mid) else 0.0,
            "dv_asymmetry": float(np.mean(dv_smooth[:n // 2]) - np.mean(dv_smooth[n // 2:])),
        }
    except Exception:
        return base




def compute_relaxation_features(voltage: np.ndarray, current: np.ndarray,
                                 dt_s: float = 10.0,
                                 rest_window: int = 6) -> dict:
 
    base = {"v_relax_final": 0.0, "v_relax_drop": 0.0,
            "v_relax_tau": 0.0, "v_relax_ir_drop": 0.0}

    if len(voltage) < rest_window + 5:
        return base

    try:
        # Find rest region (|I| < 0.05 A)
        rest_mask = np.abs(current) < 0.05
        rest_indices = np.where(rest_mask)[0]

        if len(rest_indices) < rest_window:
            # Use end of discharge as proxy
            v_end = voltage[-rest_window:]
            relax_drop = float(v_end[0] - v_end[-1])
            return {
                "v_relax_final": float(v_end[-1]),
                "v_relax_drop": relax_drop,
                "v_relax_tau": 0.0,
                "v_relax_ir_drop": relax_drop,
            }

        v_rest = voltage[rest_indices[:rest_window]]
        t_rest = np.arange(len(v_rest)) * dt_s


        v_inf = v_rest[-1]
        dv = v_rest - v_inf
        tau_est = 60.0
        try:
            mask = dv > 0.001
            if mask.sum() > 3:
                log_dv = np.log(np.abs(dv[mask]) + 1e-10)
                slope, _ = np.polyfit(t_rest[mask], log_dv, 1)
                tau_est = float(max(1.0, -1.0 / slope)) if slope < 0 else 60.0
        except Exception:
            pass

        return {
            "v_relax_final": float(v_inf),
            "v_relax_drop": float(v_rest[0] - v_inf),
            "v_relax_tau": tau_est,
            "v_relax_ir_drop": float(v_rest[0] - v_rest[1]) if len(v_rest) > 1 else 0.0,
        }
    except Exception:
        return base




def compute_partial_features(voltage: np.ndarray, soc: np.ndarray,
                              current: np.ndarray,
                              soc_low: float = 0.1,
                              soc_high: float = 0.8) -> dict:
    
    base = {"partial_v_10_80_range": 0.0, "partial_capacity_10_80": 0.0,
            "partial_v_slope": 0.0, "partial_energy_Wh": 0.0,
            "partial_r_est": 0.0}

    if len(voltage) < 10:
        return base

    try:
        mask = (soc >= soc_low) & (soc <= soc_high)
        if mask.sum() < 5:
            return base

        v_win = voltage[mask]
        soc_win = soc[mask]
        i_win = current[mask] if len(current) == len(voltage) else np.ones(mask.sum())

        v_range = float(v_win[-1] - v_win[0])
        q_range = float(soc_win[-1] - soc_win[0])
        slope = v_range / (q_range + 1e-8)


        if len(v_win) > 1:
            dt_approx = 10.0
            energy = float(np.trapz(np.abs(v_win * i_win), dx=dt_approx) / 3600)
        else:
            energy = 0.0


        r_est = 0.0
        if len(v_win) > 3 and len(i_win) > 3:
            di = np.diff(i_win[:5])
            dv = np.diff(v_win[:5])
            mask_di = np.abs(di) > 0.05
            if mask_di.any():
                r_est = float(np.mean(np.abs(dv[mask_di] / di[mask_di])))

        return {
            "partial_v_10_80_range": v_range,
            "partial_capacity_10_80": q_range,
            "partial_v_slope": slope,
            "partial_energy_Wh": energy,
            "partial_r_est": r_est,
        }
    except Exception:
        return base



def compute_eis_features(R0: float, R1: float, C1: float,
                         sei_thickness: float = 0.0) -> dict:
 
    tau = R1 * C1
    R_ct = R1 + sei_thickness * 0.02
    Z_w = 0.005 + sei_thickness * 0.01
    f_mid = 10.0  # Hz
    Z_total = R0 + R_ct + Z_w
    phase_angle = -np.degrees(np.arctan(1.0 / (2 * np.pi * f_mid * R_ct * C1 + 1e-10)))

    return {
        "eis_R0": float(R0),
        "eis_Rct": float(R_ct),
        "eis_Zw": float(Z_w),
        "eis_phase_angle": float(phase_angle),
        "eis_total_impedance": float(Z_total),
        "eis_tau": float(tau),
    }




def add_rolling_features(df: pd.DataFrame,
                          feature_cols: list,
                          window: int = 10,
                          group_col: str = "battery_id") -> pd.DataFrame:

    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        # Rolling mean
        df[f"{col}_roll_mean"] = (
            df.groupby(group_col)[col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        # Rolling slope (linear trend over window)
        def _slope(series):
            result = series.copy() * 0.0
            for i in range(len(series)):
                start = max(0, i - window + 1)
                seg = series.iloc[start:i + 1].values
                if len(seg) < 2:
                    result.iloc[i] = 0.0
                else:
                    x = np.arange(len(seg), dtype=float)
                    m, _ = np.polyfit(x, seg, 1)
                    result.iloc[i] = m
            return result

        df[f"{col}_roll_slope"] = (
            df.groupby(group_col)[col]
            .transform(_slope)
        )
    return df




def coulomb_counting(current_A: np.ndarray, dt_s: float,
                     initial_soc: float = 1.0,
                     capacity_Ah: float = 3.0,
                     coulombic_efficiency: float = 0.98) -> np.ndarray:

    soc = initial_soc
    soc_arr = np.zeros(len(current_A))

    for i, I in enumerate(current_A):
        if I > 0:  # discharge
            dsoc = -I * dt_s / (capacity_Ah * 3600)
        else:  # charge
            dsoc = -I * dt_s * coulombic_efficiency / (capacity_Ah * 3600)
        soc = float(np.clip(soc + dsoc, 0.0, 1.0))
        soc_arr[i] = soc

    return soc_arr




def estimate_internal_resistance(voltage: np.ndarray,
                                  current: np.ndarray,
                                  min_di: float = 0.1) -> float:
  
    dI = np.diff(current)
    dV = np.diff(voltage)
    mask = np.abs(dI) > min_di
    if not mask.any():
        return np.nan
    r_estimates = np.abs(dV[mask] / dI[mask])
    # Keep physically reasonable values
    r_estimates = r_estimates[(r_estimates > 0.005) & (r_estimates < 0.5)]
    return float(np.median(r_estimates)) if len(r_estimates) > 0 else np.nan




def extract_cycle_features(cycle_row: pd.Series,
                            v_arr: np.ndarray = None,
                            soc_arr: np.ndarray = None,
                            i_arr: np.ndarray = None,
                            dt_s: float = 10.0) -> dict:
   
    features = {}

    # Base features from row
    for col in ["cycle", "temperature", "ambient_temperature", "internal_resistance",
                "crate_mean", "dod_mean", "calendar_days", "sei_thickness_nm",
                "v_mean", "v_std", "v_relaxation", "v_drop_start",
                "energy_Wh", "capacity_throughput_Ah"]:
        features[col] = float(cycle_row.get(col, 0.0))

    # IC/DV/partial from arrays if available
    if v_arr is not None and soc_arr is not None and len(v_arr) > 10:
        cap = float(cycle_row.get("measured_capacity_Ah", 1.0))
        features.update(compute_ic_features(v_arr, soc_arr, cap))
        features.update(compute_dv_features(v_arr, soc_arr, cap))
        if i_arr is not None:
            features.update(compute_partial_features(v_arr, soc_arr, i_arr))
            features.update(compute_relaxation_features(v_arr, i_arr, dt_s))
            r_est = estimate_internal_resistance(v_arr, i_arr)
            if not np.isnan(r_est):
                features["internal_resistance_dv_di"] = r_est
    else:
        # Fall back to pre-computed from data_generator
        for col in ["ic_peak_height", "ic_peak_voltage", "ic_peak_width",
                    "ic_area", "ic_valley_depth",
                    "dv_peak_height", "dv_peak_soc", "dv_slope_early", "dv_slope_late",
                    "partial_v_10_80_range", "partial_capacity_10_80", "partial_v_slope"]:
            features[col] = float(cycle_row.get(col, 0.0))

    # EIS features
    R0 = float(cycle_row.get("internal_resistance", cycle_row.get("eis_R0", 0.025)))
    R1 = float(cycle_row.get("Rct", 0.015))
    C1 = float(cycle_row.get("C1", 3000.0))
    sei = float(cycle_row.get("sei_thickness_nm", 0.0)) / 1e9
    features.update(compute_eis_features(R0, R1, C1, sei))

    return features


# ─── Batch Feature Engineering ────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame,
                          rolling_window: int = 10,
                          add_rolling: bool = True) -> pd.DataFrame:

    print(f"Building feature matrix from {len(df)} rows, {df['battery_id'].nunique()} batteries...")

    # Extract per-row features
    rows = []
    for _, row in df.iterrows():
        f = extract_cycle_features(row)
        # Preserve identifiers and targets
        f["battery_id"] = row.get("battery_id", "unknown")
        f["chemistry"] = row.get("chemistry", "NMC")
        f["form_factor"] = row.get("form_factor", "cylindrical")
        f["aging_mode"] = row.get("aging_mode", "cycle_aging")
        f["soh"] = float(row.get("soh", 100.0))
        f["soc_true"] = float(row.get("soc_true", 0.0))
        f["measured_capacity_Ah"] = float(row.get("measured_capacity_Ah",
                                                    row.get("Capacity", 1.0)))
        rows.append(f)

    feat_df = pd.DataFrame(rows)

    # Fill NaN with column median (per chemistry for accuracy)
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns
    feat_df[numeric_cols] = feat_df[numeric_cols].fillna(
        feat_df[numeric_cols].median()
    )

    # Rolling features on key degradation indicators
    if add_rolling:
        rolling_cols = ["internal_resistance", "measured_capacity_Ah",
                        "eis_R0", "eis_Rct", "ic_peak_height", "v_mean"]
        rolling_cols = [c for c in rolling_cols if c in feat_df.columns]
        feat_df = add_rolling_features(feat_df, rolling_cols,
                                        window=rolling_window)

    print(f"Feature matrix: {feat_df.shape[1]} columns, {len(feat_df)} rows")
    return feat_df


if __name__ == "__main__":
    # Quick smoke test
    sys.path.insert(0, os.path.dirname(__file__))
    from data_generator import generate_dataset

    print("Generating small dataset for feature engineering test...")
    df = generate_dataset(n_batteries_per_chemistry=1, n_cycles_per_battery=30)
    feat_df = build_feature_matrix(df, rolling_window=5)
    print(feat_df[["battery_id", "cycle", "soh", "ic_peak_height",
                    "eis_R0", "v_mean"]].head(10))
    print(f"Total features: {feat_df.shape[1]}")