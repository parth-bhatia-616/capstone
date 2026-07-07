

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GroupShuffleSplit
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))


SOH_FEATURE_COLS = [
    # Cycle-level measurements
    "cycle",
    "internal_resistance",
    "temperature",
    "ambient_temperature",
    # IC/DV curve features
    "ic_peak_height",
    "ic_peak_voltage",
    "ic_peak_width",
    "ic_area",
    "ic_valley_depth",
    "dv_peak_height",
    "dv_peak_soc",
    "dv_slope_early",
    "dv_slope_late",
    # EIS-inspired
    "eis_R0",
    "eis_Rct",
    "eis_Zw",
    "eis_phase_angle",
    "eis_total_impedance",
    # Partial charge / energy
    "partial_v_10_80_range",
    "partial_capacity_10_80",
    "partial_v_slope",
    "energy_Wh",
    "capacity_throughput_Ah",
    # Derived / stress
    "dod_mean",
    "crate_mean",
    "calendar_days",
    "sei_thickness_nm",
    # Voltage stats
    "v_mean",
    "v_std",
    "v_relaxation",
    "v_drop_start",
]

SOC_FEATURE_COLS = [
    "voltage",
    "current",
    "temperature",
    "dv_dt",
    "di_dt",
    "power",
    "cumulative_charge_Ah",
]

TARGET_SOH = "soh"
TARGET_SOC = "soc_true"



def clean_battery_data(df: pd.DataFrame, min_soh: float = 55.0) -> pd.DataFrame:
  
    original_len = len(df)


    cap_col = "Capacity" if "Capacity" in df.columns else "measured_capacity_Ah"
    if cap_col in df.columns:
        df = df[df[cap_col] > 0.01].copy()


    if TARGET_SOH in df.columns:
        df = df[df[TARGET_SOH] >= min_soh].copy()


    for col in ["temperature", "ambient_temperature", "temp_mean"]:
        if col in df.columns:
            df[col] = df[col].clip(-10, 70)


    for col in ["crate_mean", "charge_crate", "discharge_crate"]:
        if col in df.columns:
            df[col] = df[col].clip(0.1, 6.0)


    if "internal_resistance" in df.columns:
        df["internal_resistance"] = (
            df.groupby("battery_id")["internal_resistance"]
            .transform(lambda x: x.ffill().bfill())
        )


    sort_cols = ["battery_id", "cycle"] if "cycle" in df.columns else ["battery_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    print(f"Cleaned: {original_len} → {len(df)} rows "
          f"({original_len - len(df)} dropped)")
    return df




def align_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
   
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        warnings.warn(f"Features not found, filling with 0: {missing}")
        for c in missing:
            df[c] = 0.0
    return df[feature_cols].copy()




def split_by_battery(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> tuple:
   
    battery_ids = df["battery_id"].unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(battery_ids)

    n = len(battery_ids)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))

    test_ids  = battery_ids[:n_test]
    val_ids   = battery_ids[n_test:n_test + n_val]
    train_ids = battery_ids[n_test + n_val:]

    train_df = df[df["battery_id"].isin(train_ids)].copy()
    val_df   = df[df["battery_id"].isin(val_ids)].copy()
    test_df  = df[df["battery_id"].isin(test_ids)].copy()

    print(f"Split: train={len(train_df)} ({len(train_ids)} batteries), "
          f"val={len(val_df)} ({len(val_ids)} batteries), "
          f"test={len(test_df)} ({len(test_ids)} batteries)")
    return train_df, val_df, test_df


class BatteryScaler:
  

    def __init__(self, feature_cols: list):
        self.feature_cols = feature_cols
        self.scaler = RobustScaler()
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        X = align_features(df, self.feature_cols)
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = align_features(df, self.feature_cols)
        return self.scaler.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(X)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Scaler saved → {path}")

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    sequence_len: int = 50,
    stride: int = 1,
    group_col: str = "battery_id",
) -> tuple:
 
    X_list, y_list, g_list = [], [], []

    for bid, group in df.groupby(group_col):
        group = group.sort_values("cycle") if "cycle" in group.columns else group
        features = align_features(group, feature_cols).values
        targets  = group[target_col].values

        for i in range(0, len(features) - sequence_len, stride):
            X_list.append(features[i: i + sequence_len])
            y_list.append(targets[i + sequence_len - 1])
            g_list.append(bid)

    if not X_list:
        raise ValueError("No sequences built — check sequence_len vs available data per battery.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    groups = np.array(g_list)

    print(f"Built {len(X)} sequences of shape {X.shape[1:]}")
    return X, y, groups

def inject_noise(
    X: np.ndarray,
    noise_std: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
  
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, size=X.shape)
    return X + noise


def map_nasa_to_schema(df: pd.DataFrame) -> pd.DataFrame:
   
    rename_map = {
        "Capacity": "measured_capacity_Ah",
        "Re": "internal_resistance",
        "Temperature_measured": "temperature",
        "Voltage_measured": "v_mean",
        "Current_measured": "i_mean",
        "Rct": "eis_Rct",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


    if "measured_capacity_Ah" in df.columns and "rated_capacity" not in df.columns:
        df["rated_capacity"] = df.groupby("battery_id")["measured_capacity_Ah"].transform("max")


    for col in SOH_FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    return df




def run_preprocessing(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = TARGET_SOH,
    save_scaler_path: str = None,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
) -> dict:
   
    if feature_cols is None:
        feature_cols = SOH_FEATURE_COLS

    df = clean_battery_data(df)
    train_df, val_df, test_df = split_by_battery(df, test_frac, val_frac)

    scaler = BatteryScaler(feature_cols)
    train_X = scaler.fit_transform(train_df)
    val_X   = scaler.transform(val_df)
    test_X  = scaler.transform(test_df)

    train_y = train_df[target_col].values
    val_y   = val_df[target_col].values
    test_y  = test_df[target_col].values

    if save_scaler_path:
        scaler.save(save_scaler_path)

    return {
        "train_X": train_X, "train_y": train_y,
        "val_X": val_X,   "val_y": val_y,
        "test_X": test_X,  "test_y": test_y,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_generator import generate_dataset

    df = generate_dataset(n_batteries_per_chemistry=2, n_cycles_per_battery=80)
    result = run_preprocessing(df)
    print("train_X shape:", result["train_X"].shape)
    print("test_y stats:", result["test_y"].mean(), result["test_y"].std())