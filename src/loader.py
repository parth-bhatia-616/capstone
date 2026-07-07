
import os
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path


def discover_battery_mat_files(nasa_battery_dir: str) -> list:
   
    import glob, os
    patterns = [
        os.path.join(nasa_battery_dir, "*.mat"),
        os.path.join(nasa_battery_dir, "randomized", "**", "*.mat"),
        os.path.join(nasa_battery_dir, "hirf",       "*.mat"),
    ]
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(pat, recursive=True))
    return sorted(set(all_files))



try:
    import scipy.io as sio
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("nope.")

warnings.filterwarnings("ignore")




def _extract_battery_cycles(mat_data: dict, battery_name: str) -> pd.DataFrame:

    rows = []
    try:
        battery = mat_data[battery_name]
        cycle_struct = battery["cycle"][0][0]
        cycles = cycle_struct[0]

        for i, cyc in enumerate(cycles):
            try:
                cycle_type = str(cyc["type"][0])
                ambient_temp = float(cyc["ambient_temperature"][0][0])
                data_struct = cyc["data"][0][0]

                row = {
                    "battery_id": battery_name,
                    "cycle_idx": i,
                    "type": cycle_type,
                    "ambient_temperature": ambient_temp,
                }

                if cycle_type == "discharge":
                    row["Voltage_measured"] = _safe_mean(data_struct, "Voltage_measured")
                    row["Current_measured"] = _safe_mean(data_struct, "Current_measured")
                    row["Temperature_measured"] = _safe_mean(data_struct, "Temperature_measured")
                    row["Capacity"] = _safe_scalar(data_struct, "Capacity")
                    row["Energy"] = _safe_scalar(data_struct, "Energy") if "Energy" in data_struct.dtype.names else np.nan

                elif cycle_type == "impedance":
                    row["Re"] = _safe_scalar(data_struct, "Re")
                    row["Rct"] = _safe_scalar(data_struct, "Rct")

                elif cycle_type == "charge":
                    row["Voltage_measured"] = _safe_mean(data_struct, "Voltage_measured")
                    row["Current_measured"] = _safe_mean(data_struct, "Current_measured")

                rows.append(row)
            except Exception:
                continue
    except Exception as e:
        warnings.warn(f"Could not parse {battery_name}: {e}")

    return pd.DataFrame(rows)


def _safe_mean(struct, field):
    try:
        val = struct[field][0][0].flatten()
        return float(np.nanmean(val))
    except Exception:
        return np.nan


def _safe_scalar(struct, field):
    try:
        val = struct[field][0][0]
        if hasattr(val, '__len__'):
            val = val.flatten()
            return float(val[0]) if len(val) > 0 else np.nan
        return float(val)
    except Exception:
        return np.nan


def load_nasa_battery_dataset(data_dir: str) -> pd.DataFrame:
    
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required: pip install scipy")

    mat_files = glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True)
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")

    print(f"Found {len(mat_files)} .mat files")
    all_dfs = []

    for fpath in sorted(mat_files):
        try:
            mat = sio.loadmat(fpath, squeeze_me=False, struct_as_record=False)
            battery_keys = [k for k in mat.keys() if k.startswith("B")]
            for key in battery_keys:
                df = _extract_battery_cycles(mat, key)
                if not df.empty:
                    all_dfs.append(df)
                    print(f"  Loaded {key}: {len(df)} cycles")
        except Exception as e:
            print(f"  Warning: could not load {fpath}: {e}")

    if not all_dfs:
        raise ValueError("No battery data could be extracted.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = _post_process_battery_df(combined)
    print(f"\nTotal: {len(combined)} cycles from {combined['battery_id'].nunique()} batteries")
    return combined


def _post_process_battery_df(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill impedance data into discharge cycles and compute SoH."""
    df = df.sort_values(["battery_id", "cycle_idx"]).reset_index(drop=True)

    # Forward-fill Re and Rct from impedance cycles into discharge cycles
    df["Re"] = df.groupby("battery_id")["Re"].transform(lambda x: x.ffill())
    df["Rct"] = df.groupby("battery_id")["Rct"].transform(lambda x: x.ffill())

    # Keep only discharge cycles for ML
    discharge_df = df[df["type"] == "discharge"].copy()

    # Rename cycle_idx to cycle count per battery
    discharge_df["cycle"] = discharge_df.groupby("battery_id").cumcount() + 1

    # Compute rated capacity (max per battery) and SoH
    rated = discharge_df.groupby("battery_id")["Capacity"].transform("max")
    discharge_df["rated_capacity"] = rated
    discharge_df["soh"] = (discharge_df["Capacity"] / rated.replace(0, np.nan)) * 100

    # Rename for consistency
    discharge_df = discharge_df.rename(columns={
        "Temperature_measured": "temperature",
        "Re": "internal_resistance",
    })

    keep_cols = [
        "battery_id", "cycle", "ambient_temperature", "temperature",
        "Capacity", "rated_capacity", "soh", "internal_resistance",
        "Rct", "Voltage_measured", "Current_measured",
    ]
    keep_cols = [c for c in keep_cols if c in discharge_df.columns]
    return discharge_df[keep_cols].reset_index(drop=True)


def load_igbt_smu_data(smu_dir: str) -> pd.DataFrame:
 
    rows = []
    part_dirs = sorted(glob.glob(os.path.join(smu_dir, "**", "Part *"), recursive=True))
    if not part_dirs:
        part_dirs = sorted(glob.glob(os.path.join(smu_dir, "Part *")))

    for part_dir in part_dirs:
        part_name = os.path.basename(part_dir)
        device_type = _infer_device_type(part_dir)
        row = {"part": part_name, "device_type": device_type}

        # Breakdown voltage
        bd_file = os.path.join(part_dir, "Breakdown.csv")
        if os.path.exists(bd_file):
            try:
                bd_df = pd.read_csv(bd_file, header=None, names=["voltage", "current"])
                bd_df = bd_df.apply(pd.to_numeric, errors="coerce").dropna()
                if not bd_df.empty:
                    row["breakdown_voltage_V"] = float(bd_df["voltage"].iloc[-1])
                    row["breakdown_current_A"] = float(bd_df["current"].iloc[-1])
            except Exception:
                pass

        # Leakage current
        lk_file = os.path.join(part_dir, "LeakageIV.csv")
        if os.path.exists(lk_file):
            try:
                lk_df = pd.read_csv(lk_file, header=None, names=["voltage", "current"])
                lk_df = lk_df.apply(pd.to_numeric, errors="coerce").dropna()
                if not lk_df.empty:
                    row["leakage_current_uA"] = float(lk_df["current"].max() * 1e6)
            except Exception:
                pass

        # Turn-on threshold
        ton_file = os.path.join(part_dir, "Turn On.csv")
        if os.path.exists(ton_file):
            try:
                ton_df = pd.read_csv(ton_file, header=None, names=["voltage", "current"])
                ton_df = ton_df.apply(pd.to_numeric, errors="coerce").dropna()
                if not ton_df.empty:
                    # Threshold = voltage at 1% of peak current
                    i_thresh = ton_df["current"].max() * 0.01
                    mask = ton_df["current"] >= i_thresh
                    if mask.any():
                        row["threshold_voltage_V"] = float(ton_df.loc[mask, "voltage"].iloc[0])
            except Exception:
                pass

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Loaded IGBT SMU data: {len(df)} devices from {smu_dir}")
    return df


def _infer_device_type(part_dir: str) -> str:
    """Infer IGBT vs MOSFET from parent directory name."""
    parent = str(Path(part_dir).parent.parent)
    if "IGBT" in parent:
        return "IGBT"
    elif "MOSFET" in parent:
        return "MOSFET"
    return "unknown"


def load_igbt_aging_mat(mat_dir: str) -> dict:
   
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required")

    mat_files = glob.glob(os.path.join(mat_dir, "**", "*.mat"), recursive=True)
    result = {}

    for fpath in sorted(mat_files):
        try:
            mat = sio.loadmat(fpath, squeeze_me=True)
            device_name = Path(fpath).stem
            arrays = {}
            for key in mat.keys():
                if key.startswith("_"):
                    continue
                val = mat[key]
                if isinstance(val, np.ndarray) and val.ndim <= 2:
                    arrays[key] = val.flatten()
            if arrays:
                result[device_name] = arrays
                print(f"  Loaded IGBT aging: {device_name} — keys: {list(arrays.keys())[:5]}")
        except Exception as e:
            print(f"  Warning: {fpath}: {e}")

    return result


def load_or_generate(
    synthetic_csv: str,
    nasa_dir: str = None,
    prefer_nasa: bool = True,
    n_batteries: int = 8,
    n_cycles: int = 400,
) -> pd.DataFrame:
   
    if prefer_nasa and nasa_dir and os.path.isdir(nasa_dir):
        try:
            df = load_nasa_battery_dataset(nasa_dir)
            print("Using NASA real battery data.")
            return df
        except Exception as e:
            print(f"NASA load failed ({e}), falling back to synthetic.")

    if os.path.exists(synthetic_csv):
        df = pd.read_csv(synthetic_csv)
        print(f"Loaded cached synthetic data: {len(df)} rows from {synthetic_csv}")
        return df

    # Generate fresh
    print("Generating synthetic battery data...")
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_generator import generate_dataset

    df = generate_dataset(
        n_batteries_per_chemistry=n_batteries,
        n_cycles_per_battery=n_cycles,
    )
    os.makedirs(os.path.dirname(synthetic_csv), exist_ok=True)
    df.to_csv(synthetic_csv, index=False)
    print(f"Saved synthetic data to {synthetic_csv}")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    # Quick test with synthetic fallback
    base = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base, "data", "synthetic", "battery_aging_dataset.csv")
    df = load_or_generate(csv_path, prefer_nasa=False, n_batteries=2, n_cycles=50)
    print(df.head())
    print(f"Columns: {list(df.columns)}")