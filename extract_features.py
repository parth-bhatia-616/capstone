import os
import scipy.io
import pandas as pd
import numpy as np
from pathlib import Path

# Paths based on your folder structure
RAW_DATA_DIR = Path("./data/raw/nasa_battery")
PROCESSED_DIR = Path("./data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def parse_nasa_mat(file_path):
   
    mat = scipy.io.loadmat(file_path)
    # Extract filename without extension (e.g., 'B0005')
    filename = Path(file_path).stem
    
    # Locate the core data structure
    core_key = [k for k in mat.keys() if not k.startswith('__')][0]
    cycles_data = mat[core_key][0, 0]['cycle'][0]

    soc_time_series = []
    soh_cycle_features = []

    for idx, cycle in enumerate(cycles_data):
        cycle_type = cycle['type'][0]
        cycle_id = idx + 1
        
        # 1. DISCHARGE CYCLES (Crucial for SoC and SoH)
        if cycle_type == 'discharge':
            data = cycle['data'][0, 0]
            
            # Extract time-series arrays
            v_disp = data['Voltage_measured'][0]
            i_disp = data['Current_measured'][0]
            t_disp = data['Temperature_measured'][0]
            time_disp = data['Time'][0]
            
            # Extract true capacity for SoH target variable
            try:
                capacity = data['Capacity'][0, 0]
            except (KeyError, IndexError):
                capacity = np.nan

            # Features for SoH (Cycle level summary metrics)
            if len(v_disp) > 0 and not np.isnan(capacity):
                # Internal resistance calculation approximation: delta_V / delta_I at start
                delta_v = v_disp[0] - v_disp[1] if len(v_disp) > 1 else 0
                delta_i = i_disp[0] - i_disp[1] if len(i_disp) > 1 else 1e-3
                internal_res = abs(delta_v / (delta_i if delta_i != 0 else 1e-3))

                soh_cycle_features.append({
                    'battery_id': filename,
                    'cycle': cycle_id,
                    'capacity': capacity,
                    'max_temp': np.max(t_disp),
                    'min_voltage': np.min(v_disp),
                    'internal_resistance': internal_res,
                    'discharge_duration': time_disp[-1] if len(time_disp) > 0 else 0
                })

            # Time-Series Features for SoC estimation
            if len(v_disp) > 0:
                # Coulomb Counting setup for target SoC calculation
                # SoC(t) = 1 - (Integral(I dt) / Total Capacity)
                dt = np.diff(time_disp, prepend=0)
                cumulative_ah = np.cumsum(i_disp * dt) / 3600.0  # Convert to Ah
                total_ah_discharged = cumulative_ah[-1] if len(cumulative_ah) > 0 else 1.0
                
                for t in range(len(v_disp)):
                    # Target SoC drops from 1.0 (100%) down to 0.0 during discharge
                    calculated_soc = 1.0 - (cumulative_ah[t] / (total_ah_discharged if total_ah_discharged != 0 else 1.0))
                    
                    soc_time_series.append({
                        'battery_id': filename,
                        'cycle': cycle_id,
                        'time': time_disp[t],
                        'voltage': v_disp[t],
                        'current': i_disp[t],
                        'temperature': t_disp[t],
                        'target_soc': max(0.0, min(1.0, calculated_soc)) # clip between 0 and 1
                    })

    return pd.DataFrame(soh_cycle_features), pd.DataFrame(soc_time_series)

def run_pipeline():
    print("Starting NASA Battery Data Processing Pipeline...")
    
    all_soh_data = []
    all_soc_data = []
    
    # We will focus on the highly structured standard aging batteries first
    target_files = [f"B000{i}.mat" for i in [5, 6, 7, 18]]
    
    for file_name in target_files:
        file_path = RAW_DATA_DIR / file_name
        if not file_path.exists():
            print(f"Warning: {file_name} not found in path. Skipping.")
            continue
            
        print(f"Extracting profiles from: {file_name}...")
        soh_df, soc_df = parse_nasa_mat(file_path)
        
        all_soh_data.append(soh_df)
        all_soc_data.append(soc_df)
    
    # Concatenate dataframes
    final_soh_df = pd.concat(all_soh_data, ignore_index=True)
    final_soc_df = pd.concat(all_soc_data, ignore_index=True)
    
    # Save outputs as highly optimized parquet files
    soh_output = PROCESSED_DIR / "soh_ml_features.parquet"
    soc_output = PROCESSED_DIR / "soc_time_series_features.parquet"
    
    final_soh_df.to_parquet(soh_output, index=False)
    final_soc_df.to_parquet(soc_output, index=False)
    
    print(" Processing Pipeline Complete!")
    print(f"Saved SoH Tabular Data ({final_soh_df.shape[0]} rows) -> {soh_output}")
    print(f" Saved SoC Sequence Data ({final_soc_df.shape[0]} rows) -> {soc_output}")

if __name__ == "__main__":
    run_pipeline()