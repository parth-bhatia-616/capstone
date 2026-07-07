import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from train_soc import PhysicsInformedBMSNet

# Paths
PROCESSED_DIR = Path("./data/processed")
DATA_PATH = PROCESSED_DIR / "soc_time_series_features.parquet"
SOH_DATA_PATH = PROCESSED_DIR / "soh_ml_features.parquet"
device = torch.device("cpu")

def run_bms_realtime_test(target_battery='B0007', test_cycle=50):
    print(f"Initializing Verification Pipeline for Cell: {target_battery} (Cycle {test_cycle})...")
    
  
    df = pd.read_parquet(DATA_PATH)
    
    
    df['dt'] = df.groupby(['battery_id', 'cycle'])['time'].diff().fillna(0)
    df['amphour_spent'] = (df['current'] * df['dt']) / 3600.0
    df['cum_amphour_spent'] = df.groupby(['battery_id', 'cycle'])['amphour_spent'].cumsum()
    
    features_soc = ['voltage', 'current', 'temperature', 'cum_amphour_spent']
    
    # Isolate training set to match scaler profile exactly
    train_df = df[df['battery_id'].isin(['B0005', 'B0006'])].copy()
    test_df = df[df['battery_id'] == target_battery].copy()
    
    scaler = StandardScaler()
    scaler.fit(train_df[features_soc])
    
    # Isolate a single continuous discharge cycle for tracking test
    cycle_data = test_df[test_df['cycle'] == test_cycle].copy()
    if cycle_data.empty:
        test_cycle = test_df['cycle'].unique()[0]
        cycle_data = test_df[test_df['cycle'] == test_cycle].copy()
        
    raw_times = cycle_data['time'].values
    true_soc = cycle_data['target_soc'].values
    
    # Scale features dynamically
    scaled_features = scaler.transform(cycle_data[features_soc])
    
    # --- 2. RUN REAL-TIME STATE OF CHARGE (SoC) PREDICTION ---
    window_size = 35
    X_windows = []
    plot_times = []
    y_true_clipped = []
    
    for i in range(len(scaled_features) - window_size):
        X_windows.append(scaled_features[i : i + window_size])
        plot_times.append(raw_times[i + window_size])
        y_true_clipped.append(true_soc[i + window_size])
        
    X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32).to(device)
    
    soc_model = PhysicsInformedBMSNet(input_dim=4).to(device)
    soc_model.load_state_dict(torch.load(PROCESSED_DIR / "high_perf_soc_model.pt", map_location=device))
    soc_model.eval()
    
    with torch.no_grad():
        predicted_soc = soc_model(X_tensor).cpu().numpy()
        
    # --- 3. RUN STATE OF HEALTH (SoH) PREDICTION ---
    soh_df = pd.read_parquet(SOH_DATA_PATH)
    features_soh = ['cycle', 'max_temp', 'min_voltage', 'internal_resistance', 'discharge_duration']
    
    # Extract cycle-level summary parameters for this specific cycle
    cell_soh_data = soh_df[(soh_df['battery_id'] == target_battery) & (soh_df['cycle'] == test_cycle)]
    
    # Train a quick evaluation surrogate to pull predictions
    X_train_soh = soh_df[soh_df['battery_id'] != target_battery][features_soh]
    y_train_soh = soh_df[soh_df['battery_id'] != target_battery]['capacity']
    
    soh_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, verbosity=-1, random_state=42, num_threads=1, n_jobs=1)
    soh_model.fit(X_train_soh, y_train_soh)
    
    true_soh_cap = cell_soh_data['capacity'].values[0]
    predicted_soh_cap = soh_model.predict(cell_soh_data[features_soh])[0]
    
    # --- 4. GENERATE HIGH-CONTRAST BMS DASHBOARD GRAPH ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('black')
    
    # Plot 1: SoC Real-time tracking
    ax1.set_facecolor('black')
    ax1.plot(plot_times, y_true_clipped, label='True SoC Reference', color='white', linewidth=2.5)
    ax1.plot(plot_times, predicted_soc, label='Physics-Informed Estimator', color='#ff355e', linestyle='--', linewidth=2)
    ax1.set_title(f'Real-Time SoC Fuel Gauge Tracking (Cell {target_battery})', color='white', fontsize=12, pad=10)
    ax1.set_xlabel('Discharge Time (Seconds)', color='white')
    ax1.set_ylabel('State of Charge %', color='white')
    ax1.grid(True, color='gray', alpha=0.2, linestyle=':')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
    
    # Plot 2: SoH Capacity comparison bar chart for Cell Balancing Layer
    ax2.set_facecolor('black')
    bars = ax2.bar(['True Capacity', 'Model Prediction'], [true_soh_cap, predicted_soh_cap], color=['white', '#ff355e'], width=0.4)
    ax2.set_title(f'SoH Capacity Estimation Profile (Cycle {test_cycle})', color='white', fontsize=12, pad=10)
    ax2.set_ylabel('Max Available Capacity (Ah)', color='white')
    ax2.set_ylim(0, 2.2)
    ax2.grid(True, color='gray', alpha=0.1, linestyle=':')
    ax2.tick_params(colors='white')
    
    # Add numerical value tags on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.4f} Ah", ha='center', va='bottom', color='white', fontweight='bold')
        
    for ax in [ax1, ax2]:
        ax.spines['bottom'].color = 'white'
        ax.spines['left'].color = 'white'
        ax.spines['top'].color = 'none'
        ax.spines['right'].color = 'none'

    save_path = PROCESSED_DIR / "bms_performance_dashboard.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"saved to: {save_path}")

if __name__ == "__main__":
    run_bms_realtime_test()