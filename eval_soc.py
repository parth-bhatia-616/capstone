import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
from train_soc import PhysicsInformedBMSNet

PROCESSED_DIR = Path("./data/processed")
DATA_PATH = PROCESSED_DIR / "soc_time_series_features.parquet"
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def evaluate_and_plot_soc():
    print("Loading confirmation loop...")
    df = pd.read_parquet(DATA_PATH)
    
    df['dt'] = df.groupby(['battery_id', 'cycle'])['time'].diff().fillna(0)
    df['amphour_spent'] = (df['current'] * df['dt']) / 3600.0
    df['cum_amphour_spent'] = df.groupby(['battery_id', 'cycle'])['amphour_spent'].cumsum()
    
    features = ['voltage', 'current', 'temperature', 'cum_amphour_spent']
    
    train_df = df[df['battery_id'].isin(['B0005', 'B0006'])].copy()
    test_df = df[df['battery_id'] == 'B0007'].copy()
    
    scaler = StandardScaler()
    scaler.fit(train_df[features])
    test_df[features] = scaler.transform(test_df[features])
    
    window_size = 35
    X_test, y_test, plot_times = [], [], []
    
    target_cycle = 50 
    cycle_group = test_df[test_df['cycle'] == target_cycle]
    if cycle_group.empty:
        target_cycle = test_df['cycle'].unique()[0]
        cycle_group = test_df[test_df['cycle'] == target_cycle]
        
    feat_arr = cycle_group[features].values
    targ_arr = cycle_group['target_soc'].values
    time_arr = cycle_group['time'].values
    
    for i in range(len(feat_arr) - window_size):
        X_test.append(feat_arr[i : i + window_size])
        y_test.append(targ_arr[i + window_size])
        plot_times.append(time_arr[i + window_size])
        
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)
    
    model = PhysicsInformedBMSNet(input_dim=4).to(device)
    model.load_state_dict(torch.load(PROCESSED_DIR / "high_perf_soc_model.pt", map_location=device))
    model.eval()
    
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()
        
    plt.figure(figsize=(12, 6))
    plt.plot(plot_times, y_test, label='True SoC Reference', color='white', linewidth=2.5)
    plt.plot(plot_times, preds, label='Physics-Informed Estimator', color='#ff355e', linestyle='--', linewidth=2)
    
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.gcf().patch.set_facecolor('black')
    ax.spines['bottom'].color = 'white'
    ax.spines['left'].color = 'white'
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    
    plt.title(f'Optimized State of Charge (SoC) Tracking — Battery B0007 (Cycle {target_cycle})', color='white', fontsize=14, pad=15)
    plt.xlabel('Discharge Time Steps (Seconds)')
    plt.ylabel('State of Charge (Percentage)')
    plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
    plt.grid(True, color='gray', alpha=0.2, linestyle=':')
    
    save_path = PROCESSED_DIR / "soc_tracking_performance.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())
    print(f"Redrawn track output pinned flawlessly to destination: {save_path}")

if __name__ == "__main__":
    evaluate_and_plot_soc()