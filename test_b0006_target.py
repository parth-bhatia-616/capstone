import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
from train_soc import PhysicsInformedBMSNet

# Paths
PROCESSED_DIR = Path("./data/processed")
SOC_DATA_PATH = PROCESSED_DIR / "soc_time_series_features.parquet"
SOH_DATA_PATH = PROCESSED_DIR / "soh_ml_features.parquet"
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def run_target_b0006_test():
    print("Re-configuring validation matrix: Setting Target Cell -> B0006 (Fold 3)...")
    
    soh_df = pd.read_parquet(SOH_DATA_PATH)
    features_soh = ['cycle', 'max_temp', 'min_voltage', 'internal_resistance', 'discharge_duration']
    
    # Train on B0005 + B0007, Test on B0006
    train_soh = soh_df[soh_df['battery_id'].isin(['B0005', 'B0007'])]
    test_soh = soh_df[soh_df['battery_id'] == 'B0006']
    
    soh_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, verbosity=-1, random_state=42)
    soh_model.fit(train_soh[features_soh], train_soh['capacity'])
    
    soh_preds = soh_model.predict(test_soh[features_soh])
    soh_mae = mean_absolute_error(test_soh['capacity'], soh_preds)
    
    print("SoH Fold 3 Performance (Target: B0006) ---")
    print(f"🔹 Absolute Capacity Drift Error (MAE): {soh_mae:.4f} Ah")
    
    df_soc = pd.read_parquet(SOC_DATA_PATH)
    df_soc['dt'] = df_soc.groupby(['battery_id', 'cycle'])['time'].diff().fillna(0)
    df_soc['amphour_spent'] = (df_soc['current'] * df_soc['dt']) / 3600.0
    df_soc['cum_amphour_spent'] = df_soc.groupby(['battery_id', 'cycle'])['amphour_spent'].cumsum()
    
    features_soc = ['voltage', 'current', 'temperature', 'cum_amphour_spent']
    
    train_df = df_soc[df_soc['battery_id'].isin(['B0005', 'B0007'])].copy()
    test_df = df_soc[df_soc['battery_id'] == 'B0006'].copy()
    
    scaler = StandardScaler()
    scaler.fit(train_df[features_soc])
    test_df[features_soc] = scaler.transform(test_df[features_soc])
    
    window_size = 35
    X_test, y_true, plot_times = [], [], []
    
    for _, group in test_df.groupby(['cycle']):
        feat_arr = group[features_soc].values
        targ_arr = group['target_soc'].values
        time_arr = group['time'].values
        if len(feat_arr) > window_size:
            for i in range(len(feat_arr) - window_size):
                X_test.append(feat_arr[i : i + window_size])
                y_true.append(targ_arr[i + window_size])
                if group['cycle'].iloc[0] == 50:
                    plot_times.append(time_arr[i + window_size])
                    
    X_test = np.array(X_test)
    y_true = np.array(y_true)
    
    soc_model = PhysicsInformedBMSNet(input_dim=4).to(device)
    soc_model.load_state_dict(torch.load(PROCESSED_DIR / "high_perf_soc_model.pt", map_location=device))
    soc_model.eval()
    
    batch_size, preds_list = 1024, []
    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            batch_x = torch.tensor(X_test[start:start+batch_size], dtype=torch.float32).to(device)
            p = soc_model(batch_x).cpu().numpy()
            if p.ndim == 0: p = np.expand_dims(p, axis=0)
            preds_list.append(p)
            
    preds = np.concatenate(preds_list, axis=0)
    
    c50_len = len(plot_times)
    y_true_50 = y_true[:c50_len]
    preds_50 = preds[:c50_len]
    
    soc_r2 = r2_score(y_true, preds)
    soc_mae = mean_absolute_error(y_true, preds)
    
    print("SoC Fold 3 Performance (Target: B0006) ---")
    print(f"Variance Explained (R² Score)     : {soc_r2 * 100:.2f}%")
    print(f"Mean Absolute Tracking Error (MAE): {soc_mae * 100:.3f}%")
    print("-" * 50)
    
    plt.figure(figsize=(10, 5))
    plt.gcf().patch.set_facecolor('black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    plt.plot(plot_times, y_true_50, label='True SoC (B0006 Reference)', color='white', linewidth=2.5)
    plt.plot(plot_times, preds_50, label='Conv-BiLSTM Fold 3 Estimate', color='#ff355e', linestyle='--', linewidth=2)
    
    plt.title('Cross-Battery SoC Evaluation Vector — Target: Battery B0006 (Cycle 50)', color='white', fontsize=12, pad=12)
    plt.xlabel('Discharge Time (Seconds)', color='white')
    plt.ylabel('State of Charge (Percentage)', color='white')
    plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
    plt.grid(True, color='#333333', alpha=0.5, linestyle=':')
    
    ax.tick_params(colors='white')
    ax.spines['bottom'].color = 'white'
    ax.spines['left'].color = 'white'
    ax.spines['top'].color = 'none'
    ax.spines['right'].color = 'none'
    
    save_path = PROCESSED_DIR / "b0006_cross_validation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"\nTarget shifted alignment chart rendered cleanly to: {save_path}")

if __name__ == "__main__":
    run_target_b0006_test()