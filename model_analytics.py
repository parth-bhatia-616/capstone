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
device = torch.device("cpu")

def analyze_bms_performance():
    print("Extracting Deep Diagnostics & Metrology Metrics...\n")
  
    soh_df = pd.read_parquet(SOH_DATA_PATH)
    features_soh = ['cycle', 'max_temp', 'min_voltage', 'internal_resistance', 'discharge_duration']
    
    X_soh = soh_df[features_soh]
    y_soh = soh_df['capacity']
    
    soh_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, verbosity=-1, random_state=42, num_threads=1, n_jobs=1)
    soh_model.fit(X_soh, y_soh)
    
    importances = soh_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("--- 🔋 State of Health (SoH) Feature Dominance ---")
    for f in range(X_soh.shape[1]):
        print(f"{f + 1}. Feature [{features_soh[indices[f]]:<20}] -> Relative Importance Score: {importances[indices[f]]}")
    print("-" * 50 + "\n")


    df_soc = pd.read_parquet(SOC_DATA_PATH)
    df_soc['dt'] = df_soc.groupby(['battery_id', 'cycle'])['time'].diff().fillna(0)
    df_soc['amphour_spent'] = (df_soc['current'] * df_soc['dt']) / 3600.0
    df_soc['cum_amphour_spent'] = df_soc.groupby(['battery_id', 'cycle'])['amphour_spent'].cumsum()
    
    features_soc = ['voltage', 'current', 'temperature', 'cum_amphour_spent']
    
    train_df = df_soc[df_soc['battery_id'].isin(['B0005', 'B0006'])].copy()
    test_df = df_soc[df_soc['battery_id'] == 'B0007'].copy()
    
    scaler = StandardScaler()
    scaler.fit(train_df[features_soc])
    test_df[features_soc] = scaler.transform(test_df[features_soc])
    
    window_size = 35
    X_test, y_true = [], []
    
    for _, group in test_df.groupby(['cycle']):
        feat_arr = group[features_soc].values
        targ_arr = group['target_soc'].values
        if len(feat_arr) > window_size:
            for i in range(len(feat_arr) - window_size):
                X_test.append(feat_arr[i : i + window_size])
                y_true.append(targ_arr[i + window_size])
                
    X_test = np.array(X_test)
    y_true = np.array(y_true)
    

    soc_model = PhysicsInformedBMSNet(input_dim=4).to(device)
    soc_model.load_state_dict(torch.load(PROCESSED_DIR / "high_perf_soc_model.pt", map_location=device))
    soc_model.eval()
    

    batch_size = 1024
    preds_list = []
    
    with torch.no_grad():
        for start_idx in range(0, len(X_test), batch_size):
            end_idx = start_idx + batch_size
            batch_x = torch.tensor(X_test[start_idx:end_idx], dtype=torch.float32).to(device)
            batch_preds = soc_model(batch_x).cpu().numpy()
            

            if batch_preds.ndim == 0:
                batch_preds = np.expand_dims(batch_preds, axis=0)
            preds_list.append(batch_preds)
            
    preds = np.concatenate(preds_list, axis=0)
        

    soc_r2 = r2_score(y_true, preds)
    soc_rmse = np.sqrt(mean_squared_error(y_true, preds))
    soc_mae = mean_absolute_error(y_true, preds)
    
    absolute_errors = np.abs(preds - y_true)
    max_error = np.max(absolute_errors)
    std_error = np.std(absolute_errors)
    
    print(" State of Charge (SoC) Statistical Metrics ---")
    print(f"Variance Explained (R² Score)     : {soc_r2 * 100:.2f}%")
    print(f" Root Mean Squared Error (RMSE)    : {soc_rmse * 100:.3f}%")
    print(f" Mean Absolute Error (MAE)         : {soc_mae * 100:.3f}%")
    print(f" Peak Operational Error (Max Dev)  : {max_error * 100:.2f}%")
    print(f" Error Deviation Variance (±σ)     : {std_error * 100:.3f}%")
    print("-" * 50)
    

    plt.figure(figsize=(8, 5))
    plt.gcf().patch.set_facecolor('black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    plt.hist(absolute_errors * 100, bins=50, color='#ff355e', alpha=0.8, edgecolor='black')
    plt.title('SoC Model Absolute Error Variance Distribution', color='white', fontsize=12, pad=10)
    plt.xlabel('Absolute Error Percentage (%)', color='white')
    plt.ylabel('Frequency (Data Window Points)', color='white')
    plt.grid(True, color='gray', alpha=0.15, linestyle=':')
    
    ax.tick_params(colors='white')
    ax.spines['bottom'].color = 'white'
    ax.spines['left'].color = 'white'
    ax.spines['top'].color = 'none'
    ax.spines['right'].color = 'none'
    
    hist_path = PROCESSED_DIR / "soc_error_distribution.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())
    print(f"Error distribution plot compiled and saved to: {hist_path}")

if __name__ == "__main__":
    analyze_bms_performance()