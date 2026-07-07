import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from train_soc import PhysicsInformedBMSNet

# Paths & Setup
PROCESSED_DIR = Path("./data/processed")
SOC_DATA_PATH = PROCESSED_DIR / "soc_time_series_features.parquet"
SOH_DATA_PATH = PROCESSED_DIR / "soh_ml_features.parquet"
device = torch.device("cpu")

# Apply Global Custom Premium Dark Theme UI Aesthetics
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'figure.facecolor': 'black',
    'axes.facecolor': 'black',
    'grid.color': '#333333',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'savefig.facecolor': 'black'
})
BRAND_COLOR = '#ff355e' # Crimson Pink Accent

def load_and_process_data():
    print("⏳ Extracting time-series sequences and lifecycle metrics...")
    soh_df = pd.read_parquet(SOH_DATA_PATH)
    df_soc = pd.read_parquet(SOC_DATA_PATH)
    
    # Recompute physics constraint feature
    df_soc['dt'] = df_soc.groupby(['battery_id', 'cycle'])['time'].diff().fillna(0)
    df_soc['amphour_spent'] = (df_soc['current'] * df_soc['dt']) / 3600.0
    df_soc['cum_amphour_spent'] = df_soc.groupby(['battery_id', 'cycle'])['amphour_spent'].cumsum()
    
    return soh_df, df_soc

def generate_soh_plots(soh_df):
    print("Generating State of Health (SoH) Metrology Graphics...")
    features_soh = ['cycle', 'max_temp', 'min_voltage', 'internal_resistance', 'discharge_duration']
    
    # Leave-One-Out setup for evaluation validation
    train_mask = soh_df['battery_id'].isin(['B0005', 'B0006'])
    test_mask = soh_df['battery_id'] == 'B0007'
    
    X_train, y_train = soh_df[train_mask][features_soh], soh_df[train_mask]['capacity']
    X_test, y_test = soh_df[test_mask][features_soh], soh_df[test_mask]['capacity']
    cycles_test = sorted(soh_df[test_mask]['cycle'].values)
    
    model_soh = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, verbosity=-1, random_state=42, num_threads=1, n_jobs=1)
    model_soh.fit(X_train, y_train)
    y_pred = model_soh.predict(X_test)
    residuals = y_test - y_pred

    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. Feature Importance Plot
    importances = model_soh.feature_importances_
    indices = np.argsort(importances)
    axs[0, 0].barh([features_soh[i] for i in indices], importances[indices], color=BRAND_COLOR, height=0.5)
    axs[0, 0].set_title("SoH Model (LightGBM) Feature Importance", fontsize=12, pad=10)
    axs[0, 0].set_xlabel("Relative Importance Weight (Gain)")

    # 2. Predicted vs Actual Capacity
    axs[0, 1].scatter(y_test, y_pred, color=BRAND_COLOR, alpha=0.7, edgecolors='black')
    axs[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'w--', lw=2)
    axs[0, 1].set_title("Predicted vs. Actual Capacity", fontsize=12, pad=10)
    axs[0, 1].set_xlabel("Actual Capacity (Ah)")
    axs[0, 1].set_ylabel("Predicted Capacity (Ah)")

    # 3. Capacity Fade Curve
    axs[1, 0].plot(cycles_test, y_test, label="True Degraded Curve", color='white', lw=2.5)
    axs[1, 0].plot(cycles_test, y_pred, label="LightGBM Estimate", color=BRAND_COLOR, linestyle='--', lw=2)
    axs[1, 0].set_title("Continuous Lifecycle Capacity Fade Tracking (Unseen Cell B0007)", fontsize=12, pad=10)
    axs[1, 0].set_xlabel("Operational Life Cycle Count")
    axs[1, 0].set_ylabel("Capacity (Ah)")
    axs[1, 0].legend()

    # 4. Residual Error Plot
    axs[1, 1].scatter(y_pred, residuals, color=BRAND_COLOR, alpha=0.6, edgecolors='black')
    axs[1, 1].axhline(y=0, color='white', linestyle='--', lw=1.5)
    axs[1, 1].set_title("Residual Variance Analysis Plot", fontsize=12, pad=10)
    axs[1, 1].set_xlabel("Predicted Capacity Values (Ah)")
    axs[1, 1].set_ylabel("Residual Errors (Ah)")

    # 5. Internal Resistance vs Cycle (Raw Physical Aging Indicator)
    axs[2, 0].plot(cycles_test, X_test['internal_resistance'], color=BRAND_COLOR, lw=2)
    axs[2, 0].set_title("Internal Ohmic Resistance Degradation Trend", fontsize=12, pad=10)
    axs[2, 0].set_xlabel("Operational Life Cycle Count")
    axs[2, 0].set_ylabel("Resistance ($\Omega$)")

    # Clean empty subplot frame
    fig.delaxes(axs[2, 1])
    
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "soh_comprehensive_metrics.png", dpi=300, facecolor='black')
    plt.close()

def generate_soc_plots(df_soc):
    print(" Extracting Deep State of Charge (SoC) Neural Performance Analysis...")
    features_soc = ['voltage', 'current', 'temperature', 'cum_amphour_spent']
    
    train_df = df_soc[df_soc['battery_id'].isin(['B0005', 'B0006'])].copy()
    test_df = df_soc[df_soc['battery_id'] == 'B0007'].copy()
    
    scaler = StandardScaler()
    scaler.fit(train_df[features_soc])
    
    # Isolate specific mid-life cycle group 50 for time-series extraction
    cycle_50_df = test_df[test_df['cycle'] == 50].copy()
    raw_voltage = cycle_50_df['voltage'].values
    
    test_df[features_soc] = scaler.transform(test_df[features_soc])
    
    window_size = 35
    X_test, y_true, plot_times = [], [], []
    
    # Slicing complete profile for global statistical matrices
    for _, group in test_df.groupby(['cycle']):
        feat_arr = group[features_soc].values
        targ_arr = group['target_soc'].values
        time_arr = group['time'].values
        if len(feat_arr) > window_size:
            for i in range(len(feat_arr) - window_size):
                X_test.append(feat_arr[i : i + window_size])
                y_true.append(targ_arr[i + window_size])
                # Only log timestamps belonging to cycle 50 for the temporal error plot
                if group['cycle'].iloc[0] == 50:
                    plot_times.append(time_arr[i + window_size])
                    
    X_test = np.array(X_test)
    y_true = np.array(y_true)
    
    soc_model = PhysicsInformedBMSNet(input_dim=4).to(device)
    soc_model.load_state_dict(torch.load(PROCESSED_DIR / "high_perf_soc_model.pt", map_location=device))
    soc_model.eval()
    
    # Chunked evaluation loop to handle Apple Silicon MPS memory boundary bounds cleanly
    batch_size, preds_list = 1024, []
    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            batch_x = torch.tensor(X_test[start:start+batch_size], dtype=torch.float32).to(device)
            p = soc_model(batch_x).cpu().numpy()
            if p.ndim == 0: p = np.expand_dims(p, axis=0)
            preds_list.append(p)
    preds = np.concatenate(preds_list, axis=0)
    
    # Isolate targets relating specifically to cycle 50 layout visualizations
    c50_len = len(plot_times)
    y_true_50 = y_true[:c50_len]
    preds_50 = preds[:c50_len]
    voltage_50 = raw_voltage[window_size:]
    
    errors_global = preds - y_true
    abs_errors_50 = np.abs(preds_50 - y_true_50)

    fig, axs = plt.subplots(3, 2, figsize=(16, 18))

    # 1. Actual SoC vs Predicted SoC
    axs[0, 0].plot(plot_times, y_true_50, label='True SoC Reference', color='white', lw=2.5)
    axs[0, 0].plot(plot_times, preds_50, label='Conv-BiLSTM Estimate', color=BRAND_COLOR, linestyle='--', lw=2)
    axs[0, 0].set_title("Real-Time SoC Fuel Gauge Validation (Cycle 50)", fontsize=12, pad=10)
    axs[0, 0].set_xlabel("Discharge Duration Time (Seconds)")
    axs[0, 0].set_ylabel("State of Charge %")
    axs[0, 0].legend()

    # 2. Error Over Time
    axs[0, 1].plot(plot_times, abs_errors_50 * 100, color=BRAND_COLOR, lw=2)
    axs[0, 1].set_title("Absolute Instantaneous Drift Over Time", fontsize=12, pad=10)
    axs[0, 1].set_xlabel("Discharge Duration Time (Seconds)")
    axs[0, 1].set_ylabel("Absolute Error Deviation %")

    # 3. Error Distribution Histogram
    axs[1, 0].hist(np.abs(errors_global) * 100, bins=50, color=BRAND_COLOR, alpha=0.8, edgecolor='black')
    axs[1, 0].set_title("Global SoC Absolute Error Distribution", fontsize=12, pad=10)
    axs[1, 0].set_xlabel("Absolute Deviation Window Percentage (%)")
    axs[1, 0].set_ylabel("Frequency Count")

    # 4. Residual Distribution Density
    sns.kdeplot(errors_global * 100, ax=axs[1, 1], color=BRAND_COLOR, lw=2, fill=True, alpha=0.2)
    axs[1, 1].axvline(x=0, color='white', linestyle='--', lw=1.5)
    axs[1, 1].set_title("Residual Error Normal Density Curve Profile", fontsize=12, pad=10)
    axs[1, 1].set_xlabel("Residual Delta Error Score (%)")

    # 5. Scatter Plot Actual vs Predicted SoC
    axs[2, 0].scatter(y_true, preds, color=BRAND_COLOR, alpha=0.1, s=5, rasterized=True)
    axs[2, 0].plot([0, 1], [0, 1], 'w--', lw=1.5)
    axs[2, 0].set_title("Global Linear Regression Fit Correlate", fontsize=12, pad=10)
    axs[2, 0].set_xlabel("Laboratory Ground Truth SoC")
    axs[2, 0].set_ylabel("Neural Output Predictions")

    # 6. Voltage vs SoC Signature Line
    axs[2, 1].plot(preds_50 * 100, voltage_50, color=BRAND_COLOR, lw=2.5)
    axs[2, 1].set_title("Electrochemical Open Circuit Voltage (OCV) vs. SoC Signature", fontsize=12, pad=10)
    axs[2, 1].set_xlabel("Estimated State of Charge %")
    axs[2, 1].set_ylabel("Terminal Operational Voltage (V)")

    for row in axs:
        for ax in row:
            ax.grid(True, linestyle=':', alpha=0.2)
            ax.spines['top'].color = 'none'
            ax.spines['right'].color = 'none'
            ax.spines['bottom'].color = 'white'
            ax.spines['left'].color = 'white'

    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "soc_comprehensive_metrics.png", dpi=300, facecolor='black')
    plt.close()

if __name__ == "__main__":
    soh_df, df_soc = load_and_process_data()
    generate_soh_plots(soh_df)
    generate_soc_plots(df_soc)
    print("\n🚀 Verification Complete! Dual compilation matrix output saved inside: ./data/processed/")