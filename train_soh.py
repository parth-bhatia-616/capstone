import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
# Paths
PROCESSED_DIR = Path("./data/processed")
DATA_PATH = PROCESSED_DIR / "soh_ml_features.parquet"
# Model save directory
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)
def train_and_evaluate_soh():
    print("Loading processed SoH features...")
    df = pd.read_parquet(DATA_PATH)
    

    features = ['cycle', 'max_temp', 'min_voltage', 'internal_resistance', 'discharge_duration']
    target = 'capacity'
    
    batteries = df['battery_id'].unique()
    print(f"Batteries found for SoH training: {batteries}")
    

    plt.figure(figsize=(12, 6))
    

    for test_bat in batteries:
        print(f"Evaluating Unseen Test Target: {test_bat}")
        
        # Split data
        train_df = df[df['battery_id'] != test_bat]
        test_df = df[df['battery_id'] == test_bat]
        
        X_train, y_train = train_df[features], train_df[target]
        X_test, y_test = test_df[features], test_df[target]
        

        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        

        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)
        

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        
        print(f"Results for {test_bat} -> RMSE: {rmse:.4f} Ah | MAE: {mae:.4f} Ah")
        

        plt.plot(test_df['cycle'].values, y_test.values, label=f'{test_bat} Actual', linestyle='-')
        plt.plot(test_df['cycle'].values, preds, label=f'{test_bat} Predicted (RMSE: {rmse:.3f})', linestyle='--')

    plt.title('State of Health (SoH) Capacity Degradation Estimation via LightGBM')
    plt.xlabel('Cycle Number')
    plt.ylabel('Battery Capacity (Ah)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    

    chart_path = PROCESSED_DIR / "soh_predictions_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Performance comparison chart saved to: {chart_path}")

    print(" Training final SoH model on ALL batteries...")

    X_all = df[features]
    y_all = df[target]

    final_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )

    final_model.fit(X_all, y_all)

    # Save trained model
    model_path = MODELS_DIR / "soh_lightgbm.pkl"
    joblib.dump(final_model, model_path)

    print(f"Final SoH model saved to: {model_path}")
if __name__ == "__main__":
    train_and_evaluate_soh()