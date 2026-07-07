
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
SYNTHETIC_DIR   = os.path.join(DATA_DIR, "synthetic")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR       = os.path.join(OUTPUTS_DIR, "plots")
REPORTS_DIR     = os.path.join(OUTPUTS_DIR, "reports")

# NASA dataset paths (place unzipped .mat files here)
NASA_BATTERY_DIR = os.path.join(RAW_DIR, "nasa_battery")
NASA_IGBT_DIR    = os.path.join(RAW_DIR, "nasa_igbt")


DATA_GEN = {
    "n_batteries_per_chemistry": 8,
    "n_cycles_per_battery": 400,
    "seed": 42,
    "chemistries": ["NMC", "LFP", "NCA", "LCO"],
    "form_factors": ["cylindrical", "prismatic", "pouch"],
    "aging_modes": [
        "cycle_aging", "calendar_aging", "fast_charge_stress",
        "deep_discharge_stress", "thermal_abuse", "mixed_usage",
    ],
}


FEATURE_ENG = {
    "savgol_window": 11,
    "savgol_poly": 3,
    "ic_bins": 200,
    "partial_soc_low": 0.1,
    "partial_soc_high": 0.8,
    "rolling_window": 10,
}


SPLIT = {
    "test_frac": 0.15,
    "val_frac": 0.15,
    "split_by": "battery_id",  
    "random_state": 42,
}


RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

GPR_PARAMS = {
    "alpha": 1e-3,     
    "n_restarts_optimizer": 5,
    "normalize_y": True,
}

CNN_LSTM_PARAMS = {
    "sequence_len": 50,      
    "cnn_filters": 64,
    "cnn_kernel": 3,
    "lstm_units": 128,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 100,
    "patience": 15,
}


EKF_PARAMS = {
    "Q_soc": 1e-7,
    "Q_vrc": 1e-8,
    "R_voltage": 1e-4,
    "initial_soc": 1.0,
    "adaptive_window": 30,
}


ENSEMBLE = {
    "kalman_weight": 0.25,
    "gpr_weight": 0.35,
    "rf_weight": 0.20,
    "xgb_weight": 0.20,
    "uncertainty_threshold": 3.0, 
}


GRADING = {
    "grade_A": 80.0,  
    "grade_B": 60.0,   

}


BALANCING = {
    "dv_passive_threshold": 0.010,   
    "dv_active_threshold": 0.020,    
    "temp_safe_max": 50.0,           
    "soh_protect_threshold": 60.0,   
}


NOISE = {
    "voltage_std": 0.005,    
    "current_std": 0.01,     
    "temperature_std": 0.5,  
    "soc_init_std": 0.02,  
}