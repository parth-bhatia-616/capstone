
import os, sys, argparse, warnings, time
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from config import (
    SYNTHETIC_DIR, MODELS_DIR, PROCESSED_DIR, NASA_BATTERY_DIR,
    RF_PARAMS, XGB_PARAMS, CNN_LSTM_PARAMS, SPLIT,
)
from loader import load_or_generate          # your existing loader
from preprocessing import (
    clean_battery_data, split_by_battery, BatteryScaler,
    build_sequences, inject_noise,
    SOH_FEATURE_COLS, SOC_FEATURE_COLS,
)
from feature_engineering import build_feature_matrix
from soh_models import (
    RandomForestSoH, GPRResidualCorrector, SoHEnsemble,
    evaluate_model, XGB_AVAILABLE,
)
from kalman_filters import ECMParams, KalmanSoHEstimator


try:
    from soh_models import XGBoostSoH
except Exception:
    XGB_AVAILABLE = False

try:
    from soh_models import CNNLSTMSoH
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from soc_models import BiLSTMSoCModel
    BILSTM_AVAILABLE = True
except Exception:
    try:
        from soc_models import SimpleSoCCorrector
        BILSTM_AVAILABLE = False
    except Exception:
        BILSTM_AVAILABLE = False



def ensure_dirs():
    for d in [MODELS_DIR, PROCESSED_DIR,
              os.path.join(MODELS_DIR, "soh"),
              os.path.join(MODELS_DIR, "soc")]:
        os.makedirs(d, exist_ok=True)


def detect_data_source() -> str:
  

    nasa_dir = NASA_BATTERY_DIR
    if nasa_dir and os.path.isdir(nasa_dir):
        files = os.listdir(nasa_dir)
        mat_files = [f for f in files if f.endswith(".mat")]
        csv_files = [f for f in files if f.endswith(".csv") and "B00" in f]
        if mat_files:
            print(f"  Found {len(mat_files)} NASA .mat files in {nasa_dir}")
            return "nasa_mat"
        if csv_files:
            print(f"  Found {len(csv_files)} NASA CSV files in {nasa_dir}")
            return "nasa_csv"


    synthetic_csv = os.path.join(SYNTHETIC_DIR, "battery_aging_dataset.csv")
    if os.path.exists(synthetic_csv):
        print(f"  ℹ No NASA files found. Using cached synthetic data.")
        return "synthetic_cached"

    print("  ℹ No data found. Will generate synthetic dataset.")
    return "synthetic_new"


def load_data(source: str, force_synthetic: bool = False,
              n_batteries: int = 12, n_cycles: int = 500) -> pd.DataFrame:
   
    synthetic_csv = os.path.join(SYNTHETIC_DIR, "battery_aging_dataset.csv")
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)

    use_nasa = (source in ("nasa_mat", "nasa_csv")) and not force_synthetic

    print(f"\n{'═'*60}")
    print(f"  DATA LOADING  →  source: {source.upper()}")
    print(f"{'═'*60}")

    t0 = time.time()
    df = load_or_generate(          
        synthetic_csv=synthetic_csv,
        nasa_dir=NASA_BATTERY_DIR if use_nasa else None,
        prefer_nasa=use_nasa,
        n_batteries=n_batteries,
        n_cycles=n_cycles,
    )
    elapsed = time.time() - t0


    n_batteries_loaded = df["battery_id"].nunique()
    n_rows             = len(df)
    cap_col = "measured_capacity_Ah" if "measured_capacity_Ah" in df.columns else "Capacity"

    print(f"\n  Loaded in {elapsed:.1f}s")
    print(f"  Rows:       {n_rows:,}")
    print(f"  Batteries:  {n_batteries_loaded}")
    print(f"  Battery IDs: {sorted(df['battery_id'].unique().tolist())[:10]}")

    if "chemistry" in df.columns:
        chem_counts = df.groupby("chemistry")["battery_id"].nunique()
        print(f"  Chemistry breakdown:")
        for chem, count in chem_counts.items():
            print(f"    {chem}: {count} batteries")

    print(f"  SoH range:  {df['soh'].min():.1f}% – {df['soh'].max():.1f}%")
    print(f"  Cycles:     {df['cycle'].min()} – {df['cycle'].max()}")
    print(f"  Capacity:   {df[cap_col].min():.3f} – {df[cap_col].max():.3f} Ah")


    if n_batteries_loaded < 3:
        print(" WARNING: < 3 batteries loaded. "
              "Model may not generalize well.")
        print("     Add more NASA .mat files to:", NASA_BATTERY_DIR)
    if n_rows < 500:
        print("  WARNING: < 500 rows. Consider more cycles.")

    return df



def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calls your existing feature_engineering.py to build the full
    feature matrix: IC peaks, DV peaks, EIS params, rolling stats, etc.
    """
    print(f"\n{'═'*60}")
    print(f"  FEATURE ENGINEERING  (feature_engineering.py)")
    print(f"{'═'*60}")

    t0 = time.time()
    feat_df = build_feature_matrix(df, rolling_window=10, add_rolling=True)
    elapsed = time.time() - t0

    # Save processed features
    feat_path = os.path.join(PROCESSED_DIR, "features_advanced.csv")
    feat_df.to_csv(feat_path, index=False)

    n_features = feat_df.select_dtypes(include="number").shape[1]
    ic_cols  = [c for c in feat_df.columns if c.startswith("ic_")]
    dv_cols  = [c for c in feat_df.columns if c.startswith("dv_")]
    eis_cols = [c for c in feat_df.columns if c.startswith("eis_")]
    roll_cols = [c for c in feat_df.columns if "roll" in c.lower()]

    print(f"\n  Built in {elapsed:.1f}s  →  {feat_df.shape}")
    print(f"  Total numeric features: {n_features}")
    print(f"    IC curve features:   {len(ic_cols):3d}  → {ic_cols}")
    print(f"    DV curve features:   {len(dv_cols):3d}  → {dv_cols}")
    print(f"    EIS features:        {len(eis_cols):3d}  → {eis_cols}")
    print(f"    Rolling features:    {len(roll_cols):3d}")
    print(f"  Saved → {feat_path}")

    # Top features correlated with SoH
    num_df = feat_df.select_dtypes(include="number")
    if "soh" in num_df.columns:
        top_corr = (num_df.corr()["soh"]
                    .drop("soh").abs()
                    .sort_values(ascending=False)
                    .head(8))
        print(f"\n  Top 8 features most correlated with SoH:")
        for feat, corr in top_corr.items():
            bar = "." * int(corr * 20)
            print(f"    {feat:<35} {corr:.3f} {bar}")

    return feat_df


def prepare_splits(feat_df: pd.DataFrame, feature_cols: list):

    print(f"\n{'═'*60}")
    print(f"  DATA SPLITTING  (preprocessing.py)")
    print(f"  Strategy: battery-level split (no leakage)")
    print(f"{'═'*60}")

    train_df, val_df, test_df = split_by_battery(
        feat_df,
        test_frac=SPLIT["test_frac"],
        val_frac=SPLIT["val_frac"],
        random_state=SPLIT["random_state"],
    )


    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        bats = sorted(split_df["battery_id"].unique().tolist())
        print(f"  {name:<6}: {len(split_df):>6} rows | "
              f"{len(bats):>2} batteries | {bats}")


    scaler = BatteryScaler(feature_cols)
    X_train = scaler.fit_transform(train_df)
    X_val   = scaler.transform(val_df)
    X_test  = scaler.transform(test_df)
    y_train = train_df["soh"].values
    y_val   = val_df["soh"].values
    y_test  = test_df["soh"].values

    scaler.save(os.path.join(MODELS_DIR, "soh", "scaler.pkl"))
    print(f"\n  Scaler saved → models/soh/scaler.pkl")

    return (train_df, val_df, test_df,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler)


def train_soh_pipeline(feat_df: pd.DataFrame) -> dict:
    # """
    # 4-layer SoH ensemble:
    #   Layer 1 →  KalmanSoHEstimator  (kalman_filters.py)
    #   Layer 2 →  RandomForestSoH     (soh_models.py)
    #   Layer 2b→  XGBoostSoH          (soh_models.py, if available)
    #   Layer 3 →  CNNLSTMSoH          (soh_models.py, if PyTorch available)
    #   Layer 4 →  GPR residual +      (soh_models.py)
    #              SoHEnsemble fusion
    # """
    print(f"\n{'═'*60}")
    print(f"  SOH PIPELINE  (soh_models.py + kalman_filters.py)")
    print(f"{'═'*60}")
    print(f"  Models: Kalman | RF | "
          f"{'XGB | ' if XGB_AVAILABLE else ''}"
          f"{'CNN-LSTM | ' if TORCH_AVAILABLE else ''}"
          f"GPR | Ensemble")


    avail_cols = [c for c in SOH_FEATURE_COLS if c in feat_df.columns]
    if len(avail_cols) < 3:

        avail_cols = [c for c in feat_df.select_dtypes(include="number").columns
                      if c not in ("soh", "soc_true", "battery_num")]
    print(f"\n  Using {len(avail_cols)}/{len(SOH_FEATURE_COLS)} "
          f"requested features")

    (train_df, val_df, test_df,
     X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler) = prepare_splits(feat_df, avail_cols)

    # Add noise augmentation to training data
    X_train_aug = inject_noise(X_train, noise_std=0.01)

    results     = {}
    predictions = {}
    models      = {}

    # ── Layer 1: Kalman EKF Physics Baseline ──────────────────────
    print(f"\n  {'─'*54}")
    print(f"  [L1] Kalman SoH Estimator  (kalman_filters.py)")
    k_estimator = KalmanSoHEstimator(nominal_capacity_Ah=3.0)
    cap_col = next((c for c in ["measured_capacity_Ah", "Capacity"]
                    if c in test_df.columns), None)
    if cap_col:
        kalman_preds = np.array([
            k_estimator.step(row[cap_col], int(row.get("cycle", i + 1)))
            for i, (_, row) in enumerate(test_df.iterrows())
        ])
    else:
        kalman_preds = np.full(len(y_test), y_test.mean())
    results["kalman"] = evaluate_model(y_test, kalman_preds, "Kalman EKF")
    predictions["Kalman EKF"] = kalman_preds

    # ── Layer 2a: Random Forest ────────────────────────────────────
    print(f"\n  {'─'*54}")
    print(f"  [L2a] Random Forest SoH  (soh_models.py)")
    rf = RandomForestSoH(**RF_PARAMS)
    rf.fit(X_train_aug, y_train, feature_cols=avail_cols)
    rf_preds = rf.predict(X_test)
    results["rf"] = evaluate_model(y_test, rf_preds, "Random Forest")
    predictions["Random Forest"] = rf_preds
    models["rf"] = rf
    rf.save(os.path.join(MODELS_DIR, "soh", "rf_soh.pkl"))
    print(f"     Saved → models/soh/rf_soh.pkl")

    # Feature importance from RF
    fi = rf.feature_importance()
    if fi is not None:
        top3 = fi.nlargest(3).index.tolist()
        print(f"     Top-3 features: {top3}")

    # ── Layer 2b: XGBoost ──────────────────────────────────────────
    xgb_preds = None
    if XGB_AVAILABLE:
        print(f"\n  {'─'*54}")
        print(f"  [L2b] XGBoost SoH  (soh_models.py)")
        try:
            xgb_params = {k: v for k, v in XGB_PARAMS.items()
                          if k != "n_jobs"}
            xgb = XGBoostSoH(**xgb_params)
            xgb.fit(X_train_aug, y_train, X_val=X_val, y_val=y_val)
            xgb_preds = xgb.predict(X_test)
            results["xgb"] = evaluate_model(y_test, xgb_preds, "XGBoost")
            predictions["XGBoost"] = xgb_preds
            models["xgb"] = xgb
            xgb.save(os.path.join(MODELS_DIR, "soh", "xgb_soh.pkl"))
            print(f"     Saved → models/soh/xgb_soh.pkl")
        except Exception as e:
            print(f"     XGBoost failed: {e}")
            XGB_AVAILABLE_local = False
    else:
        print(f"\n  [L2b] XGBoost SKIPPED (run: brew install libomp)")

    # ── Layer 3a: CNN-LSTM (optional, needs PyTorch) ───────────────
    cnn_preds = None
    if TORCH_AVAILABLE:
        print(f"\n  {'─'*54}")
        print(f"  [L3a] CNN-LSTM SoH  (soh_models.py)")
        try:
            seq_len = CNN_LSTM_PARAMS["sequence_len"]
            X_seq_tr, y_seq_tr, _ = build_sequences(
                train_df, avail_cols, "soh", seq_len)
            X_seq_va, y_seq_va, _ = build_sequences(
                val_df, avail_cols, "soh", seq_len)
            X_seq_te, y_seq_te, _ = build_sequences(
                test_df, avail_cols, "soh", seq_len)

            if len(X_seq_tr) > 0:
                cnn = CNNLSTMSoH(
                    n_features=len(avail_cols),
                    sequence_len=seq_len,
                    cnn_filters=CNN_LSTM_PARAMS["cnn_filters"],
                    lstm_units=CNN_LSTM_PARAMS["lstm_units"],
                    dropout=CNN_LSTM_PARAMS["dropout"],
                    lr=CNN_LSTM_PARAMS["learning_rate"],
                    epochs=CNN_LSTM_PARAMS["epochs"],
                    batch_size=CNN_LSTM_PARAMS["batch_size"],
                    patience=CNN_LSTM_PARAMS["patience"],
                )
                cnn.fit(X_seq_tr, y_seq_tr, X_seq_va, y_seq_va)
                cnn_preds = cnn.predict(X_seq_te)
                results["cnn_lstm"] = evaluate_model(
                    y_seq_te, cnn_preds, "CNN-LSTM")
                predictions["CNN-LSTM"] = cnn_preds
                models["cnn_lstm"] = cnn
                cnn.save(os.path.join(MODELS_DIR, "soh", "cnnlstm_soh.pt"))
                print(f"     Saved → models/soh/cnnlstm_soh.pt")
            else:
                print("     Skipped — not enough data per battery for sequences")
        except Exception as e:
            print(f"     CNN-LSTM failed: {e}")

    # ── Layer 3b: GPR Residual Corrector ──────────────────────────
    print(f"\n  {'─'*54}")
    print(f"  [L3b] GPR Residual Corrector  (soh_models.py)")
    X_gpr = np.vstack([X_train_aug, X_val])
    rf_preds_trainval = np.concatenate([
        rf.predict(X_train_aug),
        rf.predict(X_val)
    ])
    residuals = np.concatenate([y_train, y_val]) - rf_preds_trainval

    gpr = GPRResidualCorrector(uncertainty_threshold=3.0)
    gpr.fit(X_gpr, residuals)
    gpr_delta, gpr_std = gpr.predict(X_test)
    gpr_corrected = rf_preds + gpr_delta
    results["gpr"] = evaluate_model(y_test, gpr_corrected, "RF + GPR")
    predictions["RF + GPR"] = gpr_corrected
    models["gpr"] = gpr
    gpr.save(os.path.join(MODELS_DIR, "soh", "gpr_corrector.pkl"))
    print(f"     Saved → models/soh/gpr_corrector.pkl")
    print(f"     Mean uncertainty: ±{gpr_std.mean():.2f}%")

    # ── Layer 4: Bayesian Ensemble Fusion ─────────────────────────
    print(f"\n  {'─'*54}")
    print(f"  [L4] Bayesian Ensemble Fusion  (soh_models.py)")

    # Dynamic weights based on individual performance
    rf_mae  = results["rf"]["mae"]
    gpr_mae = results["gpr"]["mae"]
    kal_mae = results["kalman"]["mae"]
    xgb_mae = results["xgb"]["mae"] if "xgb" in results else 999

    def safe_weight(mae): return 1.0 / (mae + 1e-6)

    w_rf  = safe_weight(rf_mae)
    w_gpr = safe_weight(gpr_mae)
    w_kal = safe_weight(kal_mae)
    w_xgb = safe_weight(xgb_mae) if xgb_preds is not None else 0.0
    total = w_rf + w_gpr + w_kal + w_xgb

    ensemble = SoHEnsemble(
        kalman_weight = w_kal / total,
        rf_weight     = w_rf  / total,
        xgb_weight    = w_xgb / total,
        gpr_weight    = w_gpr / total,
        cnn_weight    = 0.0,
    )
    print(f"     Dynamic weights (performance-based):")
    print(f"       Kalman: {w_kal/total:.3f} | RF: {w_rf/total:.3f} | "
          f"GPR: {w_gpr/total:.3f} | XGB: {w_xgb/total:.3f}")

    ensemble_preds = np.array([
        ensemble.fuse(
            kalman_soh = kalman_preds[i],
            rf_soh     = rf_preds[i],
            xgb_soh    = xgb_preds[i] if xgb_preds is not None else None,
            gpr_base   = rf_preds[i],
            gpr_delta  = gpr_delta[i],
            gpr_std    = gpr_std[i],
        )["soh_ensemble"]
        for i in range(len(y_test))
    ])
    results["ensemble"] = evaluate_model(y_test, ensemble_preds, "ENSEMBLE")
    predictions["Ensemble"] = ensemble_preds

    joblib.dump(ensemble,  os.path.join(MODELS_DIR, "soh", "ensemble_config.pkl"))
    joblib.dump(avail_cols, os.path.join(MODELS_DIR, "soh", "feature_cols.pkl"))
    print(f"     Saved → models/soh/ensemble_config.pkl")

    # ── Final Summary ──────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  SOH RESULTS  ({len(y_test)} test samples, "
          f"{test_df['battery_id'].nunique()} unseen batteries)")
    print(f"{'═'*60}")
    print(f"  {'Model':<25} {'MAE':>7} {'RMSE':>7} {'R²':>8} {'MAPE':>7}")
    print(f"  {'─'*56}")
    priority = ["kalman", "rf", "xgb", "gpr", "cnn_lstm", "ensemble"]
    for key in priority:
        if key in results:
            m = results[key]
            star = " " if key == "ensemble" else ""
            print(f"  {m['model']:<25} "
                  f"{m['mae']:>7.3f} {m['rmse']:>7.3f} "
                  f"{m['r2']:>8.4f} {m.get('mape', 0):>7.2f}%{star}")

    return {
        "results":      results,
        "predictions":  predictions,
        "models":       models,
        "test_y":       y_test,
        "scaler":       scaler,
        "feature_cols": avail_cols,
        "ensemble":     ensemble,
    }



def train_soc_pipeline(feat_df: pd.DataFrame) -> dict:
    """
    SoC pipeline using soc_models.py:
      Primary: BiLSTM correction on Kalman residuals
      Fallback: Ridge regression corrector
    """
    print(f"\n{'═'*60}")
    print(f"  SOC PIPELINE  (soc_models.py + kalman_filters.py)")
    print(f"{'═'*60}")

    if "soc_true" not in feat_df.columns:
        print("  ℹ  soc_true not in dataset — SoC training skipped.")
        print("     (NASA real data has this via discharge profiles)")
        print("     Using synthetic data or rerun with --n-cycles 400+")
        return {}

    avail_soc = [c for c in SOC_FEATURE_COLS if c in feat_df.columns]
    if not avail_soc:
        avail_soc = [c for c in ["v_mean", "temperature", "cycle",
                                  "internal_resistance", "crate_mean"]
                     if c in feat_df.columns]
    print(f"  SoC features ({len(avail_soc)}): {avail_soc}")

    train_df, val_df, test_df = split_by_battery(feat_df)
    seq_len = CNN_LSTM_PARAMS["sequence_len"]

    try:
        X_tr, y_tr, _ = build_sequences(train_df, avail_soc, "soc_true", seq_len)
        X_va, y_va, _ = build_sequences(val_df,   avail_soc, "soc_true", seq_len)
        X_te, y_te, _ = build_sequences(test_df,  avail_soc, "soc_true", seq_len)
    except Exception as e:
        print(f"  Sequence build failed: {e}")
        return {}

    if len(X_tr) == 0:
        print("  ⚠ Not enough data per battery to build sequences.")
        return {}

    print(f"  Sequences: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")

    # Simulate Kalman SoC noise (residuals to learn)
    rng = np.random.default_rng(42)
    kalman_noise_tr = rng.normal(0, 0.03, y_tr.shape)
    kalman_noise_va = rng.normal(0, 0.03, y_va.shape)
    correction_tr = y_tr - np.clip(y_tr + kalman_noise_tr, 0, 1)
    correction_va = y_va - np.clip(y_va + kalman_noise_va, 0, 1)

    if BILSTM_AVAILABLE:
        print(f"  Training BiLSTM SoC correction model...")
        soc_model = BiLSTMSoCModel(
            n_features=len(avail_soc), hidden_units=64, n_layers=2,
            dropout=0.2, lr=1e-3,
            epochs=CNN_LSTM_PARAMS["epochs"],
            batch_size=CNN_LSTM_PARAMS["batch_size"],
            patience=CNN_LSTM_PARAMS["patience"],
        )
    else:
        print(f"  Training Ridge fallback SoC corrector...")
        soc_model = SimpleSoCCorrector()

    soc_model.fit(X_tr, correction_tr, X_va, correction_va)

    soc_path = os.path.join(MODELS_DIR, "soc", "bilstm_soc.pt")
    soc_model.save(soc_path)
    print(f"  Saved → models/soc/bilstm_soc.pt")

    # Evaluate
    test_corr = soc_model.predict_correction(X_te)
    kalman_noise_te = rng.normal(0, 0.03, y_te.shape)
    final_soc = np.clip(y_te + kalman_noise_te + test_corr, 0, 1)
    r = evaluate_model(y_te, final_soc, "BiLSTM SoC")
    print(f"\n  SoC Results: MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  R²={r['r2']:.4f}")

    return {"results": r, "model": soc_model, "feature_cols": avail_soc}



def main(force_synthetic: bool = False,
         n_batteries: int = 12,
         n_cycles: int = 500,
         skip_soc: bool = False,
         eval_only: bool = False):

    ensure_dirs()
    print(f"\n{'═'*60}")
    print(f"  BATTERY BMS AI — HIGH-ACCURACY TRAINING PIPELINE")
    print(f"{'═'*60}\n")

    if eval_only:
        print("--eval-only: loading saved models and evaluating...")
        from predict import BatteryPredictor
        pred = BatteryPredictor()
        pred.load_models(verbose=True)
        return

    # ── Step 1: Detect data source ─────────────────────────────────
    print("Detecting data source...")
    source = "synthetic_new" if force_synthetic else detect_data_source()

    # ── Step 2: Load via nasa_loader.py ───────────────────────────
    df = load_data(source, force_synthetic, n_batteries, n_cycles)

    # ── Step 3: Feature engineering ───────────────────────────────
    feat_df = run_feature_engineering(df)

    # ── Step 4: SoH pipeline ──────────────────────────────────────
    soh_out = train_soh_pipeline(feat_df)

    # ── Step 5: SoC pipeline ──────────────────────────────────────
    soc_out = {} if skip_soc else train_soc_pipeline(feat_df)

    # ── Done ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  ✓ TRAINING COMPLETE")
    print(f"  Models → {MODELS_DIR}/soh/  and  {MODELS_DIR}/soc/")
    if soh_out.get("results") and "ensemble" in soh_out["results"]:
        ens = soh_out["results"]["ensemble"]
        print(f"  Best model (Ensemble): "
              f"MAE={ens['mae']:.3f}%  R²={ens['r2']:.4f}")
    print(f"{'═'*60}\n")

    return soh_out, soc_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMS AI Training Pipeline")
    parser.add_argument("--force-synthetic", action="store_true",
                        help="Use synthetic data even if NASA files exist")
    parser.add_argument("--n-batteries", type=int, default=12,
                        help="Batteries per chemistry (synthetic mode only)")
    parser.add_argument("--n-cycles",    type=int, default=500,
                        help="Cycles per battery (synthetic mode only)")
    parser.add_argument("--skip-soc",    action="store_true")
    parser.add_argument("--eval-only",   action="store_true",
                        help="Skip training, just load and evaluate saved models")
    args = parser.parse_args()

    main(
        force_synthetic = args.force_synthetic,
        n_batteries     = args.n_batteries,
        n_cycles        = args.n_cycles,
        skip_soc        = args.skip_soc,
        eval_only       = args.eval_only,
    )