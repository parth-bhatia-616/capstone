
import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from config import MODELS_DIR
from preprocessing import SOH_FEATURE_COLS, BatteryScaler
from soh_models import RandomForestSoH, GPRResidualCorrector, SoHEnsemble
from kalman_filters import ECMParams, AdaptiveEKF, UnscentedKalmanFilter, KalmanSoHEstimator
from soc_models import SoCFusion, OnlineSoCFeatureBuilder
from grading import grade_battery, get_recommendation
from decision_engine import full_decision

try:
    from soh_models import XGBoostSoH
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from soc_models import BiLSTMSoCModel
    TORCH_AVAILABLE = True
except Exception:
    from soc_models import SimpleSoCCorrector
    TORCH_AVAILABLE = False


class BatteryPredictor:
    """
    Complete battery state estimation engine.

    Quick start:
        pred = BatteryPredictor()
        pred.load_models()
        result = pred.predict(features, cell_voltages)
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.soh_dir = os.path.join(models_dir, "soh")
        self.soc_dir = os.path.join(models_dir, "soc")

        self.rf = self.xgb = self.gpr = self.ensemble = None
        self.scaler: Optional[BatteryScaler] = None
        self.soc_model = None
        self.feature_cols = None

        # Online estimators (stateful)
        self._kalman_soh = KalmanSoHEstimator(nominal_capacity_Ah=3.0)
        self._ecm = ECMParams()
        self._aekf: Optional[AdaptiveEKF] = None
        self._ukf: Optional[UnscentedKalmanFilter] = None
        self._soc_fusion = SoCFusion(kalman_trust=0.6, ml_trust=0.4)
        self._feat_builder = OnlineSoCFeatureBuilder(window_size=50)
        self._loaded = False

    # ─── Model Loading ────────────────────────────────────────────

    def load_models(self, verbose: bool = True):
        """Load all available trained models from disk."""
        soh_d = self.soh_dir

        def _try_load(path, loader, label):
            if os.path.exists(path):
                obj = loader(path)
                if verbose:
                    print(f"  ✓ {label}")
                return obj
            if verbose:
                print(f"  – {label} not found (train first)")
            return None

        if verbose:
            print("Loading BMS AI models...")

        self.scaler   = _try_load(os.path.join(soh_d, "scaler.pkl"),
                                   BatteryScaler.load, "Scaler")
        self.rf       = _try_load(os.path.join(soh_d, "rf_soh.pkl"),
                                   RandomForestSoH.load, "Random Forest SoH")
        if XGB_AVAILABLE:
            self.xgb  = _try_load(os.path.join(soh_d, "xgb_soh.pkl"),
                                   XGBoostSoH.load, "XGBoost SoH")
        self.gpr      = _try_load(os.path.join(soh_d, "gpr_corrector.pkl"),
                                   GPRResidualCorrector.load, "GPR Corrector")

        ens_path = os.path.join(soh_d, "ensemble_config.pkl")
        self.ensemble = joblib.load(ens_path) if os.path.exists(ens_path) else SoHEnsemble()

        soc_path = os.path.join(self.soc_dir, "bilstm_soc.pt")
        if os.path.exists(soc_path):
            try:
                self.soc_model = (BiLSTMSoCModel if TORCH_AVAILABLE else
                                  SimpleSoCCorrector).load(soc_path)
                if verbose:
                    print("  ✓ SoC BiLSTM model")
            except Exception as e:
                if verbose:
                    print(f"  – SoC model failed: {e}")

        if self.scaler:
            self.feature_cols = self.scaler.feature_cols
        self._loaded = True

    # ─── Feature Preparation ──────────────────────────────────────

    def _to_array(self, features: Dict) -> np.ndarray:
        cols = self.feature_cols or SOH_FEATURE_COLS
        x = np.array([features.get(c, 0.0) for c in cols],
                     dtype=np.float32).reshape(1, -1)
        if self.scaler:
            x = self.scaler.scaler.transform(x)
        return x

    # ─── SoH Prediction ───────────────────────────────────────────

    def predict_soh(self, features: Dict,
                    measured_capacity_Ah: float = None,
                    cycle: int = None) -> Dict:
        """
        Ensemble SoH prediction.

        Returns dict with: soh, grade, uncertainty, recommendation,
                           soh_kalman, soh_rf, soh_xgb, gpr_correction, alert
        """
        if not self._loaded:
            self.load_models(verbose=False)

        X = self._to_array(features)

        kalman_val = None
        if measured_capacity_Ah is not None and cycle is not None:
            kalman_val = self._kalman_soh.step(measured_capacity_Ah, cycle)

        rf_soh  = float(self.rf.predict(X)[0])   if self.rf  else None
        xgb_soh = float(self.xgb.predict(X)[0])  if self.xgb else None

        gpr_delta = gpr_std = None
        if self.gpr and rf_soh is not None:
            d, s = self.gpr.predict(X)
            gpr_delta, gpr_std = float(d[0]), float(s[0])

        fuse_result = self.ensemble.fuse(
            kalman_soh=kalman_val,
            rf_soh=rf_soh,
            xgb_soh=xgb_soh,
            gpr_base=rf_soh,
            gpr_delta=gpr_delta,
            gpr_std=gpr_std,
        ) if self.ensemble else {
            "soh_ensemble": np.mean([v for v in [kalman_val, rf_soh] if v]),
            "soh_uncertainty": 2.0,
        }

        soh  = fuse_result["soh_ensemble"]
        unc  = fuse_result.get("soh_uncertainty")
        grade = grade_battery(soh, unc)

        return {
            "soh":             soh,
            "grade":           grade,
            "uncertainty":     unc,
            "recommendation":  get_recommendation(soh, grade, unc),
            "soh_kalman":      kalman_val,
            "soh_rf":          rf_soh,
            "soh_xgb":         xgb_soh,
            "gpr_correction":  gpr_delta,
            "gpr_std":         gpr_std,
            "alert":           fuse_result.get("alert", False),
        }

    # ─── SoC Online Step ──────────────────────────────────────────

    def step_soc_online(self, voltage: float, current: float,
                        temperature: float, dt_s: float = 1.0) -> Dict:
        """
        Real-time SoC estimation (call every sensor timestep).
        Returns fused SoC dict: soc, soc_percent, soc_ekf, soc_ukf, ml_correction
        """
        if self._aekf is None:
            self._aekf = AdaptiveEKF(self._ecm, initial_soc=0.9)
            self._ukf  = UnscentedKalmanFilter(self._ecm, initial_soc=0.9)

        ekf_out = self._aekf.step(current, voltage, dt_s)
        ukf_out = self._ukf.step(current, voltage, dt_s)

        ml_corr = 0.0
        if self.soc_model:
            seq = self._feat_builder.update(voltage, current, temperature)
            if seq is not None:
                try:
                    ml_corr = float(self.soc_model.predict_correction(seq)[0])
                except Exception:
                    pass

        return self._soc_fusion.fuse(
            soc_ekf=ekf_out["soc_ekf"],
            soc_ukf=ukf_out["soc_ukf"],
            ml_correction=ml_corr,
            ekf_uncertainty=ekf_out["p_soc"],
        )

    # ─── Complete Prediction ──────────────────────────────────────

    def predict(self, features: Dict, cell_voltages: List[float],
                measured_capacity_Ah: float = None, cycle: int = None,
                temperature: float = 25.0) -> Dict:
        """Full battery state: SoH + grade + balancing decision."""
        soh_result  = self.predict_soh(features, measured_capacity_Ah, cycle)
        bal_decision = full_decision(cell_voltages, soh_result["soh"], temperature)
        return {**soh_result, **bal_decision, "cell_voltages": cell_voltages}

    # ─── Pretty Print ──────────────────────────────────────────────

    def summary(self, result: Dict) -> str:
        lines = [
            "═" * 50,
            "  BATTERY BMS AI — PREDICTION RESULT",
            "═" * 50,
            f"  SoH:           {result['soh']:.1f}%"
              + (f" ± {result['uncertainty']:.1f}%" if result.get("uncertainty") else ""),
            f"  Grade:         {result['grade']}",
        ]
        for key, label in [("soh_kalman", "Kalman"), ("soh_rf", "RF"),
                            ("soh_xgb", "XGBoost")]:
            if result.get(key) is not None:
                lines.append(f"  {label+':':<14} {result[key]:.1f}%")
        if result.get("gpr_std") is not None:
            lines.append(f"  GPR std:       {result['gpr_std']:.2f}%")
        if result.get("mode"):
            lines += [
                f"  Balancing:     {result['mode']}",
                f"  ΔV:            {result.get('delta_v', 0)*1000:.1f} mV",
                f"  Weakest Cell:  Cell {result.get('weakest_cell', '?')}",
            ]
        if result.get("alert"):
            lines.append("  ⚠️  HIGH UNCERTAINTY — check calibration")
        lines.append(f"\n  {result.get('recommendation', '')}")
        lines.append("═" * 50)
        return "\n".join(lines)


# ─── Backward-Compatible Convenience Function ─────────────────────────────────

_global_predictor: Optional[BatteryPredictor] = None

def predict_soh(features: Dict) -> float:
    """Drop-in replacement for old predict_soh(). Returns SoH %."""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = BatteryPredictor()
        _global_predictor.load_models(verbose=False)
    return _global_predictor.predict_soh(features)["soh"]


if __name__ == "__main__":
    pred = BatteryPredictor()
    pred.load_models()

    features = {
        "cycle": 150, "internal_resistance": 0.032,
        "temperature": 28, "ambient_temperature": 25,
        "ic_peak_height": 4.2, "ic_peak_voltage": 3.6,
        "v_mean": 3.65, "v_std": 0.12,
        "energy_Wh": 9.8, "dod_mean": 0.8, "crate_mean": 1.2,
    }
    result = pred.predict(
        features,
        cell_voltages=[3.65, 3.72, 3.68, 3.70],
        measured_capacity_Ah=2.45, cycle=150
    )
    print(pred.summary(result))