

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler



try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except (ImportError, Exception):
    XGB_AVAILABLE = False
    XGBRegressor = None  

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. CNN-LSTM layer disabled.")




class RandomForestSoH:
 
    def __init__(self, **kwargs):
        params = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }
        params.update(kwargs)
        self.model = RandomForestRegressor(**params)
        self.feature_cols = None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, feature_cols=None):
        self.feature_cols = feature_cols
        self.model.fit(X, y)
        self.fitted = True
        train_pred = self.model.predict(X)
        print(f"RF train MAE: {mean_absolute_error(y, train_pred):.3f}%  "
              f"R²: {r2_score(y, train_pred):.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> Optional[pd.Series]:
        if self.feature_cols and self.fitted:
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_cols
            ).sort_values(ascending=False)
        return None

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"RandomForestSoH saved → {path}")

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)




class XGBoostSoH:
 

    def __init__(self, **kwargs):
        if not XGB_AVAILABLE:
            raise ImportError("xgboost required: pip install xgboost")
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        params.update(kwargs)
        self.model = XGBRegressor(**params)
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )
        self.fitted = True
        train_pred = self.model.predict(X)
        print(f"XGB train MAE: {mean_absolute_error(y, train_pred):.3f}%  "
              f"R²: {r2_score(y, train_pred):.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"XGBoostSoH saved → {path}")

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)




class GPRResidualCorrector:
   

    def __init__(self, uncertainty_threshold: float = 3.0):
        """
        uncertainty_threshold: Alert if GPR std > this value (% SoH).
        """
        self.uncertainty_threshold = uncertainty_threshold
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
            Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2)) +
            WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
        )
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-3,
            n_restarts_optimizer=5,
            normalize_y=True,
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: np.ndarray, residuals: np.ndarray):
        """
        Train GPR on (features, residuals).
        residuals = true_SoH - baseline_prediction
        """
        X_scaled = self.scaler.fit_transform(X)
        # Subsample for GPR tractability (GPR scales as O(n^3))
        max_samples = 1000
        if len(X_scaled) > max_samples:
            idx = np.random.default_rng(42).choice(len(X_scaled), max_samples, replace=False)
            X_fit = X_scaled[idx]
            y_fit = residuals[idx]
        else:
            X_fit, y_fit = X_scaled, residuals

        print(f"Fitting GPR on {len(X_fit)} samples...")
        self.gpr.fit(X_fit, y_fit)
        self.fitted = True
        pred_res, std = self.gpr.predict(X_fit, return_std=True)
        print(f"GPR residual MAE: {mean_absolute_error(y_fit, pred_res):.4f}%  "
              f"mean std: {std.mean():.4f}")
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (correction_delta, uncertainty_std).
        """
        X_scaled = self.scaler.transform(X)
        delta, std = self.gpr.predict(X_scaled, return_std=True)
        return delta, std

    def is_uncertain(self, std: float) -> bool:
        return std > self.uncertainty_threshold

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"GPRResidualCorrector saved → {path}")

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)




if TORCH_AVAILABLE:
    class CNNLSTMBlock(nn.Module):
        """
        CNN-LSTM for temporal SoH patterns.
        CNN extracts local features → LSTM captures long-range degradation trends.
        """

        def __init__(self, n_features: int, sequence_len: int,
                     cnn_filters: int = 64, cnn_kernel: int = 3,
                     lstm_units: int = 128, dropout: float = 0.3):
            super().__init__()

            self.conv1 = nn.Conv1d(n_features, cnn_filters, kernel_size=cnn_kernel, padding=cnn_kernel // 2)
            self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=cnn_kernel, padding=cnn_kernel // 2)
            self.bn1 = nn.BatchNorm1d(cnn_filters)
            self.bn2 = nn.BatchNorm1d(cnn_filters)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool1d(2)

            lstm_input_len = sequence_len // 2
            self.lstm = nn.LSTM(
                input_size=cnn_filters,
                hidden_size=lstm_units,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=False,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(lstm_units, 64)
            self.fc2 = nn.Linear(64, 1)

        def forward(self, x):
            # x: (batch, seq_len, features) → transpose for Conv1d
            x = x.permute(0, 2, 1)             # (batch, features, seq_len)
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)                    # (batch, filters, seq_len//2)
            x = x.permute(0, 2, 1)             # (batch, seq_len//2, filters)
            x, _ = self.lstm(x)
            x = self.dropout(x[:, -1, :])       # take last timestep
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x.squeeze(-1)


    class CNNLSTMSoH:
        """
        Wrapper for CNN-LSTM SoH model with training loop.
        """

        def __init__(self, n_features: int, sequence_len: int = 50,
                     cnn_filters: int = 64, lstm_units: int = 128,
                     dropout: float = 0.3, lr: float = 1e-3,
                     epochs: int = 100, batch_size: int = 64,
                     patience: int = 15, device: str = None):

            self.sequence_len = sequence_len
            self.epochs = epochs
            self.batch_size = batch_size
            self.patience = patience
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.fitted = False

            self.net = CNNLSTMBlock(
                n_features, sequence_len, cnn_filters, 3, lstm_units, dropout
            ).to(self.device)

            self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=5, factor=0.5, verbose=False
            )
            self.criterion = nn.MSELoss()
            self.history = {"train_loss": [], "val_loss": []}

        def fit(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray = None, y_val: np.ndarray = None):
            """
            X_train: (N, sequence_len, n_features)
            y_train: (N,) — SoH %
            """
            X_t = torch.FloatTensor(X_train).to(self.device)
            y_t = torch.FloatTensor(y_train).to(self.device)
            dataset = TensorDataset(X_t, y_t)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            has_val = X_val is not None and y_val is not None
            if has_val:
                X_v = torch.FloatTensor(X_val).to(self.device)
                y_v = torch.FloatTensor(y_val).to(self.device)

            best_val_loss = float("inf")
            patience_counter = 0
            best_weights = None

            print(f"Training CNN-LSTM on {self.device} | {len(X_train)} sequences")
            for epoch in range(self.epochs):
                self.net.train()
                epoch_loss = 0.0
                for Xb, yb in loader:
                    self.optimizer.zero_grad()
                    pred = self.net(Xb)
                    loss = self.criterion(pred, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    self.optimizer.step()
                    epoch_loss += loss.item() * len(Xb)
                epoch_loss /= len(X_train)
                self.history["train_loss"].append(epoch_loss)

                if has_val:
                    self.net.eval()
                    with torch.no_grad():
                        val_pred = self.net(X_v)
                        val_loss = self.criterion(val_pred, y_v).item()
                    self.history["val_loss"].append(val_loss)
                    self.scheduler.step(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = {k: v.clone() for k, v in self.net.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            print(f"  Early stopping at epoch {epoch+1}")
                            break

                if (epoch + 1) % 20 == 0:
                    val_str = f"  val_loss={val_loss:.4f}" if has_val else ""
                    print(f"  Epoch {epoch+1}/{self.epochs}  train_loss={epoch_loss:.4f}{val_str}")

            if best_weights:
                self.net.load_state_dict(best_weights)

            self.fitted = True
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            self.net.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(self.device)
                out = self.net(X_t).cpu().numpy()
            return out

        def save(self, path: str):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                "model_state": self.net.state_dict(),
                "history": self.history,
            }, path)
            print(f"CNNLSTMSoH saved → {path}")

        def load_weights(self, path: str):
            ckpt = torch.load(path, map_location=self.device)
            self.net.load_state_dict(ckpt["model_state"])
            self.history = ckpt.get("history", {})
            self.fitted = True


# ─── Bayesian Ensemble Fusion ─────────────────────────────────────────────────

class SoHEnsemble:
 

    def __init__(self,
                 kalman_weight: float = 0.20,
                 rf_weight: float = 0.25,
                 xgb_weight: float = 0.25,
                 gpr_weight: float = 0.30,
                 cnn_weight: float = 0.0,
                 uncertainty_threshold: float = 3.0):
        """
        Weights should sum to 1.0.
        cnn_weight is 0 unless CNN-LSTM is available.
        """
        self.base_weights = {
            "kalman": kalman_weight,
            "rf": rf_weight,
            "xgb": xgb_weight,
            "gpr": gpr_weight,
            "cnn": cnn_weight,
        }
        self.uncertainty_threshold = uncertainty_threshold

    def fuse(self,
             kalman_soh: float = None,
             rf_soh: float = None,
             xgb_soh: float = None,
             gpr_delta: float = None,
             gpr_std: float = None,
             gpr_base: float = None,
             cnn_soh: float = None) -> Dict:
    
        estimates = {}
        weights = {}

        if kalman_soh is not None:
            estimates["kalman"] = float(kalman_soh)
            weights["kalman"] = self.base_weights["kalman"]

        if rf_soh is not None:
            estimates["rf"] = float(rf_soh)
            weights["rf"] = self.base_weights["rf"]

        if xgb_soh is not None:
            estimates["xgb"] = float(xgb_soh)
            weights["xgb"] = self.base_weights["xgb"]

        if gpr_delta is not None and gpr_base is not None:
            gpr_corrected = gpr_base + gpr_delta
            estimates["gpr"] = float(gpr_corrected)
            # Reduce GPR weight if very uncertain
            if gpr_std is not None and gpr_std > self.uncertainty_threshold:
                gpr_w = self.base_weights["gpr"] * (self.uncertainty_threshold / gpr_std)
            else:
                gpr_w = self.base_weights["gpr"]
            weights["gpr"] = gpr_w

        if cnn_soh is not None:
            estimates["cnn"] = float(cnn_soh)
            weights["cnn"] = self.base_weights["cnn"]

        if not estimates:
            raise ValueError("No SoH estimates provided to ensemble.")

        # Normalize weights
        total_w = sum(weights.values())
        norm_weights = {k: v / total_w for k, v in weights.items()}

        # Weighted average
        fused_soh = sum(estimates[k] * norm_weights[k] for k in estimates)
        fused_soh = float(np.clip(fused_soh, 50.0, 100.0))

        # Weighted variance (epistemic uncertainty)
        variance = sum(
            norm_weights[k] * (estimates[k] - fused_soh) ** 2
            for k in estimates
        )
        uncertainty = float(np.sqrt(variance))

        return {
            "soh_ensemble": fused_soh,
            "soh_uncertainty": uncertainty,
            "estimates": estimates,
            "weights_used": norm_weights,
            "alert": gpr_std > self.uncertainty_threshold if gpr_std else False,
        }




def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   model_name: str = "Model") -> Dict:
    """Compute and print standard regression metrics."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)

    print(f"\n{model_name}:")
    print(f"  MAE : {mae:.3f}%")
    print(f"  RMSE: {rmse:.3f}%")
    print(f"  R²  : {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return {"model": model_name, "mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


if __name__ == "__main__":
    print("Testing SoH model components...")
    X_dummy = np.random.randn(200, 10)
    y_dummy = 80 + np.random.randn(200) * 10

    rf = RandomForestSoH(n_estimators=50)
    rf.fit(X_dummy, y_dummy)
    preds = rf.predict(X_dummy[:10])
    print(f"RF predictions (first 5): {preds[:5].round(2)}")

    # GPR test
    residuals = y_dummy - rf.predict(X_dummy)
    gpr = GPRResidualCorrector()
    gpr.fit(X_dummy[:200], residuals[:200])
    delta, std = gpr.predict(X_dummy[:5])
    print(f"GPR corrections: {delta.round(3)}, std: {std.round(3)}")

    # Ensemble test
    ensemble = SoHEnsemble()
    result = ensemble.fuse(kalman_soh=85.0, rf_soh=83.2, xgb_soh=84.1,
                           gpr_base=83.2, gpr_delta=0.5, gpr_std=1.2)
    print(f"Ensemble SoH: {result['soh_ensemble']:.2f}% ± {result['soh_uncertainty']:.2f}%")