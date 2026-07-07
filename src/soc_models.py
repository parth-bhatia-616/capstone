
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed — BiLSTM SoC model unavailable.")




if TORCH_AVAILABLE:

    class TemporalAttention(nn.Module):
       

        def __init__(self, hidden_dim: int):
            super().__init__()
            self.attention_weights = nn.Linear(hidden_dim, 1)

        def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            lstm_out: (batch, seq_len, hidden_dim)
            Returns:  (context_vector, attention_weights)
            """
            scores = self.attention_weights(lstm_out)          # (batch, seq_len, 1)
            attn_weights = torch.softmax(scores, dim=1)        # normalize over time
            context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim)
            return context, attn_weights.squeeze(-1)


    class BiLSTMSoCNetwork(nn.Module):
      

        def __init__(self, n_features: int, hidden_units: int = 128,
                     n_layers: int = 2, dropout: float = 0.3):
            super().__init__()

            self.norm = nn.LayerNorm(n_features)

            self.bilstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_units,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
                bidirectional=True,
            )

            # Attention over bidirectional output (hidden_units * 2)
            self.attention = TemporalAttention(hidden_units * 2)
            self.dropout = nn.Dropout(dropout)

            # Correction head
            self.fc1 = nn.Linear(hidden_units * 2, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor):
            x = self.norm(x)
            lstm_out, _ = self.bilstm(x)                    # (batch, seq_len, 2*hidden)
            context, attn_w = self.attention(lstm_out)      # (batch, 2*hidden)
            context = self.dropout(context)
            h = self.relu(self.fc1(context))
            h = self.dropout(h)
            h = self.relu(self.fc2(h))
            correction = self.fc3(h).squeeze(-1)            # (batch,)
            return correction, attn_w


    class BiLSTMSoCModel:
       

        def __init__(self, n_features: int,
                     hidden_units: int = 128, n_layers: int = 2,
                     dropout: float = 0.3, lr: float = 1e-3,
                     epochs: int = 100, batch_size: int = 128,
                     patience: int = 15, device: str = None):

            self.n_features = n_features
            self.epochs = epochs
            self.batch_size = batch_size
            self.patience = patience
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.fitted = False

            self.net = BiLSTMSoCNetwork(n_features, hidden_units, n_layers, dropout).to(self.device)
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
            self.criterion = nn.HuberLoss(delta=0.05)  # robust to outliers

            self.scaler = StandardScaler()
            self.history = {"train_loss": [], "val_loss": []}

        def _prepare(self, X: np.ndarray, fit_scaler: bool = False) -> torch.Tensor:
            """Normalize and convert to tensor. X: (N, seq_len, features)"""
            N, T, F = X.shape
            X_flat = X.reshape(-1, F)
            if fit_scaler:
                X_flat = self.scaler.fit_transform(X_flat)
            else:
                X_flat = self.scaler.transform(X_flat)
            return torch.FloatTensor(X_flat.reshape(N, T, F)).to(self.device)

        def fit(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray = None, y_val: np.ndarray = None):
          
            X_t = self._prepare(X_train, fit_scaler=True)
            y_t = torch.FloatTensor(y_train).to(self.device)
            dataset = TensorDataset(X_t, y_t)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

            has_val = X_val is not None and y_val is not None
            if has_val:
                X_v = self._prepare(X_val)
                y_v = torch.FloatTensor(y_val).to(self.device)

            best_loss = float("inf")
            patience_count = 0
            best_weights = None

            print(f"Training BiLSTM SoC on {self.device} | {len(X_train)} sequences")
            for epoch in range(self.epochs):
                self.net.train()
                epoch_loss = 0.0
                for Xb, yb in loader:
                    self.optimizer.zero_grad()
                    corr, _ = self.net(Xb)
                    loss = self.criterion(corr, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    self.optimizer.step()
                    epoch_loss += loss.item() * len(Xb)

                epoch_loss /= len(X_train)
                self.history["train_loss"].append(epoch_loss)
                self.scheduler.step()

                if has_val:
                    self.net.eval()
                    with torch.no_grad():
                        val_corr, _ = self.net(X_v)
                        val_loss = self.criterion(val_corr, y_v).item()
                    self.history["val_loss"].append(val_loss)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_weights = {k: v.clone() for k, v in self.net.state_dict().items()}
                        patience_count = 0
                    else:
                        patience_count += 1
                        if patience_count >= self.patience:
                            print(f"  Early stopping at epoch {epoch+1}")
                            break

                if (epoch + 1) % 20 == 0:
                    vs = f"  val={val_loss:.5f}" if has_val else ""
                    print(f"  Epoch {epoch+1}/{self.epochs}  train={epoch_loss:.5f}{vs}")

            if best_weights:
                self.net.load_state_dict(best_weights)

            self.fitted = True
            return self

        def predict_correction(self, X: np.ndarray) -> np.ndarray:
            """Returns SoC correction array (add to Kalman estimate)."""
            self.net.eval()
            X_t = self._prepare(X)
            with torch.no_grad():
                corr, _ = self.net(X_t)
            return corr.cpu().numpy()

        def predict_with_attention(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Returns (corrections, attention_weights) for visualization."""
            self.net.eval()
            X_t = self._prepare(X)
            with torch.no_grad():
                corr, attn = self.net(X_t)
            return corr.cpu().numpy(), attn.cpu().numpy()

        def save(self, path: str):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                "model_state": self.net.state_dict(),
                "scaler": self.scaler,
                "history": self.history,
                "n_features": self.n_features,
            }, path)
            print(f"BiLSTMSoCModel saved → {path}")

        @classmethod
        def load(cls, path: str, device: str = None):
            ckpt = torch.load(path, map_location=device or "cpu")
            model = cls(n_features=ckpt["n_features"], device=device)
            model.net.load_state_dict(ckpt["model_state"])
            model.scaler = ckpt["scaler"]
            model.history = ckpt.get("history", {})
            model.fitted = True
            return model


class SimpleSoCCorrector:
 
    def __init__(self):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """X: (N, seq_len, features) — flattened."""
        N = X_train.shape[0]
        X_flat = X_train.reshape(N, -1)
        X_scaled = self.scaler.fit_transform(X_flat)
        self.model.fit(X_scaled, y_train)
        self.fitted = True
        train_pred = self.model.predict(X_scaled)
        print(f"Ridge SoC train MAE: {mean_absolute_error(y_train, train_pred):.4f}")
        return self

    def predict_correction(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        X_flat = self.scaler.transform(X.reshape(N, -1))
        return self.model.predict(X_flat)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)


class SoCFusion:
 

    def __init__(self, kalman_trust: float = 0.6, ml_trust: float = 0.4):
        self.kalman_trust = kalman_trust
        self.ml_trust = ml_trust
        assert abs(kalman_trust + ml_trust - 1.0) < 1e-6, "Weights must sum to 1"

    def fuse(self,
             soc_ekf: float,
             soc_ukf: float = None,
             ml_correction: float = 0.0,
             ekf_uncertainty: float = None,
             impedance_soc: float = None) -> Dict:
    
        # Fuse EKF and UKF
        if soc_ukf is not None:
            # Average of two Kalman variants (equal weight, or bias EKF)
            kalman_soc = 0.6 * soc_ekf + 0.4 * soc_ukf
        else:
            kalman_soc = soc_ekf

        # ML-corrected estimate
        ml_soc = kalman_soc + ml_correction

        # Adjust trust if uncertainty is high
        if ekf_uncertainty is not None and ekf_uncertainty > 0.01:
            # High uncertainty → trust ML more
            extra = min(0.2, ekf_uncertainty * 2)
            kalman_w = max(0.2, self.kalman_trust - extra)
            ml_w = 1.0 - kalman_w
        else:
            kalman_w, ml_w = self.kalman_trust, self.ml_trust

        fused = kalman_w * kalman_soc + ml_w * ml_soc
        fused = float(np.clip(fused, 0.0, 1.0))

        # Override with impedance measurement if available (most accurate at rest)
        if impedance_soc is not None:
            fused = 0.3 * fused + 0.7 * float(impedance_soc)

        return {
            "soc": fused,
            "soc_percent": fused * 100,
            "soc_ekf": float(soc_ekf),
            "soc_ukf": float(soc_ukf) if soc_ukf is not None else None,
            "ml_correction": float(ml_correction),
            "kalman_weight": kalman_w,
            "ml_weight": ml_w,
        }




class OnlineSoCFeatureBuilder:
   

    def __init__(self, window_size: int = 50, dt_s: float = 1.0):
        self.window_size = window_size
        self.dt_s = dt_s
        self._buffer: List[Dict] = []
        self._cumulative_charge = 0.0

    def update(self, voltage: float, current: float, temperature: float) -> Optional[np.ndarray]:
       

        v_prev = self._buffer[-1]["voltage"] if self._buffer else voltage
        i_prev = self._buffer[-1]["current"] if self._buffer else current

        dv_dt = (voltage - v_prev) / self.dt_s
        di_dt = (current - i_prev) / self.dt_s
        power = voltage * current
        self._cumulative_charge += current * self.dt_s / 3600.0  # Ah

        self._buffer.append({
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
            "dv_dt": dv_dt,
            "di_dt": di_dt,
            "power": power,
            "cumulative_charge_Ah": self._cumulative_charge,
        })


        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        if len(self._buffer) >= self.window_size:
            seq = np.array([[
                s["voltage"], s["current"], s["temperature"],
                s["dv_dt"], s["di_dt"], s["power"], s["cumulative_charge_Ah"]
            ] for s in self._buffer], dtype=np.float32)
            return seq[np.newaxis, :, :]  # (1, window, 7)
        return None

    def reset(self):
        self._buffer.clear()
        self._cumulative_charge = 0.0


if __name__ == "__main__":
    print("Testing SoC model components...")


    builder = OnlineSoCFeatureBuilder(window_size=10, dt_s=1.0)
    for i in range(12):
        feat = builder.update(3.7 - i * 0.01, 1.0, 25.0)
        if feat is not None:
            print(f"Step {i}: features shape {feat.shape}")

    # Test SoCFusion
    fusion = SoCFusion(kalman_trust=0.6, ml_trust=0.4)
    result = fusion.fuse(
        soc_ekf=0.75,
        soc_ukf=0.73,
        ml_correction=0.02,
        ekf_uncertainty=0.005,
    )
    print(f"\nFused SoC: {result['soc_percent']:.2f}%")
    print(f"EKF: {result['soc_ekf']*100:.2f}%  UKF: {result['soc_ukf']*100:.2f}%  Correction: {result['ml_correction']*100:.3f}%")

    if TORCH_AVAILABLE:
        print("\nPyTorch available — testing BiLSTM...")
        model = BiLSTMSoCModel(n_features=7, hidden_units=32, n_layers=1,
                               epochs=2, batch_size=16)
        X_dummy = np.random.randn(50, 10, 7).astype(np.float32)
        y_dummy = np.random.randn(50).astype(np.float32) * 0.05
        model.fit(X_dummy, y_dummy)
        corrections = model.predict_correction(X_dummy[:5])
        print(f"BiLSTM corrections: {corrections.round(4)}")
    else:
        print("\nPyTorch not available — testing SimpleSoCCorrector...")
        model = SimpleSoCCorrector()
        X_dummy = np.random.randn(50, 10, 7)
        y_dummy = np.random.randn(50) * 0.05
        model.fit(X_dummy, y_dummy)
        print("SimpleSoCCorrector OK")