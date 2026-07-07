

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ECMParams:
  
    R0: float = 0.025     # Series resistance (Ohm)
    R1: float = 0.015     # RC branch resistance (Ohm)
    C1: float = 3000.0    # RC branch capacitance (F)
    Q_nom: float = 3.0    # Nominal capacity (Ah)
    eta: float = 0.98     # Coulombic efficiency

    @property
    def tau(self):
        return self.R1 * self.C1




def get_ocv_function(chemistry: str = "NMC"):

    def ocv(soc: float) -> float:
        soc = float(np.clip(soc, 0.01, 0.99))
        if chemistry == "LFP":
            v_min, v_max = 2.5, 3.65
            return v_min + (v_max - v_min) * (
                0.1 * soc +
                0.85 / (1 + np.exp(-20 * (soc - 0.5))) +
                0.05 * soc ** 3
            )
        elif chemistry in ["NMC", "NCA"]:
            v_min, v_max = 2.8, 4.2
            return v_min + (v_max - v_min) * (-0.8 * soc**3 + 1.4 * soc**2 + 0.4 * soc)
        else:  # LCO
            v_min, v_max = 3.0, 4.2
            return v_min + (v_max - v_min) * (soc**0.5 * 0.6 + soc * 0.4)

    def docv(soc: float) -> float:
        """Numerical derivative dOCV/dSoC."""
        h = 1e-4
        return (ocv(soc + h) - ocv(soc - h)) / (2 * h)

    return ocv, docv




class ExtendedKalmanFilter:


    def __init__(self, params: ECMParams, chemistry: str = "NMC",
                 initial_soc: float = 1.0,
                 Q_noise: Optional[np.ndarray] = None,
                 R_noise: float = 1e-4):
        self.params = params
        self.ocv, self.docv = get_ocv_function(chemistry)

        # State: [SoC, V_RC]
        self.x = np.array([initial_soc, 0.0])

        # Error covariance
        self.P = np.diag([0.01, 1e-6])

        # Process noise covariance
        self.Q = Q_noise if Q_noise is not None else np.diag([1e-7, 1e-8])

        # Measurement noise covariance
        self.R = np.array([[R_noise]])

        self.history = {"soc": [], "v_rc": [], "v_terminal": [], "innovation": []}

    def predict(self, current_A: float, dt_s: float) -> np.ndarray:
        """EKF predict step (time update)."""
        p = self.params
        soc, v_rc = self.x

        alpha = np.exp(-dt_s / p.tau)

        # State transition
        soc_new = soc - (p.eta * current_A * dt_s) / (p.Q_nom * 3600)
        v_rc_new = alpha * v_rc + p.R1 * (1 - alpha) * current_A

        self.x = np.array([np.clip(soc_new, 0.0, 1.0), v_rc_new])

        # Jacobian F (state transition matrix)
        F = np.array([
            [1.0, 0.0],
            [0.0, alpha],
        ])

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

        return self.x.copy()

    def update(self, v_measured: float, current_A: float) -> Tuple[np.ndarray, float]:
        """EKF update step (measurement update)."""
        p = self.params
        soc, v_rc = self.x

        # Predicted terminal voltage
        v_pred = self.ocv(soc) - v_rc - p.R0 * current_A

        # Jacobian H (observation matrix)
        H = np.array([[self.docv(soc), -1.0]])

        # Innovation
        innovation = v_measured - v_pred

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K.flatten() * innovation
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Log
        self.history["soc"].append(float(self.x[0]))
        self.history["v_rc"].append(float(self.x[1]))
        self.history["v_terminal"].append(float(v_pred))
        self.history["innovation"].append(float(innovation))

        return self.x.copy(), float(innovation)

    def step(self, current_A: float, v_measured: float, dt_s: float) -> Dict:
        """One complete EKF cycle: predict + update."""
        self.predict(current_A, dt_s)
        x, innov = self.update(v_measured, current_A)
        return {
            "soc_ekf": float(x[0]),
            "v_rc_ekf": float(x[1]),
            "innovation": float(innov),
            "p_soc": float(self.P[0, 0]),  # SoC variance
        }




class AdaptiveEKF(ExtendedKalmanFilter):


    def __init__(self, params: ECMParams, chemistry: str = "NMC",
                 initial_soc: float = 1.0, window_size: int = 30):
        super().__init__(params, chemistry, initial_soc)
        self.window_size = window_size
        self.innovation_buffer = []
        self.R_adapt = self.R.copy()
        self.forgetting_factor = 0.98  # exponential forgetting

    def update(self, v_measured: float, current_A: float):
        # Run standard update
        x, innov = super().update(v_measured, current_A)

        # Adaptive R update
        self.innovation_buffer.append(innov)
        if len(self.innovation_buffer) > self.window_size:
            self.innovation_buffer.pop(0)

        if len(self.innovation_buffer) >= 5:
            innov_arr = np.array(self.innovation_buffer)

            # Observation Jacobian
            soc = self.x[0]
            H = np.array([[self.docv(soc), -1.0]])

            # Estimated measurement noise
            C_innov = np.var(innov_arr)
            R_est = max(C_innov - (H @ self.P @ H.T)[0, 0], 1e-8)

            # Smooth update with forgetting factor
            alpha = self.forgetting_factor
            self.R[0, 0] = alpha * self.R[0, 0] + (1 - alpha) * R_est

        return x, innov




class UnscentedKalmanFilter:


    def __init__(self, params: ECMParams, chemistry: str = "NMC",
                 initial_soc: float = 1.0,
                 alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.params = params
        self.ocv, self.docv = get_ocv_function(chemistry)
        self.n = 2  # State dimension

        # UKF tuning
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = alpha**2 * (self.n + kappa) - self.n

        # Weights
        self.Wm, self.Wc = self._compute_weights()

        # State
        self.x = np.array([initial_soc, 0.0])
        self.P = np.diag([0.01, 1e-6])
        self.Q = np.diag([1e-7, 1e-8])
        self.R_meas = 1e-4

        self.history = {"soc": [], "v_terminal": []}

    def _compute_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        n, lam = self.n, self.lam
        Wm = np.full(2 * n + 1, 1 / (2 * (n + lam)))
        Wm[0] = lam / (n + lam)
        Wc = Wm.copy()
        Wc[0] += (1 - self.alpha**2 + self.beta)
        return Wm, Wc

    def _sigma_points(self) -> np.ndarray:
        n, lam = self.n, self.lam
        L = np.linalg.cholesky((n + lam) * self.P)
        sigma = np.zeros((2 * n + 1, n))
        sigma[0] = self.x
        for i in range(n):
            sigma[i + 1] = self.x + L[:, i]
            sigma[n + i + 1] = self.x - L[:, i]
        return sigma

    def _state_transition(self, sigma: np.ndarray, current_A: float, dt_s: float) -> np.ndarray:
        p = self.params
        alpha_rc = np.exp(-dt_s / p.tau)
        propagated = np.zeros_like(sigma)
        for i, s in enumerate(sigma):
            soc = np.clip(s[0] - (p.eta * current_A * dt_s) / (p.Q_nom * 3600), 0.0, 1.0)
            v_rc = alpha_rc * s[1] + p.R1 * (1 - alpha_rc) * current_A
            propagated[i] = [soc, v_rc]
        return propagated

    def _measurement_model(self, sigma: np.ndarray, current_A: float) -> np.ndarray:
        p = self.params
        return np.array([self.ocv(s[0]) - s[1] - p.R0 * current_A for s in sigma])

    def step(self, current_A: float, v_measured: float, dt_s: float) -> Dict:
        """One UKF step."""
        # Generate sigma points
        sigma = self._sigma_points()

        # Propagate through state model
        sigma_pred = self._state_transition(sigma, current_A, dt_s)

        # Predicted mean and covariance
        x_pred = np.sum(self.Wm[:, None] * sigma_pred, axis=0)
        P_pred = self.Q.copy()
        for i, s in enumerate(sigma_pred):
            d = (s - x_pred)[:, None]
            P_pred += self.Wc[i] * (d @ d.T)

        # Measurement sigma points
        z_sigma = self._measurement_model(sigma_pred, current_A)
        z_pred = np.sum(self.Wm * z_sigma)

        # Innovation covariance
        S = self.R_meas
        for i, z in enumerate(z_sigma):
            S += self.Wc[i] * (z - z_pred) ** 2

        # Cross covariance
        Pxz = np.zeros(self.n)
        for i, (s, z) in enumerate(zip(sigma_pred, z_sigma)):
            Pxz += self.Wc[i] * (s - x_pred) * (z - z_pred)

        # Kalman gain
        K = Pxz / S

        # Update
        innovation = v_measured - z_pred
        self.x = x_pred + K * innovation
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        self.P = P_pred - np.outer(K, K) * S

        self.history["soc"].append(float(self.x[0]))
        self.history["v_terminal"].append(float(z_pred))

        return {
            "soc_ukf": float(self.x[0]),
            "v_rc_ukf": float(self.x[1]),
            "innovation_ukf": float(innovation),
            "p_soc_ukf": float(self.P[0, 0]),
        }




class ECMParameterIdentifier:
   

    def __init__(self, forgetting_factor: float = 0.98):
        self.lam = forgetting_factor
        self.R0_est = 0.025
        self.R1_est = 0.015
        self.C1_est = 3000.0
        self._prev_I = 0.0
        self._prev_V = None
        self._v_step_buffer = []  

    def update(self, voltage: float, current: float, dt_s: float) -> Dict:

        result = {
            "R0_est": self.R0_est,
            "R1_est": self.R1_est,
            "C1_est": self.C1_est,
        }

        dI = current - self._prev_I
        if self._prev_V is not None:
            dV = voltage - self._prev_V


            if abs(dI) > 0.1:
                r0_obs = abs(dV / dI)
                if 0.005 < r0_obs < 0.2:
                    alpha = 0.1
                    self.R0_est = (1 - alpha) * self.R0_est + alpha * r0_obs
                    result["R0_est"] = self.R0_est
                    self._v_step_buffer = [(voltage, current, 0.0)]
            elif self._v_step_buffer:

                _, _, t_last = self._v_step_buffer[-1]
                self._v_step_buffer.append((voltage, current, t_last + dt_s))


                if len(self._v_step_buffer) > 10:
                    self._fit_rc_params()
                    self.C1_est = max(100, self.R1_est * 3000 / max(self.R1_est, 0.001))
                    result["R1_est"] = self.R1_est
                    result["C1_est"] = self.C1_est

        self._prev_I = current
        self._prev_V = voltage
        result["tau_est"] = self.R1_est * self.C1_est
        return result

    def _fit_rc_params(self):

        try:
            vs = np.array([v for v, _, _ in self._v_step_buffer])
            ts = np.array([t for _, _, t in self._v_step_buffer])
            v0 = vs[0]
            v_inf = vs[-1]
            dv = vs - v_inf


            mask = dv > 0.001
            if mask.sum() > 5:
                log_dv = np.log(dv[mask])
                t_fit = ts[mask]
                slope, intercept = np.polyfit(t_fit, log_dv, 1)
                tau_fit = max(10, -1 / slope) if slope < 0 else 300
                A_fit = np.exp(intercept)
                r1_fit = A_fit
                if 0.001 < r1_fit < 0.1:
                    self.R1_est = 0.9 * self.R1_est + 0.1 * r1_fit
                    self.C1_est = tau_fit / max(self.R1_est, 1e-6)
        except Exception:
            pass




class KalmanSoHEstimator:
  

    def __init__(self, nominal_capacity_Ah: float = 3.0):
        self.Q_nom = nominal_capacity_Ah

        # State: [SoH (0-1), fade_rate]
        self.x = np.array([1.0, -0.0003])
        self.P = np.diag([0.001, 1e-8])
        self.Q_noise = np.diag([1e-6, 1e-10])
        self.R_obs = np.array([[1e-4]])

        self.history = {"soh_ekf": [], "cycle": []}

    def step(self, measured_capacity_Ah: float, cycle: int) -> float:


        F = np.array([[1.0, 1.0], [0.0, 1.0]])  
        self.x = F @ self.x
        self.x[0] = np.clip(self.x[0], 0.5, 1.0)
        self.P = F @ self.P @ F.T + self.Q_noise

        # Observation
        H = np.array([[1.0, 0.0]])
        z = measured_capacity_Ah / self.Q_nom

        z_pred = self.x[0]
        S = H @ self.P @ H.T + self.R_obs
        K = self.P @ H.T @ np.linalg.inv(S)

        innovation = z - z_pred
        self.x = self.x + K.flatten() * innovation
        self.x[0] = np.clip(self.x[0], 0.4, 1.0)

        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ self.P

        soh_pct = float(self.x[0] * 100)
        self.history["soh_ekf"].append(soh_pct)
        self.history["cycle"].append(cycle)
        return soh_pct


if __name__ == "__main__":
    print("Testing EKF and UKF...")
    params = ECMParams(R0=0.025, R1=0.015, C1=3000, Q_nom=3.0)

    ekf = AdaptiveEKF(params, chemistry="NMC", initial_soc=0.9)
    ukf = UnscentedKalmanFilter(params, chemistry="NMC", initial_soc=0.9)

    np.random.seed(42)
    I = 1.0  #1A discharge
    dt = 10.0
    soc_true = 0.9

    errors_ekf, errors_ukf = [], []
    for k in range(100):
        soc_true -= I * dt / (3.0 * 3600)
        soc_true = max(soc_true, 0.01)

        v_true = 3.7 + 0.5 * soc_true - 0.025 * I + np.random.normal(0, 0.01)

        out_ekf = ekf.step(I, v_true, dt)
        out_ukf = ukf.step(I, v_true, dt)

        errors_ekf.append(abs(out_ekf["soc_ekf"] - soc_true))
        errors_ukf.append(abs(out_ukf["soc_ukf"] - soc_true))

    print(f"EKF mean SoC error: {np.mean(errors_ekf)*100:.3f}%")
    print(f"UKF mean SoC error: {np.mean(errors_ukf)*100:.3f}%")
    print("Kalman filters working correctly.")