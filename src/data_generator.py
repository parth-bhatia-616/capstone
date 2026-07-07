

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")



CHEMISTRY_PARAMS = {
    "NMC": {
        "nominal_capacity_Ah": 3.0,
        "v_nominal": 3.7,
        "v_max": 4.2,
        "v_min": 2.8,
        "v_cutoff_charge": 4.2,
        "v_cutoff_discharge": 2.8,
        "R0_base": 0.025,       # Internal resistance (Ohm)
        "capacity_fade_per_cycle": 0.0003,   # Ah per cycle
        "resistance_growth_per_cycle": 0.00005,
        "thermal_sensitivity": 0.015,  # fade multiplier per degC above 25
        "crate_sensitivity": 0.008,
        "calendar_fade_per_day": 0.00002,
        "sei_growth_rate": 0.00015,
        "li_plating_threshold_crate": 2.0,
        "color": "#2196F3",
    },
    "LFP": {
        "nominal_capacity_Ah": 3.2,
        "v_nominal": 3.2,
        "v_max": 3.65,
        "v_min": 2.5,
        "v_cutoff_charge": 3.65,
        "v_cutoff_discharge": 2.5,
        "R0_base": 0.030,
        "capacity_fade_per_cycle": 0.0001,   # Very stable
        "resistance_growth_per_cycle": 0.00003,
        "thermal_sensitivity": 0.008,
        "crate_sensitivity": 0.005,
        "calendar_fade_per_day": 0.000015,
        "sei_growth_rate": 0.00008,
        "li_plating_threshold_crate": 3.0,
        "color": "#4CAF50",
    },
    "NCA": {
        "nominal_capacity_Ah": 3.4,
        "v_nominal": 3.6,
        "v_max": 4.2,
        "v_min": 3.0,
        "v_cutoff_charge": 4.2,
        "v_cutoff_discharge": 3.0,
        "R0_base": 0.020,
        "capacity_fade_per_cycle": 0.00035,
        "resistance_growth_per_cycle": 0.00006,
        "thermal_sensitivity": 0.020,
        "crate_sensitivity": 0.010,
        "calendar_fade_per_day": 0.000025,
        "sei_growth_rate": 0.00020,
        "li_plating_threshold_crate": 1.5,
        "color": "#FF9800",
    },
    "LCO": {
        "nominal_capacity_Ah": 2.8,
        "v_nominal": 3.7,
        "v_max": 4.2,
        "v_min": 3.0,
        "v_cutoff_charge": 4.2,
        "v_cutoff_discharge": 3.0,
        "R0_base": 0.035,
        "capacity_fade_per_cycle": 0.0004,
        "resistance_growth_per_cycle": 0.00007,
        "thermal_sensitivity": 0.018,
        "crate_sensitivity": 0.012,
        "calendar_fade_per_day": 0.00003,
        "sei_growth_rate": 0.00018,
        "li_plating_threshold_crate": 1.8,
        "color": "#9C27B0",
    },
}

FORM_FACTOR_PARAMS = {
    "cylindrical": {"thermal_mass": 1.2, "surface_area": 1.0, "contact_resistance_var": 0.001},
    "prismatic":   {"thermal_mass": 1.5, "surface_area": 1.3, "contact_resistance_var": 0.0008},
    "pouch":       {"thermal_mass": 0.9, "surface_area": 1.6, "contact_resistance_var": 0.0015},
}



def ocv_from_soc(soc: np.ndarray, chemistry: str) -> np.ndarray:
    """
    Open Circuit Voltage as a function of SoC.
    Polynomial fit mimicking real OCV curves per chemistry.
    """
    p = CHEMISTRY_PARAMS[chemistry]
    v_min, v_max = p["v_min"], p["v_max"]
    soc = np.clip(soc, 0.01, 0.99)

    if chemistry == "LFP":
        # LFP has a very flat OCV curve with a plateau
        ocv = (v_min + (v_max - v_min) * (
            0.1 * soc +
            0.85 * (1 / (1 + np.exp(-20 * (soc - 0.1)))) *
                   (1 / (1 + np.exp(20 * (soc - 0.9)))) +
            0.05 * soc**3
        ))
    elif chemistry in ["NMC", "NCA"]:
        # S-shaped curve typical of NMC/NCA
        ocv = (v_min + (v_max - v_min) * (
            -0.8 * soc**3 + 1.4 * soc**2 + 0.4 * soc
        ))
    else:  # LCO
        ocv = (v_min + (v_max - v_min) * (
            soc**0.5 * 0.6 + soc * 0.4
        ))

    return np.clip(ocv, v_min, v_max)




class TheveninECM:
 

    def __init__(self, chemistry: str, capacity_Ah: float, R0: float,
                 R1: float = 0.015, C1: float = 3000.0, temperature: float = 25.0):
        self.chemistry = chemistry
        self.capacity_Ah = capacity_Ah
        self.R0 = R0
        self.R1 = R1
        self.C1 = C1
        self.temperature = temperature
        self.soc = 1.0
        self.v_rc = 0.0 
        self.tau = R1 * C1

    def step(self, current_A: float, dt_s: float) -> Tuple[float, float, float]:
 

        temp_factor = 1 + 0.003 * (25 - self.temperature) 
        R0_eff = self.R0 * temp_factor
        R1_eff = self.R1 * temp_factor


        dsoc = -current_A * dt_s / (self.capacity_Ah * 3600)
        self.soc = np.clip(self.soc + dsoc, 0.0, 1.0)


        self.v_rc = self.v_rc * np.exp(-dt_s / self.tau) + R1_eff * (1 - np.exp(-dt_s / self.tau)) * current_A


        ocv = ocv_from_soc(np.array([self.soc]), self.chemistry)[0]
        v_terminal = ocv - self.v_rc - R0_eff * current_A

        return v_terminal, self.soc, self.v_rc

    def get_params(self) -> Dict:
        return {"R0": self.R0, "R1": self.R1, "C1": self.C1, "tau": self.tau}




def compute_aging_factors(cycle: int, chemistry: str, temperature: float,
                          crate: float, dod: float, calendar_days: float,
                          sei_thickness: float) -> Tuple[float, float, float]:
 
    p = CHEMISTRY_PARAMS[chemistry]


    T_ref = 25.0
    Ea_J_mol = 30000  
    R_gas = 8.314
    T_K = temperature + 273.15
    T_ref_K = T_ref + 273.15
    arrhenius = np.exp(Ea_J_mol / R_gas * (1 / T_ref_K - 1 / T_K))


    crate_stress = 1 + p["crate_sensitivity"] * max(0, crate - 0.5) ** 1.5


    dod_stress = 1 + 0.5 * (dod - 0.8) if dod > 0.8 else 1.0


    li_plating = 0.0
    if crate > p["li_plating_threshold_crate"] and temperature < 15:
        li_plating = 0.0003 * (crate - p["li_plating_threshold_crate"]) ** 2


    cycle_fade = (p["capacity_fade_per_cycle"] * crate_stress * dod_stress * arrhenius + li_plating)


    cal_fade = p["calendar_fade_per_day"] * calendar_days * arrhenius


    new_sei = p["sei_growth_rate"] * np.sqrt(max(0, cycle + 1)) * arrhenius / 100


    res_growth = p["resistance_growth_per_cycle"] * crate_stress * arrhenius

    total_capacity_fade = cycle_fade + cal_fade
    return total_capacity_fade, res_growth, new_sei




class BatterySimulator:
   

    def __init__(self, chemistry: str = "NMC", form_factor: str = "cylindrical",
                 battery_id: str = "BAT_001", seed: int = 42):
        self.chemistry = chemistry
        self.form_factor = form_factor
        self.battery_id = battery_id
        self.rng = np.random.default_rng(seed)
        self.p = CHEMISTRY_PARAMS[chemistry]
        self.ff = FORM_FACTOR_PARAMS[form_factor]

        # Initial state
        self.nominal_capacity = self.p["nominal_capacity_Ah"]
        self.current_capacity = self.nominal_capacity
        self.R0 = self.p["R0_base"] + self.rng.normal(0, self.ff["contact_resistance_var"])
        self.sei_thickness = 0.0
        self.cycle_count = 0
        self.calendar_days = 0.0

    def simulate_cc_cv_charge(self, temperature: float, crate: float, dt_s: float = 10.0) -> Dict:
        """Simulate CC-CV charge cycle. Returns cycle metrics."""
        ecm = TheveninECM(self.chemistry, self.current_capacity, self.R0,
                          temperature=temperature)
        ecm.soc = 0.1  # start from low SoC

        v_cutoff = self.p["v_cutoff_charge"]
        I_charge = -crate * self.current_capacity  # negative = charging

        voltages, socs, currents, times = [], [], [], []
        t = 0.0
        phase = "CC"
        charge_Ah = 0.0

        while t < 7200:  # max 2 hours
            if phase == "CC":
                I = I_charge
            else:  # CV
                # Taper current (simplified)
                I = I_charge * (v_cutoff - ecm.v_rc) / v_cutoff
                I = max(I, I_charge * 0.05)  # stop at C/20

            v, soc, v_rc = ecm.step(I, dt_s)
            charge_Ah += abs(I) * dt_s / 3600

            voltages.append(v)
            socs.append(soc)
            currents.append(I)
            times.append(t)

            if phase == "CC" and v >= v_cutoff:
                phase = "CV"
            if phase == "CV" and abs(I) < abs(I_charge) * 0.05:
                break

            t += dt_s

        return {
            "voltages": np.array(voltages),
            "socs": np.array(socs),
            "currents": np.array(currents),
            "times": np.array(times),
            "charge_Ah": charge_Ah,
        }

    def simulate_discharge(self, temperature: float, crate: float, dt_s: float = 10.0) -> Dict:
        """Simulate constant-current discharge. Returns cycle metrics."""
        ecm = TheveninECM(self.chemistry, self.current_capacity, self.R0,
                          temperature=temperature)
        ecm.soc = 0.99

        v_cutoff = self.p["v_cutoff_discharge"]
        I_discharge = crate * self.current_capacity  # positive = discharging

        voltages, socs, currents, times = [], [], [], []
        t = 0.0
        discharge_Ah = 0.0
        dod = 0.0

        while t < 7200:
            v, soc, v_rc = ecm.step(I_discharge, dt_s)
            discharge_Ah += I_discharge * dt_s / 3600
            dod = 1 - soc

            voltages.append(v)
            socs.append(soc)
            currents.append(I_discharge)
            times.append(t)

            if v <= v_cutoff or soc <= 0.01:
                break
            t += dt_s

        return {
            "voltages": np.array(voltages),
            "socs": np.array(socs),
            "currents": np.array(currents),
            "times": np.array(times),
            "discharge_Ah": discharge_Ah,
            "dod": dod,
        }

    def run_cycle(self, temperature: float, charge_crate: float,
                  discharge_crate: float, aging_mode: str = "cycle",
                  calendar_days_add: float = 1.0) -> Dict:
       
        # Discharge
        discharge = self.simulate_discharge(temperature, discharge_crate)
        measured_capacity = discharge["discharge_Ah"]

        # Apply aging
        dod = discharge["dod"]
        fade, res_growth, sei = compute_aging_factors(
            self.cycle_count, self.chemistry, temperature,
            max(charge_crate, discharge_crate), dod,
            self.calendar_days, self.sei_thickness
        )

        # Add noise
        noise_cap = self.rng.normal(0, 0.0005)
        noise_res = self.rng.normal(0, 0.0001)

        self.current_capacity = max(0.1, self.current_capacity - fade * self.nominal_capacity + noise_cap)
        self.R0 = min(0.5, self.R0 + res_growth + noise_res)
        self.sei_thickness += sei
        self.cycle_count += 1
        self.calendar_days += calendar_days_add

        # Compute SoH
        soh = (self.current_capacity / self.nominal_capacity) * 100

        # ── Feature Engineering on this cycle ──────────────────────────────
        v_arr = discharge["voltages"]
        soc_arr = discharge["socs"]
        i_arr = discharge["currents"]
        t_arr = discharge["times"]

        # Relaxation voltage (last 5 points after discharge)
        v_relaxation = v_arr[-1] if len(v_arr) > 0 else 0.0

        # IC curve features (dQ/dV)
        ic_features = self._compute_ic_features(v_arr, soc_arr, measured_capacity)

        # DV curve features (dV/dQ)
        dv_features = self._compute_dv_features(v_arr, soc_arr, measured_capacity)

        # EIS-inspired features (estimate from pulse response)
        eis_features = self._compute_eis_features()

        # Partial charge features
        partial_features = self._compute_partial_features(v_arr, soc_arr)

        # Energy & throughput
        energy_Wh = np.trapezoid(np.abs(v_arr * i_arr), t_arr) / 3600 if len(t_arr) > 1 else 0.0
        capacity_throughput = self.cycle_count * measured_capacity * 2  # charge + discharge

        # Operating temperature histogram (simplified as mean/std)
        temp_mean = temperature + self.rng.normal(0, 0.5)
        temp_std = self.rng.uniform(0.5, 2.0)

        # DoD distribution
        dod_mean = dod + self.rng.normal(0, 0.02)
        crate_mean = (charge_crate + discharge_crate) / 2

        row = {
            # Identifiers
            "battery_id": self.battery_id,
            "chemistry": self.chemistry,
            "form_factor": self.form_factor,
            "cycle": self.cycle_count,
            "aging_mode": aging_mode,

            # Operating conditions
            "temperature": temperature,
            "temp_mean": temp_mean,
            "temp_std": temp_std,
            "charge_crate": charge_crate,
            "discharge_crate": discharge_crate,
            "crate_mean": crate_mean,
            "dod_mean": dod_mean,
            "calendar_days": self.calendar_days,

            # Basic measurements
            "measured_capacity_Ah": measured_capacity,
            "nominal_capacity_Ah": self.nominal_capacity,
            "internal_resistance": self.R0,
            "sei_thickness_nm": self.sei_thickness * 1e9,

            # Voltage features
            "v_mean": np.mean(v_arr) if len(v_arr) > 0 else 0.0,
            "v_min": np.min(v_arr) if len(v_arr) > 0 else 0.0,
            "v_max": np.max(v_arr) if len(v_arr) > 0 else 0.0,
            "v_std": np.std(v_arr) if len(v_arr) > 0 else 0.0,
            "v_relaxation": v_relaxation,
            "v_drop_start": v_arr[0] - v_arr[-1] if len(v_arr) > 1 else 0.0,

            # Energy
            "energy_Wh": energy_Wh,
            "energy_throughput_Wh": capacity_throughput * self.p["v_nominal"],
            "capacity_throughput_Ah": capacity_throughput,

            # IC/DV curve features
            **ic_features,
            **dv_features,
            **eis_features,
            **partial_features,

            # Targets
            "soh": soh,
            # SoC at end of discharge (should be ~0, but noise makes it interesting)
            "soc_true": float(soc_arr[-1]) if len(soc_arr) > 0 else 0.0,
        }

        return row

    def _compute_ic_features(self, v_arr, soc_arr, capacity_Ah) -> Dict:
        """Incremental Capacity dQ/dV features."""
        if len(v_arr) < 10:
            return {"ic_peak_height": 0, "ic_peak_voltage": 0, "ic_peak_width": 0,
                    "ic_area": 0, "ic_valley_depth": 0}
        try:
            dv = np.diff(v_arr)
            dq = np.diff(soc_arr) * capacity_Ah
            mask = np.abs(dv) > 1e-6
            ic = np.zeros(len(dv))
            ic[mask] = dq[mask] / dv[mask]
            ic = np.abs(ic)
            # Smooth
            from scipy.signal import savgol_filter
            ic_smooth = savgol_filter(ic, min(11, len(ic) - (len(ic) % 2 == 0)), 3)
            peak_idx = np.argmax(ic_smooth)
            peak_h = ic_smooth[peak_idx]
            peak_v = v_arr[peak_idx] if peak_idx < len(v_arr) else 0.0
            # Peak width (FWHM)
            half_max = peak_h / 2
            above = ic_smooth > half_max
            if above.any():
                width = np.sum(above) * (v_arr[-1] - v_arr[0]) / len(v_arr)
            else:
                width = 0.0
            return {
                "ic_peak_height": float(peak_h),
                "ic_peak_voltage": float(peak_v),
                "ic_peak_width": float(width),
                "ic_area": float(np.trapezoid(ic_smooth, v_arr[:-1])),
                "ic_valley_depth": float(np.min(ic_smooth)),
            }
        except Exception:
            return {"ic_peak_height": 0, "ic_peak_voltage": 0, "ic_peak_width": 0,
                    "ic_area": 0, "ic_valley_depth": 0}

    def _compute_dv_features(self, v_arr, soc_arr, capacity_Ah) -> Dict:
        """Differential Voltage dV/dQ features."""
        if len(v_arr) < 10:
            return {"dv_peak_height": 0, "dv_peak_soc": 0, "dv_slope_early": 0, "dv_slope_late": 0}
        try:
            dq = np.diff(soc_arr) * capacity_Ah
            dv = np.diff(v_arr)
            mask = np.abs(dq) > 1e-8
            dv_dq = np.zeros(len(dq))
            dv_dq[mask] = dv[mask] / dq[mask]
            dv_dq = np.abs(dv_dq)
            peak_idx = np.argmax(dv_dq)
            peak_soc = soc_arr[peak_idx] if peak_idx < len(soc_arr) else 0.0
            n = len(dv_dq)
            slope_early = np.mean(dv_dq[:n // 4]) if n > 4 else 0.0
            slope_late = np.mean(dv_dq[3 * n // 4:]) if n > 4 else 0.0
            return {
                "dv_peak_height": float(np.max(dv_dq)),
                "dv_peak_soc": float(peak_soc),
                "dv_slope_early": float(slope_early),
                "dv_slope_late": float(slope_late),
            }
        except Exception:
            return {"dv_peak_height": 0, "dv_peak_soc": 0, "dv_slope_early": 0, "dv_slope_late": 0}

    def _compute_eis_features(self) -> Dict:
        """
        EIS-inspired impedance features estimated from ECM parameters.
        In real hardware: measured directly. Here: derived from R0, R1, C1.
        """
        r0 = self.R0
        # Estimate charge transfer resistance from SEI growth
        r_ct = 0.015 + self.sei_thickness * 0.02
        # Warburg element (diffusion) grows with aging
        z_w = 0.005 + self.sei_thickness * 0.01
        # Estimated phase angle at mid-frequency
        phase_angle = -np.arctan(1 / (2 * np.pi * 10 * r_ct * 500)) * 180 / np.pi

        return {
            "eis_R0": float(r0),
            "eis_Rct": float(r_ct),
            "eis_Zw": float(z_w),
            "eis_phase_angle": float(phase_angle),
            "eis_total_impedance": float(r0 + r_ct + z_w),
        }

    def _compute_partial_features(self, v_arr, soc_arr) -> Dict:
        """Features from partial charge windows (useful for online estimation)."""
        if len(v_arr) < 20:
            return {"partial_v_10_80_range": 0, "partial_capacity_10_80": 0,
                    "partial_v_slope": 0}
        try:
            soc_10 = 0.1
            soc_80 = 0.8
            mask = (soc_arr >= soc_10) & (soc_arr <= soc_80)
            if mask.sum() > 5:
                v_range = v_arr[mask][-1] - v_arr[mask][0]
                q_range = (soc_arr[mask][-1] - soc_arr[mask][0])
                slope = v_range / (q_range + 1e-8)
            else:
                v_range = 0.0
                q_range = 0.0
                slope = 0.0
            return {
                "partial_v_10_80_range": float(v_range),
                "partial_capacity_10_80": float(q_range),
                "partial_v_slope": float(slope),
            }
        except Exception:
            return {"partial_v_10_80_range": 0, "partial_capacity_10_80": 0,
                    "partial_v_slope": 0}




def generate_dataset(
    n_batteries_per_chemistry: int = 5,
    n_cycles_per_battery: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
   
    rng = np.random.default_rng(seed)
    all_rows = []
    battery_counter = 0

    chemistries = ["NMC", "LFP", "NCA", "LCO"]
    form_factors = ["cylindrical", "prismatic", "pouch"]
    aging_modes = ["cycle_aging", "calendar_aging", "fast_charge_stress",
                   "deep_discharge_stress", "thermal_abuse", "mixed_usage"]

    temperatures_pool = [0, 10, 15, 25, 35, 45, 55, 60]
    crates_pool = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    for chem in chemistries:
        for bat_i in range(n_batteries_per_chemistry):
            battery_counter += 1
            ff = rng.choice(form_factors)
            bat_id = f"{chem}_{ff[:3].upper()}_{battery_counter:03d}"
            aging_mode = rng.choice(aging_modes)

            # Per-battery operating profile
            if aging_mode == "fast_charge_stress":
                base_charge_crate = rng.uniform(3.0, 5.0)
                base_discharge_crate = rng.uniform(0.5, 1.5)
            elif aging_mode == "deep_discharge_stress":
                base_charge_crate = rng.uniform(0.5, 1.0)
                base_discharge_crate = rng.uniform(2.0, 4.0)
            elif aging_mode == "thermal_abuse":
                base_charge_crate = rng.uniform(1.0, 2.0)
                base_discharge_crate = rng.uniform(1.0, 2.0)
            elif aging_mode == "calendar_aging":
                base_charge_crate = rng.uniform(0.5, 0.5)
                base_discharge_crate = rng.uniform(0.5, 0.5)
            elif aging_mode == "mixed_usage":
                base_charge_crate = rng.uniform(1.0, 3.0)
                base_discharge_crate = rng.uniform(1.0, 3.0)
            else:  # cycle_aging
                base_charge_crate = rng.uniform(0.5, 1.5)
                base_discharge_crate = rng.uniform(0.5, 1.5)

            if aging_mode == "thermal_abuse":
                base_temp = rng.uniform(45, 60)
            else:
                base_temp = rng.uniform(15, 35)

            sim = BatterySimulator(chem, ff, bat_id, seed=seed * 100 + battery_counter)

            for cycle_i in range(n_cycles_per_battery):
                # Random walk variations per cycle
                temp = np.clip(base_temp + rng.normal(0, 3), -5, 65)
                c_crate = np.clip(base_charge_crate + rng.normal(0, 0.1), 0.3, 5.0)
                d_crate = np.clip(base_discharge_crate + rng.normal(0, 0.1), 0.3, 5.0)

                row = sim.run_cycle(
                    temperature=temp,
                    charge_crate=c_crate,
                    discharge_crate=d_crate,
                    aging_mode=aging_mode,
                    calendar_days_add=rng.uniform(0.5, 2.0),
                )
                all_rows.append(row)

                # Stop if battery is dead
                if row["soh"] < 55:
                    break

    df = pd.DataFrame(all_rows)
    print(f"Generated dataset: {len(df)} rows, {df['battery_id'].nunique()} batteries")
    print(f"Chemistries: {df['chemistry'].value_counts().to_dict()}")
    print(f"SoH range: {df['soh'].min():.1f}% – {df['soh'].max():.1f}%")
    return df


if __name__ == "__main__":
    df = generate_dataset(n_batteries_per_chemistry=3, n_cycles_per_battery=200)
    df.to_csv("/home/claude/battery_bms_advanced/data/synthetic/battery_aging_dataset.csv", index=False)
    print("Saved to data/synthetic/battery_aging_dataset.csv")
    print(df.describe())