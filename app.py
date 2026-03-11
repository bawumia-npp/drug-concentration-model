# app.py
"""
Modeling Drug Concentration in the Bloodstream Using Python (Clinical UI)
------------------------------------------------------------------------
A clinically-oriented Streamlit interface for first-order elimination PK.

Models included (all assume first-order elimination):
1) Single dose (decay):
   C(t) = C0 * exp(-k*t)

2) Multiple dosing (superposition, instantaneous dose increases):
   C(t) = sum over doses [ C0 * exp(-k*(t - t_dose)) ] for t >= t_dose

3) IV infusion (constant input during infusion, first-order elimination):
   For 0 <= t <= T_inf:    C(t) = (R0/k) * (1 - exp(-k*t))
   For t > T_inf:         C(t) = C(T_inf) * exp(-k*(t - T_inf))

Notes:
- Units are user-defined but must be consistent (e.g., mg/L and hours).
- This is a teaching/demo tool, not medical advice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


class PharmacokineticModel:
    """
    Clinically-oriented first-order elimination PK simulator.

    The UI chooses one of:
    - single_dose_decay
    - multiple_dose_superposition
    - iv_infusion

    Therapeutic window evaluation is handled via metrics functions.
    """

    def __init__(
        self,
        k: float,
        total_time: float,
        n_points: int,
        time_unit: str,
        concentration_unit: str,
    ) -> None:
        self.k = float(k)
        self.total_time = float(total_time)
        self.n_points = int(n_points)
        self.time_unit = time_unit
        self.concentration_unit = concentration_unit
        self.time = np.linspace(0.0, self.total_time, self.n_points)

    def half_life(self) -> float:
        if self.k <= 0:
            return np.nan
        return float(np.log(2) / self.k)

    def single_dose_decay(self, c0: float) -> np.ndarray:
        c0 = float(c0)
        return c0 * np.exp(-self.k * self.time)

    def multiple_dose_superposition(
        self,
        c0_per_dose: float,
        tau: float,
        n_doses: int,
        start_time: float = 0.0,
    ) -> np.ndarray:
        """
        Multiple dosing with instantaneous dose increments (superposition).
        Each dose adds "c0_per_dose" at dosing times, then decays.
        """
        c0_per_dose = float(c0_per_dose)
        tau = float(tau)
        n_doses = int(n_doses)
        start_time = float(start_time)

        dose_times = start_time + np.arange(n_doses) * tau
        conc = np.zeros_like(self.time, dtype=float)

        for t_dose in dose_times:
            mask = self.time >= t_dose
            conc[mask] += c0_per_dose * np.exp(-self.k * (self.time[mask] - t_dose))

        return conc

    def iv_infusion(
        self,
        r0: float,
        infusion_duration: float,
    ) -> np.ndarray:
        """
        IV infusion with constant rate R0 for T_inf, then stop.
        Concentration units are user-defined; R0 is in concentration/time.
        """
        r0 = float(r0)
        t_inf = float(infusion_duration)

        conc = np.zeros_like(self.time, dtype=float)

        if self.k <= 0:
            return np.full_like(self.time, np.nan, dtype=float)

        for i, t in enumerate(self.time):
            if t <= t_inf:
                conc[i] = (r0 / self.k) * (1.0 - np.exp(-self.k * t))
            else:
                c_end = (r0 / self.k) * (1.0 - np.exp(-self.k * t_inf))
                conc[i] = c_end * np.exp(-self.k * (t - t_inf))

        return conc

    def to_dataframe(self, concentration: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                f"Time ({self.time_unit})": self.time,
                f"Concentration ({self.concentration_unit})": concentration,
            }
        )

    def auc_trapezoid(self, concentration: np.ndarray) -> float:
        # AUC over the simulated interval
        return float(np.trapezoid(concentration, self.time))

    def summarize_clinical_metrics(
        self,
        concentration: np.ndarray,
        mec: float | None,
        mtc: float | None,
    ) -> dict:
        """
        MEC = minimum effective concentration
        MTC = minimum toxic concentration (upper bound)
        """
        conc = np.asarray(concentration, dtype=float)
        t_half = self.half_life()
        auc = self.auc_trapezoid(conc)

        cmax = float(np.nanmax(conc))
        tmax = float(self.time[int(np.nanargmax(conc))])
        cmin = float(np.nanmin(conc))
        tmin = float(self.time[int(np.nanargmin(conc))])

        metrics: dict[str, float] = {
            "Half-life": float(t_half),
            "Cmax": cmax,
            "Tmax": tmax,
            "Cmin": cmin,
            "Tmin": tmin,
            "AUC (0–T)": float(auc),
        }

        # Time-in-range calculations (simple sampling approach)
        dt = self.time[1] - self.time[0] if len(self.time) > 1 else np.nan

        if mec is not None:
            mec_val = float(mec)
            time_below = float(np.sum(conc < mec_val) * dt)
            metrics["Time below MEC"] = time_below

        if mtc is not None:
            mtc_val = float(mtc)
            time_above = float(np.sum(conc > mtc_val) * dt)
            metrics["Time above MTC"] = time_above

        if mec is not None and mtc is not None:
            mec_val = float(mec)
            mtc_val = float(mtc)
            time_in = float(np.sum((conc >= mec_val) & (conc <= mtc_val)) * dt)
            metrics["Time in therapeutic range"] = time_in

        return metrics

    def plot(
        self,
        concentration: np.ndarray,
        mec: float | None,
        mtc: float | None,
        title: str,
    ) -> plt.Figure:
        df = self.to_dataframe(concentration)
        time_col = df.columns[0]
        conc_col = df.columns[1]

        fig, ax = plt.subplots()
        ax.plot(df[time_col], df[conc_col], linewidth=2)

        if mec is not None:
            ax.axhline(float(mec), linestyle="--")
            ax.text(
                0.01,
                float(mec),
                " MEC",
                va="bottom",
            )

        if mtc is not None:
            ax.axhline(float(mtc), linestyle="--")
            ax.text(
                0.01,
                float(mtc),
                " MTC",
                va="bottom",
            )

        ax.set_title(title)
        ax.set_xlabel(time_col)
        ax.set_ylabel(conc_col)
        ax.grid(True)
        return fig

    def interpretation(
        self,
        concentration: np.ndarray,
        mec: float | None,
        mtc: float | None,
    ) -> str:
        m = self.summarize_clinical_metrics(concentration, mec=mec, mtc=mtc)

        lines = []
        lines.append(
            f"Half-life is approximately {m['Half-life']:.2f} {self.time_unit} "
            f"(k = {self.k:.4f} 1/{self.time_unit})."
        )
        lines.append(
            f"Peak concentration (Cmax) is {m['Cmax']:.2f} {self.concentration_unit} "
            f"at {m['Tmax']:.2f} {self.time_unit}."
        )
        lines.append(
            f"AUC over 0–{self.total_time:.2f} {self.time_unit} is {m['AUC (0–T)']:.2f} "
            f"{self.concentration_unit}·{self.time_unit}."
        )

        if mec is not None:
            lines.append(
                f"Time below MEC ({float(mec):.2f} {self.concentration_unit}): "
                f"{m['Time below MEC']:.2f} {self.time_unit}."
            )
        if mtc is not None:
            lines.append(
                f"Time above MTC ({float(mtc):.2f} {self.concentration_unit}): "
                f"{m['Time above MTC']:.2f} {self.time_unit}."
            )
        if mec is not None and mtc is not None:
            lines.append(
                f"Time in therapeutic range: {m['Time in therapeutic range']:.2f} {self.time_unit}."
            )

        if mec is not None and float(np.nanmax(concentration)) < float(mec):
            lines.append("Clinical note: Concentrations never reach MEC (likely subtherapeutic).")
        if mtc is not None and float(np.nanmax(concentration)) > float(mtc):
            lines.append("Clinical note: Concentrations exceed MTC at some point (toxicity risk).")

        return " ".join(lines)


def validate_inputs(k: float, total_time: float, n_points: int) -> list[str]:
    errors: list[str] = []
    if k <= 0:
        errors.append("Elimination rate constant (k) must be > 0.")
    if total_time <= 0:
        errors.append("Total simulation time must be > 0.")
    if n_points < 10:
        errors.append("Number of time points should be at least 10 for stable plots.")
    return errors


def main() -> None:
    st.set_page_config(page_title="Clinical PK Simulator", layout="wide")

    st.title("Clinical Drug Concentration Simulator (First-Order PK)")
    st.caption(
        "Educational tool for pharmacokinetics visualization. Not medical advice."
    )

    # ---- Sidebar: global settings
    st.sidebar.header("Global Settings")
    time_unit = st.sidebar.selectbox("Time unit", ["hours", "days"], index=0)
    concentration_unit = st.sidebar.selectbox("Concentration unit", ["mg/L", "µg/mL", "ng/mL"], index=0)

    k = st.sidebar.number_input(
        f"Elimination rate constant k (1/{time_unit})",
        min_value=0.0001,
        value=0.15,
        step=0.01,
        format="%.4f",
    )

    total_time = st.sidebar.number_input(
        f"Total simulation time ({time_unit})",
        min_value=0.1,
        value=24.0,
        step=1.0,
    )

    n_points = st.sidebar.slider("Number of time points", 50, 2000, 400, 50)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Therapeutic Targets (optional)")
    use_targets = st.sidebar.checkbox("Overlay therapeutic window (MEC/MTC)", value=True)

    mec = None
    mtc = None
    if use_targets:
        mec = st.sidebar.number_input(
            f"MEC (min effective) [{concentration_unit}]",
            min_value=0.0,
            value=10.0,
            step=1.0,
        )
        mtc = st.sidebar.number_input(
            f"MTC (min toxic) [{concentration_unit}]",
            min_value=0.0,
            value=30.0,
            step=1.0,
        )
        if mtc <= mec:
            st.sidebar.warning("MTC should be greater than MEC for a therapeutic window.")

    st.sidebar.markdown("---")
    model_mode = st.sidebar.radio(
        "Dosing / Scenario",
        ["Single dose (decay)", "Multiple dosing (qτ)", "IV infusion"],
        index=1,
    )

    # ---- Model-specific inputs
    if model_mode == "Single dose (decay)":
        c0 = st.sidebar.number_input(
            f"Initial concentration C₀ [{concentration_unit}]",
            min_value=0.0,
            value=100.0,
            step=5.0,
        )
        scenario_inputs = {"c0": c0}

    elif model_mode == "Multiple dosing (qτ)":
        c0_per_dose = st.sidebar.number_input(
            f"Concentration jump per dose (C₀ per dose) [{concentration_unit}]",
            min_value=0.0,
            value=50.0,
            step=5.0,
            help="Teaching approximation: each dose adds an instantaneous concentration increment.",
        )
        tau = st.sidebar.number_input(
            f"Dosing interval τ ({time_unit})",
            min_value=0.1,
            value=8.0,
            step=1.0,
        )
        n_doses = st.sidebar.slider("Number of doses", 1, 30, 8, 1)
        start_time = st.sidebar.number_input(
            f"First dose time ({time_unit})",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )
        scenario_inputs = {"c0_per_dose": c0_per_dose, "tau": tau, "n_doses": n_doses, "start_time": start_time}

    else:  # IV infusion
        r0 = st.sidebar.number_input(
            f"Infusion input rate R₀ [{concentration_unit}/{time_unit}]",
            min_value=0.0,
            value=15.0,
            step=1.0,
            help="Teaching input rate in concentration/time.",
        )
        infusion_duration = st.sidebar.number_input(
            f"Infusion duration ({time_unit})",
            min_value=0.1,
            value=4.0,
            step=0.5,
        )
        scenario_inputs = {"r0": r0, "infusion_duration": infusion_duration}

    run_button = st.sidebar.button("Run Simulation", type="primary")

    if not run_button:
        st.info("Set inputs in the sidebar, then click **Run Simulation**.")
        return

    errors = validate_inputs(k=k, total_time=total_time, n_points=n_points)
    if errors:
        st.error("Fix these issues:")
        for e in errors:
            st.write(f"- {e}")
        return

    model = PharmacokineticModel(
        k=k,
        total_time=total_time,
        n_points=n_points,
        time_unit=time_unit,
        concentration_unit=concentration_unit,
    )

    # ---- Compute concentration based on scenario
    if model_mode == "Single dose (decay)":
        concentration = model.single_dose_decay(c0=scenario_inputs["c0"])
        title = "Single Dose: First-Order Elimination (Decay)"

    elif model_mode == "Multiple dosing (qτ)":
        concentration = model.multiple_dose_superposition(
            c0_per_dose=scenario_inputs["c0_per_dose"],
            tau=scenario_inputs["tau"],
            n_doses=scenario_inputs["n_doses"],
            start_time=scenario_inputs["start_time"],
        )
        title = "Multiple Dosing (qτ): Superposition with First-Order Elimination"

    else:
        concentration = model.iv_infusion(
            r0=scenario_inputs["r0"],
            infusion_duration=scenario_inputs["infusion_duration"],
        )
        title = "IV Infusion: Input During Infusion + First-Order Elimination"

    # ---- Layout
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Key Clinical Metrics")
        metrics = model.summarize_clinical_metrics(concentration, mec=mec, mtc=mtc)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Half-life ({time_unit})", f"{metrics['Half-life']:.2f}")
        c2.metric(f"Cmax ({concentration_unit})", f"{metrics['Cmax']:.2f}")
        c3.metric(f"Tmax ({time_unit})", f"{metrics['Tmax']:.2f}")
        c4.metric(f"AUC ({concentration_unit}·{time_unit})", f"{metrics['AUC (0–T)']:.2f}")

        if "Time in therapeutic range" in metrics:
            st.metric(
                f"Time in therapeutic range ({time_unit})",
                f"{metrics['Time in therapeutic range']:.2f}",
            )

        # Simple alerts
        if mec is not None and float(np.nanmax(concentration)) < float(mec):
            st.warning("Subtherapeutic: peak concentration does not reach MEC.")
        if mtc is not None and float(np.nanmax(concentration)) > float(mtc):
            st.error("Potential toxicity: concentration exceeds MTC during simulation.")

        st.subheader("Clinical Interpretation")
        st.write(model.interpretation(concentration, mec=mec, mtc=mtc))

    with right:
        st.subheader("Concentration–Time Curve")
        fig = model.plot(concentration, mec=mec, mtc=mtc, title=title)
        st.pyplot(fig, clear_figure=True)

    st.subheader("Results Table")
    df = model.to_dataframe(concentration)
    df_display = df.copy()
    df_display[df.columns[0]] = df_display[df.columns[0]].round(4)
    df_display[df.columns[1]] = df_display[df.columns[1]].round(6)
    st.dataframe(df_display, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results (CSV)",
        data=csv_data,
        file_name="pk_simulation_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()