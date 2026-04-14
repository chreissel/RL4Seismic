"""
Extended Kalman filter baseline — model-based nonlinear tracker.

This is the theoretically optimal online nonlinear estimator when the
coupling model structure is known but its parameters drift.  Unlike NLMS
and Volterra (which are blind adaptive), the EKF is given the physical
structure of the coupling and tracks the drifting parameters via a
linearised Kalman update.

Model
-----
We assume the known structure:

    y[t]  =  h·w_x[t]  +  T[t] · θ_y[t] · w_x[t]  +  n[t]

with hidden state

    z[t] = [ c,         scalar linear-FIR gain (placeholder for h⊛w_x)
             T[t] ]     scalar tilt-horizontal coupling gain

The linear FIR coupling  h ⊛ w_x  is approximated by a scalar
multiplicative gain  c · w_x[t]  — a deliberately crude simplification that
lets the EKF focus on tracking the drifting T(t), which is where its
advantage over NLMS/Volterra actually lives.  A more general EKF would
track the full FIR tap vector, at the cost of a much larger state; the
extra linear-modelling power would not change the bilinear-regime picture.

The tilt proxy θ_y[t] is computed online via a double leaky integrator of
witness_y (exactly the same formula used by the simulator).  Knowing its
structure is the "model" the EKF exploits.

Dynamics
--------
Both c and T drift as Ornstein–Uhlenbeck processes toward prior means
(c₀, T₀), with process-noise variances Q_c and Q_T that encode how fast
they are allowed to change.  The smaller these are, the more the EKF
relies on prior structure; the larger, the more it tracks instantaneous
changes.

Measurement update
------------------
At each step:

    y_hat = c · w_x[t] + T · θ_y[t] · w_x[t]
    H     = [ w_x[t],  θ_y[t] · w_x[t] ]
    S     = H P Hᵀ + R
    K     = P Hᵀ / S
    z     ← z + K · (y − y_hat)
    P     ← (I − K H) P

This is a standard scalar EKF update; the nonlinearity is in the product
structure, which is handled by evaluating the measurement Jacobian H at
the current estimate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class EKFFilter:
    """
    Extended Kalman filter for T2L-aware noise cancellation.

    Parameters
    ----------
    fs                : sample rate (Hz).
    tilt_leak_timescale : time constant of each of the two cascaded leaky-
                          integrator stages used to form the θ_y tilt proxy.
                          Must match the simulator config.
    t2l_prior          : prior mean of T(t) (= cfg.t2l_gain).
    t2l_thermal_timescale : OU timescale of T (= cfg.t2l_thermal_timescale).
    t2l_drift_sigma    : OU stationary std of T (= cfg.t2l_gain_drift_sigma).
    linear_prior       : prior mean of the scalar linear gain c (default 0).
    linear_thermal_timescale : OU timescale of c (default 600 s).
    linear_drift_sigma : OU stationary std of c (default 0.5).
    measurement_sigma  : measurement noise std (≈ cfg.sensor_noise_sigma).
    """

    def __init__(
        self,
        fs: float = 4.0,
        tilt_leak_timescale: float = 2.0,
        t2l_prior: float = 43.0,
        t2l_thermal_timescale: float = 600.0,
        t2l_drift_sigma: float = 2.7,
        linear_prior: float = 0.0,
        linear_thermal_timescale: float = 600.0,
        linear_drift_sigma: float = 0.5,
        measurement_sigma: float = 0.05,
    ):
        self.fs = float(fs)
        self.alpha_leak = 1.0 - np.exp(-1.0 / (fs * tilt_leak_timescale))

        # Prior means
        self.c_prior = float(linear_prior)
        self.T_prior = float(t2l_prior)

        # OU decay factors per step (exact discrete-time OU)
        dt = 1.0 / fs
        self.c_decay = float(np.exp(-dt / linear_thermal_timescale))
        self.T_decay = float(np.exp(-dt / t2l_thermal_timescale))

        # OU process-noise variances chosen so stationary std matches drift_sigma
        self.Q_c = float(linear_drift_sigma ** 2 * (1.0 - self.c_decay ** 2))
        self.Q_T = float(t2l_drift_sigma ** 2 * (1.0 - self.T_decay ** 2))

        # Measurement-noise variance
        self.R = float(measurement_sigma ** 2)

    # ------------------------------------------------------------------
    # Online filtering
    # ------------------------------------------------------------------

    def run(
        self,
        witness_x: np.ndarray,
        main: np.ndarray,
        witness_y: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the EKF online to the full evaluation trace.

        Returns the cleaned main channel (main minus predicted coupling).
        """
        n = len(main)
        assert len(witness_x) == n and len(witness_y) == n

        # State:   z = [c, T]
        c = self.c_prior
        T = self.T_prior
        # Covariance: initialise broad so first measurements dominate
        P = np.diag([1.0, 25.0])  # (2, 2)

        # Double-leaky-integrator state for θ_y (matches signals._double_leaky_integrator)
        alpha = self.alpha_leak
        leaky1 = 0.0
        leaky2 = 0.0

        cleaned = np.empty(n, dtype=np.float64)

        Q = np.diag([self.Q_c, self.Q_T])
        F_dyn = np.diag([self.c_decay, self.T_decay])
        c_drift_mean_coeff = 1.0 - self.c_decay  # c ← decay*c + (1-decay)*c_prior
        T_drift_mean_coeff = 1.0 - self.T_decay

        for t in range(n):
            wx = float(witness_x[t])
            wy = float(witness_y[t])

            # --- update tilt proxy (double leaky integrator of w_y) ---
            leaky1 = (1.0 - alpha) * leaky1 + alpha * wy
            leaky2 = (1.0 - alpha) * leaky2 + alpha * leaky1
            theta_y = leaky2

            # --- time update: OU drift toward priors ---
            c = self.c_decay * c + c_drift_mean_coeff * self.c_prior
            T = self.T_decay * T + T_drift_mean_coeff * self.T_prior
            P = F_dyn @ P @ F_dyn.T + Q

            # --- measurement prediction and Jacobian ---
            y_hat = c * wx + T * theta_y * wx
            H = np.array([[wx, theta_y * wx]])        # (1, 2)

            # --- residual and innovation covariance ---
            y_meas = float(main[t])
            innov = y_meas - y_hat
            S = float((H @ P @ H.T)[0, 0]) + self.R

            # --- Kalman gain and state update ---
            K_gain = (P @ H.T).flatten() / S           # (2,)
            c = c + K_gain[0] * innov
            T = T + K_gain[1] * innov

            # --- covariance update ---
            I2 = np.eye(2)
            P = (I2 - np.outer(K_gain, H.flatten())) @ P

            # Cleaned output: subtract the posterior-model coupling estimate
            y_hat_post = c * wx + T * theta_y * wx
            cleaned[t] = y_meas - y_hat_post

        return cleaned
