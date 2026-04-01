"""
Least Mean Squares (LMS / NLMS) adaptive filter — linear baseline.

The LMS filter maintains a weight vector w ∈ R^M and at each step predicts
the coupling as  â_t = w^T · x_t,  where x_t is a window of recent witness
samples.  Weights are updated online via  w ← w + μ · e_t · x_t,  where
e_t = y_t − â_t  is the residual.

This is the standard baseline used in adaptive noise cancellation (Widrow &
Hoff, 1960).  It handles *linear* coupling well but fails on non-linear or
rapidly time-varying couplings — precisely where the RL agent should excel.

Normalised LMS (NLMS, normalized=True)
---------------------------------------
For coloured inputs such as seismic 1/f² noise the eigenvalue spread of the
input autocorrelation matrix is large, which can cause plain LMS to diverge
with any fixed step size.  NLMS divides the update by the instantaneous
input power:

    w  ←  w  +  (μ / (‖x‖² + ε)) · e · x        μ ∈ (0, 2)

This normalises the effective step size to be independent of the input
spectrum, giving stable convergence for any stationary input.  It is the
standard algorithm used in seismic noise subtraction at LIGO.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class LMSFilter:
    """
    Online LMS / NLMS adaptive filter for noise cancellation.

    Parameters
    ----------
    filter_length : number of witness taps
    step_size     : learning rate μ.
                    Plain LMS: typical range 1e-4 … 1e-2.
                    NLMS (normalized=True): dimensionless, range (0, 2); 0.5 is safe.
    normalized    : if True use Normalised LMS (NLMS) — recommended for
                    coloured inputs (seismic signals, 1/f² noise).
    """

    def __init__(
        self,
        filter_length: int = 64,
        step_size: float = 1e-3,
        normalized: bool = False,
    ):
        self.M = filter_length
        self.mu = step_size
        self.normalized = normalized
        self._eps = 1e-6
        self.weights = np.zeros(filter_length)
        self._witness_buf = np.zeros(filter_length)

    def reset(self):
        self.weights[:] = 0.0
        self._witness_buf[:] = 0.0

    def update(
        self, witness_x_sample: float, main_sample: float, witness_y_sample: float = 0.0
    ) -> float:
        """
        Process one sample and return the cleaned main-channel value.

        Parameters
        ----------
        witness_x_sample : current witness X (inline) channel sample
        main_sample      : current main (obtaining) channel sample
        witness_y_sample : current witness Y (perpendicular) channel sample (optional, default 0)

        Returns
        -------
        main_clean : main_sample minus the linear coupling estimate
        """
        self._witness_buf = np.roll(self._witness_buf, 1)
        self._witness_buf[0] = witness_x_sample

        coupling_estimate = float(self.weights @ self._witness_buf)
        residual = main_sample - coupling_estimate

        if self.normalized:
            power = float(self._witness_buf @ self._witness_buf) + self._eps
            self.weights += (self.mu / power) * residual * self._witness_buf
        else:
            self.weights += self.mu * residual * self._witness_buf

        return residual

    def run(
        self,
        witness_x: np.ndarray,
        main: np.ndarray,
        witness_y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Process full arrays of data (online, sample by sample).

        Parameters
        ----------
        witness_x : (N,) witness X channel (inline, translational)
        main      : (N,) main (obtaining) channel
        witness_y : (N,) witness Y channel (perpendicular, tilt source).
                    When provided the filter length is doubled to 2·M and
                    both channels are concatenated as separate linear inputs.
                    Note: this linear filter cannot cancel the tilt-horizontal
                    coupling when T(t) drifts, but can approximate it when
                    T is nearly constant.

        Returns
        -------
        cleaned : (N,) main channel after LMS subtraction
        """
        if witness_y is not None:
            M2 = 2 * self.M
            weights2 = np.zeros(M2)
            buf2 = np.zeros(M2)
            self.reset()
            N = len(main)
            cleaned = np.empty(N)
            eps = self._eps
            for i in range(N):
                buf2 = np.roll(buf2, 1)
                buf2[0] = float(witness_x[i])
                buf2[self.M] = float(witness_y[i])
                coupling_estimate = float(weights2 @ buf2)
                residual = float(main[i]) - coupling_estimate
                if self.normalized:
                    power = float(buf2 @ buf2) + eps
                    weights2 += (self.mu / power) * residual * buf2
                else:
                    weights2 += self.mu * residual * buf2
                cleaned[i] = residual
            return cleaned

        self.reset()
        N = len(main)
        cleaned = np.empty(N)
        for i in range(N):
            cleaned[i] = self.update(float(witness_x[i]), float(main[i]))
        return cleaned
