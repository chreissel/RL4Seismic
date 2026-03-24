"""
Least Mean Squares (LMS) adaptive filter — linear baseline.

The LMS filter maintains a weight vector w ∈ R^M and at each step predicts
the coupling as  â_t = w^T · x_t,  where x_t is a window of recent witness
samples.  Weights are updated online via  w ← w + μ · e_t · x_t,  where
e_t = y_t − â_t  is the residual.

This is the standard baseline used in adaptive noise cancellation (Widrow &
Hoff, 1960).  It handles *linear* coupling well but fails on non-linear or
rapidly time-varying couplings — precisely where the RL agent should excel.
"""

from __future__ import annotations

import numpy as np


class LMSFilter:
    """
    Online LMS adaptive filter for noise cancellation.

    Parameters
    ----------
    filter_length : number of witness taps
    step_size     : LMS learning rate μ  (typical range 1e-4 … 1e-2)
    """

    def __init__(self, filter_length: int = 64, step_size: float = 1e-3):
        self.M = filter_length
        self.mu = step_size
        self.weights = np.zeros(filter_length)
        self._witness_buf = np.zeros(filter_length)  # circular-ish buffer

    def reset(self):
        self.weights[:] = 0.0
        self._witness_buf[:] = 0.0

    def update(self, witness_sample: float, main_sample: float) -> float:
        """
        Process one sample and return the cleaned main-channel value.

        Internally updates the filter weights using the LMS rule.

        Returns
        -------
        main_clean : main_sample minus the linear coupling estimate
        """
        # Shift buffer and insert new witness sample
        self._witness_buf = np.roll(self._witness_buf, 1)
        self._witness_buf[0] = witness_sample

        # Predict and compute residual
        coupling_estimate = float(self.weights @ self._witness_buf)
        residual = main_sample - coupling_estimate

        # LMS weight update
        self.weights += self.mu * residual * self._witness_buf

        return residual

    def run(self, witness: np.ndarray, main: np.ndarray) -> np.ndarray:
        """
        Process full arrays of data (online, sample by sample).

        Parameters
        ----------
        witness : (N,) witness channel
        main    : (N,) main channel

        Returns
        -------
        cleaned : (N,) main channel after LMS subtraction
        """
        self.reset()
        N = len(main)
        cleaned = np.empty(N)
        for i in range(N):
            cleaned[i] = self.update(float(witness[i]), float(main[i]))
        return cleaned
