"""
Frequency-domain Wiener filter — batch-optimal linear baseline.

The Wiener filter is the standard classical method for seismic noise
subtraction at LIGO (Driggers et al. 2012, Davis et al. 2019).  It computes
the optimal linear FIR filter by solving the normal equations on a training
segment, then applies the fixed filter to evaluation data.

    w* = argmin_w  E[ (y[t] - w^T x[t])^2 ]
       = R_xx^{-1} r_xy

where R_xx is the input autocorrelation matrix and r_xy is the cross-
correlation between input and desired signal.

In practice we solve via least-squares on the training data:

    w* = (X^T X)^{-1} X^T d

Unlike online adaptive filters (NLMS, IIR-LMS), the Wiener filter:
  - Achieves the global MSE optimum for any stationary linear system
  - Has no convergence transient on evaluation data
  - Cannot adapt to non-stationarity (drift, regime changes)

This makes it the ideal classical benchmark: it shows exactly what linear
filtering can achieve, cleanly separating the linear-cancellable component
from the nonlinear residual (tilt-horizontal coupling with drifting T(t)).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class WienerFilter:
    """
    Batch-optimal Wiener filter for noise cancellation.

    Parameters
    ----------
    filter_length : M — number of causal FIR taps per witness channel
    regularization : Tikhonov regularization (ridge parameter) added to
                     the diagonal of X^T X to stabilise the solution when
                     the training segment is short or the input is ill-
                     conditioned.  Default 1e-6.
    """

    def __init__(
        self,
        filter_length: int = 240,
        regularization: float = 1e-6,
    ):
        self.M = filter_length
        self.reg = regularization
        self.weights: Optional[np.ndarray] = None
        self._n_channels: int = 1

    def fit(
        self,
        witness_x: np.ndarray,
        main: np.ndarray,
        witness_y: Optional[np.ndarray] = None,
    ) -> "WienerFilter":
        """
        Estimate the optimal FIR filter from training data.

        Parameters
        ----------
        witness_x : (N,) witness X channel (training segment)
        main      : (N,) main / obtaining channel (training segment)
        witness_y : (N,) optional witness Y channel

        Returns
        -------
        self (fitted)
        """
        M = self.M
        N = len(main)
        n_channels = 2 if witness_y is not None else 1
        self._n_channels = n_channels
        cols = n_channels * M

        # Build causal design matrix: x[t] = [wx[t], wx[t-1], ..., wx[t-M+1],
        #                                     wy[t], wy[t-1], ..., wy[t-M+1]]
        n_rows = N - M
        X = np.zeros((n_rows, cols), dtype=np.float64)
        for k in range(M):
            X[:, k] = witness_x[M - k: N - k]
        if witness_y is not None:
            for k in range(M):
                X[:, M + k] = witness_y[M - k: N - k]

        d = main[M:].astype(np.float64)

        # Solve normal equations with Tikhonov regularization
        XtX = X.T @ X
        XtX += self.reg * np.eye(cols)
        Xtd = X.T @ d
        self.weights = np.linalg.solve(XtX, Xtd)

        return self

    def run(
        self,
        witness_x: np.ndarray,
        main: np.ndarray,
        witness_y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply the fitted Wiener filter to evaluation data.

        Parameters
        ----------
        witness_x : (N,) witness X channel (evaluation segment)
        main      : (N,) main / obtaining channel (evaluation segment)
        witness_y : (N,) optional witness Y channel

        Returns
        -------
        cleaned : (N,) main channel after Wiener subtraction
        """
        if self.weights is None:
            raise RuntimeError("Call .fit() before .run()")

        M = self.M
        N = len(main)
        w = self.weights
        cleaned = main.copy().astype(np.float64)

        for i in range(M, N):
            buf = witness_x[i - M + 1: i + 1][::-1]
            if witness_y is not None:
                buf_y = witness_y[i - M + 1: i + 1][::-1]
                buf = np.concatenate([buf, buf_y])
            cleaned[i] = main[i] - float(w @ buf)

        return cleaned
