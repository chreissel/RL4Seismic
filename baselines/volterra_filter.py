"""
Volterra filter — classical nonlinear baseline.

A second-order Volterra filter extends the linear FIR filter with an
explicit bilinear cross-channel term:

    y_hat[t] =  Σ_k  b1[k]  · w_x[t - k]
              + Σ_k  b2[k]  · w_y[t - k]                           (linear part)
              + Σ_{j,k}  c[j, k] · w_x[t - j] · w_y[t - k]          (bilinear part)

This is the simplest model class that can exactly represent the T2L
coupling  T · θ_y(t) · w_x(t)  when the coefficients c[j, k] are chosen
correctly — the double-leaky-integrator structure of θ_y is absorbed into
the j-index lag on w_y.  A time-invariant Volterra kernel cannot track
OU-drifting T(t), so on non-stationary episodes it plateaus above the true
oracle floor but still strictly below NLMS (which has no bilinear term).

We fit the coefficients by ridge regression on a training segment, matching
the offline Wiener-filter protocol.  To keep the parameter count tractable,
the bilinear kernel is sampled on a coarse grid of taps (every ``bilinear_stride``
samples in each dimension) — full M² cross terms would be ~240² ≈ 58k params,
which over-fits noisy training data.  A coarse grid trades expressiveness for
generalisation and is the standard Volterra-kernel simplification.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class VolterraFilter:
    """
    Second-order Volterra filter for noise cancellation.

    Parameters
    ----------
    filter_length    : linear FIR length per channel (taps).
    bilinear_length  : number of (j, k) tap indices per dimension for the
                       bilinear cross-channel kernel.  Full grid is
                       bilinear_length × bilinear_length cross products.
    bilinear_stride  : stride over original taps for the bilinear grid.
                       (j = 0, stride, 2*stride, …, (bilinear_length-1)*stride)
    regularization   : ridge parameter (Tikhonov) for the normal equations.
    """

    def __init__(
        self,
        filter_length: int = 240,
        bilinear_length: int = 24,
        bilinear_stride: int = 10,
        regularization: float = 1e-3,
    ):
        self.M = filter_length
        self.K = bilinear_length
        self.stride = bilinear_stride
        self.reg = regularization
        # sparse tap indices for the bilinear grid (within [0, M))
        self.bilinear_taps = [i * bilinear_stride for i in range(bilinear_length)]
        assert self.bilinear_taps[-1] < filter_length, (
            f"bilinear grid extends past filter_length: "
            f"{self.bilinear_taps[-1]} >= {filter_length}"
        )
        self.weights: Optional[np.ndarray] = None
        self._n_linear: int = 0
        self._n_bilinear: int = 0

    # ------------------------------------------------------------------
    # Design-matrix construction
    # ------------------------------------------------------------------

    def _build_design(
        self,
        witness_x: np.ndarray,
        witness_y: np.ndarray,
    ) -> np.ndarray:
        """
        Build the Volterra design matrix.

        Columns:
            [ linear_x (M) | linear_y (M) | bilinear_xy (K*K) ]

        The bilinear block contains products w_x[t-j]·w_y[t-k] for
        (j, k) drawn from the coarse tap grid.
        """
        M = self.M
        K = self.K
        taps = self.bilinear_taps
        N = len(witness_x)
        n_rows = N - M

        n_linear = 2 * M
        n_bilinear = K * K
        self._n_linear = n_linear
        self._n_bilinear = n_bilinear

        X = np.zeros((n_rows, n_linear + n_bilinear), dtype=np.float64)

        # Linear x block
        for k in range(M):
            X[:, k] = witness_x[M - k: N - k]
        # Linear y block
        for k in range(M):
            X[:, M + k] = witness_y[M - k: N - k]

        # Bilinear xy block (coarse grid)
        col = n_linear
        for j in taps:
            xj = witness_x[M - j: N - j]
            for k in taps:
                yk = witness_y[M - k: N - k]
                X[:, col] = xj * yk
                col += 1

        return X

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        witness_x: np.ndarray,
        main: np.ndarray,
        witness_y: np.ndarray,
    ) -> "VolterraFilter":
        """
        Estimate Volterra coefficients from training data via ridge regression.
        """
        M = self.M
        X = self._build_design(witness_x, witness_y)
        d = main[M:].astype(np.float64)

        XtX = X.T @ X
        XtX += self.reg * np.eye(X.shape[1])
        Xtd = X.T @ d
        self.weights = np.linalg.solve(XtX, Xtd)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(
        self,
        witness_x: np.ndarray,
        main: np.ndarray,
        witness_y: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the fitted Volterra filter to evaluation data.

        Returns the cleaned main channel.  The first ``filter_length`` samples
        are left as raw main (no filter history yet), matching the Wiener
        filter's convention.
        """
        if self.weights is None:
            raise RuntimeError("Call .fit() before .run()")

        M = self.M
        X = self._build_design(witness_x, witness_y)
        y_hat = X @ self.weights  # shape (N - M,)

        cleaned = main.copy().astype(np.float64)
        cleaned[M:] = main[M:] - y_hat
        return cleaned
