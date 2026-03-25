"""
IIR adaptive filter — closed-loop baseline.

Mirrors the closed-loop RL observation exactly:

    â_t  =  b^T · [w_t, …, w_{t-M+1}]   (feedforward: witness taps)
          +  a^T · [e_{t-1}, …, e_{t-N}] (feedback: past residual taps)

    e_t  =  y_t − â_t                    (error / residual)

Weights are updated jointly via the equation-error LMS rule (Widrow, 1975),
which treats past residuals as fixed inputs during the gradient step:

    b  ←  b + μ · e_t · witness_buf
    a  ←  a + μ · e_t · residual_buf

This approximation gives a stable, practical algorithm at the cost of a
small bias relative to the true gradient.  It is the direct IIR counterpart
of the FIR LMS filter and the natural classical benchmark for the closed-loop
RL formulation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class IIRFilter:
    """
    Online IIR adaptive filter for noise cancellation.

    Parameters
    ----------
    feedforward_length : M — witness taps (reference input)
    feedback_length    : N — past-residual taps (error feedback)
    step_size          : LMS learning rate μ
    """

    def __init__(
        self,
        feedforward_length: int = 64,
        feedback_length: int = 64,
        step_size: float = 5e-4,
    ):
        self.M = feedforward_length
        self.N = feedback_length
        self.mu = step_size

        self.b = np.zeros(feedforward_length)   # feedforward weights (witness)
        self.a = np.zeros(feedback_length)       # feedback weights (residuals)
        self._witness_buf  = np.zeros(feedforward_length)
        self._residual_buf = np.zeros(feedback_length)

    def reset(self):
        self.b[:] = 0.0
        self.a[:] = 0.0
        self._witness_buf[:]  = 0.0
        self._residual_buf[:] = 0.0

    def update(
        self, witness_sample: float, main_sample: float, w2_sample: float = 0.0
    ) -> float:
        """
        Process one sample and return the cleaned main-channel value.

        Parameters
        ----------
        witness_sample : current witness channel 1 sample
        main_sample    : current main channel sample
        w2_sample      : current witness channel 2 sample (used only when
                         the filter was initialised with two-channel buffers)

        Returns
        -------
        residual : main_sample minus the IIR coupling estimate
        """
        self._witness_buf  = np.roll(self._witness_buf,  1)
        self._residual_buf = np.roll(self._residual_buf, 1)
        self._witness_buf[0] = witness_sample

        coupling_estimate = (
            float(self.b @ self._witness_buf)
            + float(self.a @ self._residual_buf)
        )
        residual = main_sample - coupling_estimate

        self._residual_buf[0] = residual

        self.b += self.mu * residual * self._witness_buf
        self.a += self.mu * residual * self._residual_buf

        return residual

    def run(
        self, witness: np.ndarray, main: np.ndarray, witness2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Process full arrays of data (online, sample by sample).

        Parameters
        ----------
        witness  : (N,) witness channel 1
        main     : (N,) main channel
        witness2 : (N,) witness channel 2 (optional).  When provided the
                   feedforward length is doubled to 2·M so both channels
                   are used as separate inputs (no cross-term product).

        Returns
        -------
        cleaned : (N,) main channel after IIR subtraction
        """
        if witness2 is not None:
            # Re-initialise with doubled feedforward buffer for two-channel input
            M2 = 2 * self.M
            b2 = np.zeros(M2)
            a2 = np.zeros(self.N)
            w_buf = np.zeros(M2)
            r_buf = np.zeros(self.N)
            N = len(main)
            cleaned = np.empty(N)
            for i in range(N):
                w_buf = np.roll(w_buf, 1)
                r_buf = np.roll(r_buf, 1)
                w_buf[0]      = float(witness[i])
                w_buf[self.M] = float(witness2[i])
                coupling_estimate = float(b2 @ w_buf) + float(a2 @ r_buf)
                residual = float(main[i]) - coupling_estimate
                r_buf[0] = residual
                b2 += self.mu * residual * w_buf
                a2 += self.mu * residual * r_buf
                cleaned[i] = residual
            return cleaned

        self.reset()
        N = len(main)
        cleaned = np.empty(N)
        for i in range(N):
            cleaned[i] = self.update(float(witness[i]), float(main[i]))
        return cleaned
