"""
Frequency-domain Wiener filter with top-hat bandpass — matching Reissel et al. (2025).

arXiv:2511.19682 uses a frequency-domain approach to seismic feedforward
cancellation at LIGO:

  1. Estimate the transfer function H(f) = S_yx(f) / S_xx(f) between
     witness sensor(s) and target channel using Welch's cross/auto-spectral
     densities over a training segment.
  2. Apply a top-hat (rectangular) bandpass window to H(f), zeroing out
     everything outside the microseismic band [f_low, f_high] (default
     0.1–0.3 Hz in the paper).
  3. Convert H(f) to a causal FIR filter h[n] via inverse FFT, retaining
     only the causal part (n >= 0) to ensure the filter can be applied
     in real time.
  4. Convolve h[n] with the witness signal and subtract from the target
     channel.

The top-hat restricts the filter to act only in the band where seismic
coupling dominates, preventing noise amplification at other frequencies.

This is the standard LIGO feedforward subtraction technique (Driggers et al.
2012, Davis et al. 2019, Reissel et al. 2025), and serves as the linear
baseline against which the LSTM / RL agents are compared.

Parameters matching arXiv:2511.19682:
  - Sampling rate: 4 Hz (downsampled)
  - Microseismic band: 0.1–0.3 Hz
  - Input: ground / platform sensor channels (GS13, STS-2, T240)
  - Output: predicted platform motion to subtract
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import welch, csd


class TopHatWienerFilter:
    """
    Frequency-domain Wiener filter with top-hat bandpass.

    Estimates H(f) = S_yx(f) / S_xx(f) via Welch's method, applies a
    rectangular bandpass in [f_low, f_high], and converts to a causal FIR
    via IFFT.

    Parameters
    ----------
    fs          : sampling rate in Hz (default: 4.0, matching paper)
    f_low       : lower edge of top-hat passband in Hz (default: 0.1)
    f_high      : upper edge of top-hat passband in Hz (default: 0.3)
    nperseg     : FFT segment length for Welch estimation (default: 256,
                  i.e. 64 s @ 4 Hz — gives ~0.016 Hz frequency resolution)
    noverlap    : overlap between Welch segments (default: nperseg // 2)
    filter_length : length of the output causal FIR filter in samples
                    (default: 240 = 60 s @ 4 Hz)
    regularization : small constant added to S_xx denominator to prevent
                     division by zero (default: 1e-10)
    """

    def __init__(
        self,
        fs: float = 4.0,
        f_low: float = 0.1,
        f_high: float = 0.3,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        filter_length: int = 240,
        regularization: float = 1e-10,
    ):
        self.fs = fs
        self.f_low = f_low
        self.f_high = f_high
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.filter_length = filter_length
        self.reg = regularization

        self._fir_x: Optional[np.ndarray] = None
        self._fir_y: Optional[np.ndarray] = None
        self._n_channels: int = 1

    def fit(
        self,
        witness_x: np.ndarray,
        main: np.ndarray,
        witness_y: Optional[np.ndarray] = None,
    ) -> "TopHatWienerFilter":
        """
        Estimate the optimal transfer function and convert to causal FIR.

        Parameters
        ----------
        witness_x : (N,) witness X channel (training segment)
        main      : (N,) target / obtaining channel (training segment)
        witness_y : (N,) optional witness Y channel

        Returns
        -------
        self (fitted)
        """
        self._fir_x = self._estimate_fir(witness_x, main)
        self._n_channels = 1

        if witness_y is not None:
            self._fir_y = self._estimate_fir(witness_y, main)
            self._n_channels = 2
        else:
            self._fir_y = None

        return self

    def _estimate_fir(
        self, witness: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """
        Estimate the causal FIR filter for one witness channel.

        Steps:
          1. Compute S_xx(f) and S_yx(f) via Welch's method
          2. Form H(f) = S_yx(f) / S_xx(f)
          3. Apply top-hat window: zero H(f) outside [f_low, f_high]
          4. IFFT → h(t), then truncate to causal part h[0..M-1]
        """
        # Cross-spectral and auto-spectral density via Welch
        freqs, S_xx = welch(
            witness, fs=self.fs, nperseg=self.nperseg,
            noverlap=self.noverlap, return_onesided=False,
        )
        _, S_yx = csd(
            target, witness, fs=self.fs, nperseg=self.nperseg,
            noverlap=self.noverlap, return_onesided=False,
        )

        # Transfer function estimate
        H = S_yx / (S_xx + self.reg)

        # Apply top-hat bandpass: zero out frequencies outside [f_low, f_high]
        freq_abs = np.abs(freqs)
        tophat_mask = (freq_abs >= self.f_low) & (freq_abs <= self.f_high)
        H_filtered = H * tophat_mask

        # Convert to time-domain impulse response via IFFT
        h_full = np.fft.ifft(H_filtered).real

        # Extract causal part: h[0], h[1], ..., h[M-1]
        # (the IFFT output has the causal part at indices [0..N/2-1]
        #  and the acausal part wrapped around at [N/2..N-1])
        M = min(self.filter_length, len(h_full) // 2)
        h_causal = h_full[:M].copy()

        # Apply a gentle taper (Tukey window) to reduce truncation ringing
        taper = np.ones(M)
        n_taper = min(M // 8, 16)
        if n_taper > 1:
            taper[-n_taper:] = 0.5 * (1 + np.cos(
                np.pi * np.arange(n_taper) / n_taper
            ))
        h_causal *= taper

        return h_causal

    def run(
        self,
        witness_x: np.ndarray,
        main: np.ndarray,
        witness_y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply the fitted top-hat Wiener filter to evaluation data.

        Parameters
        ----------
        witness_x : (N,) witness X channel (evaluation segment)
        main      : (N,) main / obtaining channel (evaluation segment)
        witness_y : (N,) optional witness Y channel

        Returns
        -------
        cleaned : (N,) main channel after feedforward subtraction
        """
        if self._fir_x is None:
            raise RuntimeError("Call .fit() before .run()")

        N = len(main)
        cleaned = main.copy().astype(np.float64)

        # Subtract X-channel contribution
        prediction = np.convolve(witness_x, self._fir_x, mode="full")[:N]
        cleaned -= prediction

        # Subtract Y-channel contribution if fitted
        if self._fir_y is not None and witness_y is not None:
            prediction_y = np.convolve(witness_y, self._fir_y, mode="full")[:N]
            cleaned -= prediction_y

        return cleaned
