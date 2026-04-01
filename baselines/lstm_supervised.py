"""
Supervised LSTM baseline — mirrors the approach in arXiv:2511.19682.

The paper trains a causal 3-layer LSTM (hidden size 128, MSE loss) on 4 Hz
seismic sensor data to predict residual platform motion, then subtracts the
prediction as a feed-forward correction.  This is fundamentally different from
the LMS/IIR adaptive filters (which are online, gradient-based, linear) and
from the RL agent (which is closed-loop and reward-driven).

Architecture
------------
  Input  : rolling window of W witness samples → (batch, W, n_inputs)
  Layers : n_layers LSTM layers, hidden_size units each
  Output : linear projection → scalar prediction of main channel at step t

Training
--------
  Offline on a labelled training episode (separate from the evaluation episode).
  Mini-batch SGD with Adam on MSE loss, skipping the first W-step warmup.
  The LSTM sees short chunks of the sequence so gradients stay manageable.

Inference
---------
  Run the LSTM on the full witness sequence in one forward pass (causal: output
  at step t depends only on witness[0..t]).  Subtract prediction from main.

Usage
-----
>>> from baselines import SupervisedLSTM
>>> lstm = SupervisedLSTM(window_size=240)
>>> lstm.fit(train_data, verbose=True)      # offline training episode
>>> cleaned = lstm.run(eval_data['witness_x'], eval_data['main'], eval_data['witness_y'])
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, n_layers: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, n_layers,
            batch_first=True, dropout=0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: "torch.Tensor", h=None):
        # x: (batch, seq_len, input_size)
        out, h_out = self.lstm(x, h)
        # take prediction at last timestep
        return self.fc(out[:, -1, :]).squeeze(-1), h_out  # (batch,)

    def forward_sequence(self, x: "torch.Tensor"):
        """Run on full sequence, return prediction at every step."""
        out, _ = self.lstm(x)           # (1, N, hidden)
        return self.fc(out[0]).squeeze(-1)  # (N,)


class SupervisedLSTM:
    """
    Supervised causal LSTM for noise cancellation.

    Parameters
    ----------
    window_size  : W — input context length in samples (default 240 = 60 s @ 4 Hz,
                   matching the paper's 60-second context window)
    hidden_size  : LSTM hidden units (default 128, matching paper)
    n_layers     : LSTM depth (default 3, matching paper)
    n_epochs     : training epochs
    lr           : Adam learning rate
    batch_size   : mini-batch size for training
    chunk_size   : sequence chunk length for training (truncated BPTT)
    """

    def __init__(
        self,
        window_size: int = 240,
        hidden_size: int = 128,
        n_layers: int = 3,
        n_epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 64,
        chunk_size: int = 512,
    ):
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for SupervisedLSTM.  "
                "Install with: pip install torch"
            )
        self.window_size = window_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self._hidden_size = hidden_size
        self._n_layers = n_layers
        self._net: Optional[_LSTMNet] = None
        self._n_inputs: int = 1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, data: dict, verbose: bool = False) -> None:
        """
        Train the LSTM on a labelled episode.

        Input at step t : witness window  witness_x[t-W+1 .. t]  (and witness_y)
        Target at step t: main[t]  (= coupling + sensor noise; residual after subtraction)

        Parameters
        ----------
        data    : episode dict from SeismicSignalSimulator
        verbose : print loss every 10 epochs
        """
        W = self.window_size
        witness_x = np.asarray(data["witness_x"], dtype=np.float32)
        main      = np.asarray(data["main"],      dtype=np.float32)
        N = len(main)

        n_inputs = 2 if "witness_y" in data else 1
        self._n_inputs = n_inputs
        self._net = _LSTMNet(n_inputs, self._hidden_size, self._n_layers)

        # Build dataset: (N - W + 1) windows
        xs, ys = self._build_windows(witness_x, main,
                                     data.get("witness_y"), W, N)
        X = torch.from_numpy(np.array(xs, dtype=np.float32))  # (n, W, c)
        y = torch.from_numpy(np.array(ys, dtype=np.float32))  # (n,)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        optim    = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn  = nn.MSELoss()

        self._net.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            n_total    = 0
            for xb, yb in loader:
                optim.zero_grad()
                pred, _ = self._net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optim.step()
                total_loss += loss.item() * len(xb)
                n_total    += len(xb)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1:3d}/{self.n_epochs}  "
                      f"loss = {total_loss / n_total:.6f}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(
        self,
        witness_x: np.ndarray,
        main:      np.ndarray,
        witness_y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Causal inference: subtract the LSTM prediction from main.

        For timesteps t < window_size the prediction is zero (warmup).

        Parameters
        ----------
        witness_x : (N,) witness X channel (inline, translational)
        main      : (N,) main (obtaining) channel
        witness_y : (N,) witness Y channel (perpendicular, tilt source; optional)

        Returns
        -------
        cleaned : (N,) residual after LSTM subtraction
        """
        if self._net is None:
            raise RuntimeError("Call fit() before run()")

        W = self.window_size
        witness_x = np.asarray(witness_x, dtype=np.float32)
        main      = np.asarray(main,      dtype=np.float32)
        N = len(main)

        xs, _ = self._build_windows(witness_x, main, witness_y, W, N,
                                    build_targets=False)
        if len(xs) == 0:
            return main.copy().astype(float)

        X = torch.from_numpy(np.array(xs, dtype=np.float32))  # (n, W, c)

        self._net.eval()
        with torch.no_grad():
            preds, _ = self._net(X)   # (n,)
        preds_np = preds.numpy()

        cleaned = main.copy().astype(float)
        cleaned[W - 1:] = main[W - 1:] - preds_np
        return cleaned

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_windows(
        witness_x: np.ndarray,
        main:      np.ndarray,
        witness_y: Optional[np.ndarray],
        W: int,
        N: int,
        build_targets: bool = True,
    ):
        n_samples = max(0, N - W + 1)
        n_inputs  = 2 if witness_y is not None else 1

        xs = np.empty((n_samples, W, n_inputs), dtype=np.float32)
        ys = np.empty(n_samples,                dtype=np.float32)

        for i, t in enumerate(range(W - 1, N)):
            xs[i, :, 0] = witness_x[t - W + 1 : t + 1]
            if witness_y is not None:
                xs[i, :, 1] = witness_y[t - W + 1 : t + 1]
            if build_targets:
                ys[i] = main[t]

        return xs, ys
