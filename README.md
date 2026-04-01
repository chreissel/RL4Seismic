# RL4Seismic — RL-Based Seismic Noise Cancellation

A reinforcement learning environment for active seismic noise cancellation, formulated as a **closed-loop control problem**. Two RL policies are implemented and benchmarked against classical adaptive filters (NLMS, IIR) and a supervised LSTM baseline:

- **RecurrentPPO (LSTM)** — maintains hidden state across timesteps so the agent can adapt online within an episode, like an adaptive filter.
- **Deep Loop Shaping (DLS)** — PPO with a WaveNet-style dilated causal convolution feature extractor, inspired by [arXiv:2509.14016](https://arxiv.org/abs/2509.14016) (DeepMind / Caltech, 2025). No recurrent state; temporal context comes from a large receptive field (511 samples = 127 s @ 4 Hz).

Inspired by [arXiv:2511.19682](https://arxiv.org/abs/2511.19682) (Reissel et al., 2025), which applies supervised LSTMs to real LIGO data. This repository uses a synthetic simulator that matches the paper's signal processing parameters (4 Hz, 60 s context window, microseismic band) but models non-stationarity with an Ornstein–Uhlenbeck process — our own approximation, not from the paper.

---

## Problem Formulation

### Closed-loop block diagram

```
w1_t ──→ [ Plant P(z) ] ──→ y_t = h(t)⊛w1_t + [T2L] + n_t
w2_t ─┘                            │
  │                         ─── (+) ──→ e_t = y_t − a_t  (error / residual)
  │                        │
  └──→ [ Controller C ] ──→ a_t
        (RL policy π)
```

| Symbol | Meaning |
|--------|---------|
| `w1_t` | Horizontal seismometer (witness signal — broadband coloured noise) |
| `w2_t` | Vertical seismometer (optional, multi-source mode) |
| `h(t) ⊛ w1` | Linear FIR coupling with OU-drifting resonance parameters |
| `T(t)·θ(t)·w1` | Bilinear tilt-to-length (T2L) coupling (optional) |
| `n_t` | Sensor noise (Gaussian, unpredictable) |
| `y_t` | Main channel (observed: coupling + noise) |
| `a_t` | Agent's action — estimated coupling to subtract |
| `e_t = y_t − a_t` | Residual after subtraction (the "error signal") |

**Objective:** minimise `E[e_t²]` — drive the residual to the sensor noise floor.

### Seismic coupling model

Unlike polynomial toy models, the coupling here is a **linear time-varying FIR filter** — the physically correct description of seismic coupling in LIGO-like detectors:

```
y(t) = h(t) ⊛ w(t)  +  n_sensor(t)
```

- `h(t)` is a resonant FIR filter (damped mass-spring-damper, f_r ≈ 0.2 Hz, Q ≈ 5)
- Parameters drift via **Ornstein–Uhlenbeck** processes — a synthetic approximation of slow thermal/alignment drift (timescale ~10 min). Real LIGO coupling is non-stationary across many timescales: minutes–hours (thermal), months (seasonal ocean storms), and sudden discontinuities (lock-loss, maintenance). The OU model captures only the slow mean-reverting component.
- Ground motion `w(t)` is **broadband coloured noise** (1/f² spectrum, 0.05–1.5 Hz)
- Sampling rate: **4 Hz**, context window: **60 s** — matching arXiv:2511.19682

### Tilt-to-length (T2L) bilinear coupling

With `--tilt-coupling` (requires `--multi-source`), a physically motivated nonlinear term is added:

```
C_T2L(t) = T(t) · θ_proxy(t) · w1(t)
```

where `θ_proxy[t] = w2[t] − w2[t−1]` approximates ground tilt from Rayleigh waves and `T(t)` is an OU-drifting alignment gain. This **bilinear product of two channels cannot be cancelled by any linear filter** (LMS/NLMS), giving the RL agent a genuine advantage.

### Why closed-loop?

The agent observes:
```
obs_t = [ witness1[t-W+1..t], [witness2[t-W+1..t],]  residual[t-W..t-1] ]
```

Feeding back past residuals closes the loop: the agent can observe whether its previous actions over- or under-corrected and adapt accordingly.

---

## Repository Structure

```
RL4Seismic/
├── noise_removal/
│   ├── environment.py     # Gymnasium environment (closed-loop formulation)
│   ├── signals.py         # SeismicConfig + SeismicSignalSimulator
│   └── policy.py          # Dilated causal convolution feature extractor
├── baselines/
│   ├── lms_filter.py      # NLMS adaptive filter baseline
│   ├── iir_filter.py      # IIR closed-loop adaptive filter baseline
│   └── lstm_supervised.py # Supervised LSTM baseline (arXiv:2511.19682)
├── train.py               # RecurrentPPO training script
├── train_resume.py        # Resume training from checkpoint
├── evaluate.py            # Evaluation and comparison plots
└── requirements.txt
```

### Key files

**`noise_removal/signals.py`** — `SeismicConfig` + `SeismicSignalSimulator`.
Generates physically motivated seismic episodes: resonant FIR coupling, OU-drifting parameters, broadband ground motion, optional multi-source and T2L.

**`noise_removal/environment.py`** — Gymnasium-compatible environment.
- Observation: `Box((n_witnesses+1)·W,)` — witness window(s) + residual window
- Action: `Box(1,)` in `[-15, +15]` — scalar coupling estimate
- Reward: `y_t² − e_t²` — improvement in instantaneous squared amplitude

**`noise_removal/policy.py`** — `DilatedCausalConvExtractor` (Deep Loop Shaping policy).
WaveNet-style dilated causal convolution feature extractor used with standard PPO. 8 layers, kernel=3, dilation doubles per layer → receptive field = 511 samples (127 s @ 4 Hz), covering the full 240-sample seismic window with margin. Inspired by [arXiv:2509.14016](https://arxiv.org/abs/2509.14016).

**`baselines/lms_filter.py`** — Normalised LMS (NLMS) adaptive filter. Required for coloured seismic inputs (large eigenvalue spread makes plain LMS diverge).

**`baselines/iir_filter.py`** — Equation-error IIR filter using both witness (feedforward) and past residuals (feedback), mirroring the RL observation exactly.

**`baselines/lstm_supervised.py`** — Offline-trained 3-layer LSTM (hidden 128), MSE loss. Mirrors the supervised approach of arXiv:2511.19682.

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `gymnasium`, `stable-baselines3`, `sb3-contrib` (RecurrentPPO), `numpy`, `matplotlib`, `scipy`, `torch` (for supervised LSTM).

---

## Usage

### Train

```bash
# Default: single-source, OU-drifting FIR coupling, RecurrentPPO
python train.py

# Multi-source (two seismometers)
python train.py --multi-source

# Multi-source + tilt-to-length bilinear coupling
python train.py --multi-source --tilt-coupling

# Regime changes (sudden coupling path switches)
python train.py --regime-changes

# Longer run
python train.py --timesteps 5_000_000

# Deep Loop Shaping policy (dilated causal conv) instead of LSTM
python train.py --dilated-conv
python train.py --dilated-conv --multi-source --tilt-coupling
python train.py --dilated-conv --conv-layers 8 --conv-channels 64   # explicit defaults
```

Saves model to `models/ppo_noise_cancellation.zip` and VecNormalize stats to `models/ppo_noise_cancellation_vecnorm.pkl`.

### Resume training

```bash
python train_resume.py --checkpoint models/ppo_noise_cancellation_100000_steps --extra-steps 500_000
```

### Evaluate

```bash
# Compare all methods (NLMS, IIR, supervised LSTM, RL)
python evaluate.py

# Fast: baselines only (no model, no LSTM training)
python evaluate.py --no-model --no-lstm

# With multi-source + T2L (must match training config)
python evaluate.py --multi-source --tilt-coupling

# Longer episode
python evaluate.py --duration 600
```

Saves plots to `results/noise_cancellation_overview.png`.

---

## RL Policies

Two policy architectures are available, both trained with PPO-family algorithms:

### RecurrentPPO + LSTM (default)

```bash
python train.py                         # default
python train.py --multi-source --tilt-coupling
```

Uses `sb3-contrib` `RecurrentPPO` with `MlpLstmPolicy` (hidden size 256). The LSTM carries state across timesteps within an episode, enabling the agent to track slow parameter drift and re-adapt after regime changes — analogous to an online adaptive filter but with a learned nonlinear strategy.

### Deep Loop Shaping — Dilated Causal Conv (DLS)

```bash
python train.py --dilated-conv
python train.py --dilated-conv --multi-source --tilt-coupling
```

Uses standard `PPO` with a `DilatedCausalConvExtractor` feature extractor, inspired by [arXiv:2509.14016](https://arxiv.org/abs/2509.14016) (DeepMind / Caltech). The extractor replaces the LSTM with a WaveNet-style stack of dilated causal convolutions:

```
Input: (batch, n_channels, 240 samples)
  → input_proj   Conv1d n_channels → 64
  → 8 × CausalConv1d (dilation = 1, 2, 4, …, 128)   RF = 511 samples = 127 s
  → global average pool over time
  → linear → 256-dim features
  → PPO actor/critic heads
```

| Property | RecurrentPPO + LSTM | DLS (dilated conv) |
|---|---|---|
| Algorithm | RecurrentPPO | PPO |
| Temporal memory | LSTM hidden state | Dilated conv receptive field |
| Receptive field | Unbounded (recurrent) | 511 samples (127 s) |
| Training | Sequential (recurrent rollouts) | Parallel (all timesteps at once) |
| Vanishing gradients | Yes (long sequences) | No |
| Inference | Step-by-step with state | Parallel over window |

The DLS policy requires `--conv-layers ≥ 7` to cover the 240-sample (60 s) seismic observation window (7 layers → RF=255, 8 layers → RF=511).

---

## Problem Variants

| Flag | Coupling model | Linear filter floor |
|------|---------------|---------------------|
| *(default)* | `h(t) ⊛ w1` — OU-drifting resonant FIR | ~1–2× oracle |
| `--multi-source` | `h1(t)⊛w1 + h2(t)⊛w2` — two independent FIR | ~1–2× oracle |
| `--multi-source --tilt-coupling` | above + `T(t)·θ(t)·w1` bilinear | `sqrt(rms(T2L)²+oracle²)` |
| `--regime-changes` | `h_k ⊛ w` — Poisson coupling path switches | ~2–5× oracle at switch |

The T2L variant provides the most compelling RL advantage: any linear filter is provably bounded away from oracle, but an RL agent that learns the bilinear product can approach it.

---

## Results

Performance measured as RMS of the output signal normalised to the oracle (sensor noise floor).

| Method | vs Oracle | Notes |
|--------|-----------|-------|
| Raw main channel | ~40× | before subtraction |
| NLMS filter | ~30–60× | linear FIR; T2L and slow 0.2 Hz resonance limit convergence |
| IIR adaptive filter | ~30–60× | adds residual feedback; same linear floor |
| Linear filter floor (T2L) | ~3–5× | hard floor from bilinear T2L term; no linear method can beat this |
| Supervised LSTM (arXiv:2511.19682) | ~3–10× | offline-trained; limited by training episode mismatch |
| RL — RecurrentPPO (LSTM) | TBD | online adaptive |
| RL — Deep Loop Shaping (dilated conv) | TBD | online, no recurrent state |
| **Oracle** | **1×** | sensor noise floor only |

With `--tilt-coupling`, linear methods are bounded by `sqrt(rms(T2L)² + oracle²)` while the RL agent can in principle reach oracle.

---

## Why RL?

Classical adaptive filters (NLMS, IIR) are well-suited to linear, slowly-varying couplings. The RL agent becomes advantageous when:

1. **Tilt-to-length nonlinearity** — the bilinear T2L term `T(t)·θ(t)·w1(t)` cannot be represented by any linear filter; the LSTM can learn the product implicitly.
2. **Sudden regime changes** — coupling jumps discontinuously; the LSTM detects the transition from the residual spike and re-adapts faster than gradient descent.
3. **Long-context adaptation** — the 60 s FIR filter length and slow OU drift reward agents that maintain accurate internal models over minutes.
