"""
Train a RecurrentPPO agent on the seismic NoiseCancellationEnv.

Uses sb3-contrib RecurrentPPO with MlpLstmPolicy so the agent maintains
LSTM hidden state across timesteps within an episode.  This allows it to
behave like an adaptive filter — observing residuals, updating its implicit
internal model, and correcting — rather than applying a fixed open-loop
mapping as a plain MLP would.

The physically motivated seismic model (4 Hz, resonant FIR coupling,
OU-drifting parameters, two witness channels X/Y) is always used.  Options:
  --no-tilt-coupling : train on X coupling only (no tilt-horizontal term)
  --no-drift         : disable OU drift (stationary system; easier for baselines)
  --regime-changes   : sudden coupling path switches (Poisson process)

Usage
-----
    python train.py                          # default: X coupling + tilt-horizontal
    python train.py --timesteps 5_000_000   # longer run
    python train.py --no-tilt-coupling       # X coupling only

The trained model is saved to  models/ppo_noise_cancellation.zip
and VecNormalize stats to      models/ppo_noise_cancellation_vecnorm.pkl

Quick sanity check after training:
    python evaluate.py
"""

import argparse
import os

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from noise_removal import NoiseCancellationEnv, SeismicConfig, LoopShapingWrapper
from noise_removal.policy import DilatedCausalConvExtractor


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO seismic noise-cancellation agent")
    p.add_argument("--timesteps", type=int, default=2_000_000,
                   help="Total training timesteps (default: 2 000 000)")
    p.add_argument("--n-envs", type=int, default=4,
                   help="Number of parallel training environments (default: 4)")
    p.add_argument("--window-size", type=int, default=None,
                   help="Observation window in samples "
                        "(default: 240 = 60 s @ 4 Hz)")
    p.add_argument("--episode-duration", type=float, default=300.0,
                   help="Episode length in seconds (default: 300)")
    p.add_argument("--save-path", type=str, default="models/ppo_noise_cancellation",
                   help="Path to save the trained model (without .zip)")
    p.add_argument("--log-dir", type=str, default="logs/ppo_noise_cancellation",
                   help="Tensorboard log directory")
    p.add_argument("--no-drift", action="store_true",
                   help="Disable OU drift of coupling parameters (stationary system; "
                        "useful for debugging linear baselines)")
    p.add_argument("--regime-changes", action="store_true",
                   help="Enable sudden coupling regime switches (Poisson process); "
                        "adaptive filters must re-converge after each jump")
    p.add_argument("--tilt-coupling", action="store_true", default=True,
                   help="Include tilt-horizontal coupling from Y witness channel "
                        "(default: on). C_tilt(t) = T(t)·θ_y_proxy(t)")
    p.add_argument("--no-tilt-coupling", action="store_true",
                   help="Disable tilt-horizontal coupling (train on X coupling only)")
    p.add_argument("--freq-reward", action="store_true",
                   help="Use frequency-domain reward: improve power in [freq-low, freq-high] Hz")
    p.add_argument("--freq-low", type=float, default=0.05,
                   help="Lower edge of reward band in Hz (default: 0.05)")
    p.add_argument("--freq-high", type=float, default=1.5,
                   help="Upper edge of reward band in Hz (default: 1.5)")
    p.add_argument("--sensor-noise-color",
                   choices=("white", "pink", "brown"),
                   default="white",
                   help="Spectral shape of the GS13X sensor noise (oracle noise "
                        "floor). 'white' = flat PSD (default), 'pink' = 1/f, "
                        "'brown' = 1/f². The RMS is always sensor_noise_sigma.")
    p.add_argument("--sensor-noise-exponent", type=float, default=None,
                   help="Override --sensor-noise-color with a custom spectral "
                        "exponent α (PSD ∝ 1/f^α). Any float is accepted.")
    p.add_argument("--sensor-noise-band", type=float, nargs=2,
                   metavar=("F_LOW", "F_HIGH"), default=None,
                   help="Optional (f_low, f_high) Hz window over which to "
                        "enforce the sensor-noise RMS. When given, "
                        "sensor_noise_sigma is interpreted as the in-band RMS "
                        "on this window (Parseval-exact) instead of the "
                        "broadband RMS. Useful for coloured noise where the "
                        "broadband RMS is dominated by near-DC power.")
    p.add_argument("--loop-shaping", action="store_true",
                   help="Use frequency-domain loop-shaping reward (DLS, "
                        "arXiv:2509.14016): reward based on the estimated "
                        "sensitivity function |S(f)|^2 = PSD_e/PSD_y instead "
                        "of instantaneous squared error.  Overrides --freq-reward.")
    p.add_argument("--psd-window", type=int, default=256,
                   help="PSD estimation window for --loop-shaping (samples, "
                        "default: 256 = 64 s @ 4 Hz, freq resolution 0.015 Hz)")
    p.add_argument("--amplification-penalty", type=float, default=2.0,
                   help="Weight on out-of-band amplification penalty in "
                        "--loop-shaping reward (default: 2.0)")
    p.add_argument("--dilated-conv", action="store_true",
                   help="Use dilated causal convolution policy (PPO) instead of "
                        "LSTM policy (RecurrentPPO). Inspired by DeepMind DLS.")
    p.add_argument("--conv-channels", type=int, default=64,
                   help="Number of channels in dilated conv extractor (default: 64)")
    p.add_argument("--conv-layers", type=int, default=8,
                   help="Number of dilated conv layers (default: 8, RF=511 samples = 127 s @ 4 Hz). "
                        "Must be ≥7 to cover the 240-sample seismic window.")
    return p.parse_args()


def make_env(config, window_size, episode_duration,
             freq_reward=False, freq_low=0.05, freq_high=1.5,
             loop_shaping=False, psd_window=256,
             amplification_penalty=2.0):
    def _init():
        env = NoiseCancellationEnv(
            config=config,
            window_size=window_size,
            episode_duration=episode_duration,
            freq_reward=freq_reward,
            freq_band_low=freq_low,
            freq_band_high=freq_high,
        )
        if loop_shaping:
            env = LoopShapingWrapper(
                env,
                psd_window=psd_window,
                f_low=freq_low,
                f_high=freq_high,
                amplification_penalty=amplification_penalty,
                fs=config.fs,
            )
        return env
    return _init


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    color_to_exp = {"white": 0.0, "pink": 1.0, "brown": 2.0}
    sensor_exp = (
        args.sensor_noise_exponent
        if args.sensor_noise_exponent is not None
        else color_to_exp[args.sensor_noise_color]
    )

    sensor_band = tuple(args.sensor_noise_band) if args.sensor_noise_band else None

    config = SeismicConfig(
        drift=not args.no_drift,
        regime_changes=args.regime_changes,
        tilt_coupling=not args.no_tilt_coupling,
        sensor_noise_exponent=sensor_exp,
        sensor_noise_band=sensor_band,
    )

    window_size = args.window_size if args.window_size is not None else config.filter_length

    algo_name = "PPO + DilatedCausalConv" if args.dilated_conv else "RecurrentPPO (LSTM)"
    print("=" * 60)
    print(f"  RL Seismic Noise Cancellation — {algo_name}")
    print("=" * 60)
    print(f"  Sampling rate  : {config.fs} Hz")
    exp_to_name = {0.0: "white", 1.0: "pink", 2.0: "brown"}
    color_name = exp_to_name.get(config.sensor_noise_exponent,
                                 f"α={config.sensor_noise_exponent}")
    if config.sensor_noise_band is not None:
        f_lo, f_hi = config.sensor_noise_band
        rms_scope = f"in-band RMS on [{f_lo}, {f_hi}] Hz"
    else:
        rms_scope = "broadband RMS"
    print(f"  Sensor noise σ : {config.sensor_noise_sigma}  "
          f"(GS13X obtaining channel, {color_name}, {rms_scope})")
    drift_str = "OU-drifting" if config.drift else "stationary"
    if config.tilt_coupling and config.regime_changes:
        print(f"  Coupling model : h_x(t)⊛w_x + C_tilt(t)  [{drift_str}, {config.n_regimes} regimes]")
    elif config.tilt_coupling:
        print(f"  Coupling model : h_x(t)⊛w_x + T(t)·θ_y(t)  [{drift_str}] (X + tilt-horizontal)")
    elif config.regime_changes:
        print(f"  Coupling model : h_k⊛w_x  ({config.n_regimes} FIR regimes, "
              f"mean hold {config.mean_hold_time:.0f} s)")
    else:
        print(f"  Coupling model : h_x(t)⊛w_x  [{drift_str} resonant FIR, X only]")
    print(f"  Window size    : {window_size} samples = {window_size/config.fs:.1f} s")
    print(f"  Episode length : {args.episode_duration} s")
    print(f"  Total steps    : {args.timesteps:,}")
    print(f"  Parallel envs  : {args.n_envs}")
    if args.loop_shaping:
        print(f"  Reward         : loop-shaping |S(f)|² on [{args.freq_low}, {args.freq_high}] Hz "
              f"(PSD window={args.psd_window}, α={args.amplification_penalty})")
    elif args.freq_reward:
        print(f"  Reward         : band-limited [{args.freq_low}, {args.freq_high}] Hz")
    else:
        print(f"  Reward         : broadband squared-error improvement")
    if args.dilated_conv:
        rf = 1 + 2 * (2 ** args.conv_layers - 1)
        print(f"  Policy         : dilated causal conv ({args.conv_layers} layers, "
              f"{args.conv_channels} ch, RF={rf} samples = {rf/config.fs:.0f} s)")
    else:
        print(f"  Policy         : LSTM (RecurrentPPO, hidden=256)")
    print("=" * 60)

    vec_env = make_vec_env(
        make_env(config, window_size, args.episode_duration,
                 freq_reward=args.freq_reward,
                 freq_low=args.freq_low,
                 freq_high=args.freq_high,
                 loop_shaping=args.loop_shaping,
                 psd_window=args.psd_window,
                 amplification_penalty=args.amplification_penalty),
        n_envs=args.n_envs,
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if args.dilated_conv:
        policy_kwargs = dict(
            features_extractor_class=DilatedCausalConvExtractor,
            features_extractor_kwargs=dict(
                window_size=window_size,
                conv_channels=args.conv_channels,
                n_layers=args.conv_layers,
            ),
            net_arch=[256, 256],
        )
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.999,   # 250 s horizon at 4 Hz; 0.99 (25 s) is too short vs OU timescale 600 s
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=1e-3,
            policy_kwargs=policy_kwargs,
            tensorboard_log=None,
            verbose=1,
        )
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.999,   # 250 s horizon at 4 Hz; 0.99 (25 s) is too short vs OU timescale 600 s
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=1e-4,
            policy_kwargs=dict(
                net_arch=[256, 256],
                lstm_hidden_size=256,
                n_lstm_layers=1,
            ),
            tensorboard_log=None,
            verbose=1,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=os.path.dirname(args.save_path),
        name_prefix=os.path.basename(args.save_path),
        save_vecnormalize=True,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb, progress_bar=False)

    model.save(args.save_path)
    vec_env.save(args.save_path + "_vecnorm.pkl")

    print(f"\nModel saved to  {args.save_path}.zip")
    print(f"VecNormalize    {args.save_path}_vecnorm.pkl")
    print("Run  python evaluate.py  to see results.")


if __name__ == "__main__":
    main()
