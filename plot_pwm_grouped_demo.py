from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _load_pwm_lab():
    package_dir = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "pwm_lab",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pwm_lab package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pwm_lab"] = module
    spec.loader.exec_module(module)
    return module


def _demo_samples(n_samples: int) -> np.ndarray:
    n = np.arange(n_samples, dtype=np.float64)
    values = 0.52 + 0.33 * np.sin(2.0 * np.pi * n / n_samples)
    values += 0.11 * np.sin(2.0 * np.pi * 3.0 * n / n_samples)
    return np.clip(values, 0.0, 1.0)


def _save_mapping_figure(
    output_dir: Path,
    *,
    samples_per_period: int,
    channels: int,
    groups: int,
) -> Path:
    sample_index = np.zeros((channels, groups), dtype=int)
    sample_slot = np.zeros((channels, groups), dtype=int)
    for ch in range(channels):
        slot = ch % samples_per_period
        for group in range(groups):
            sample_index[ch, group] = group * samples_per_period + slot
            sample_slot[ch, group] = slot

    fig, ax = plt.subplots(figsize=(9.0, 3.8), constrained_layout=True)
    image = ax.imshow(sample_slot, aspect="auto", cmap=plt.get_cmap("tab10", samples_per_period))
    del image

    for ch in range(channels):
        for group in range(groups):
            ax.text(
                group,
                ch,
                f"x{sample_index[ch, group]}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_title("Grouped FIFO samples mapped to summed PWM channels")
    ax.set_xlabel("PWM period")
    ax.set_ylabel("Physical channel")
    ax.set_xticks(np.arange(groups), [str(idx) for idx in range(groups)])
    phases = np.arange(channels, dtype=np.float64) / float(channels)
    ax.set_yticks(
        np.arange(channels),
        [f"ch {ch}, phase {phase:.2f}" for ch, phase in enumerate(phases)],
    )
    ax.set_xticks(np.arange(-0.5, groups, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, channels, 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    path = output_dir / "pwm_grouped_mapping.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_waveform_figure(
    pwm_lab,
    output_dir: Path,
    *,
    samples_per_period: int,
    channels: int,
    groups: int,
    period_samples: int,
) -> Path:
    config = pwm_lab.PwmConfig(f_clk=float(period_samples), f_pwm=1.0, resolution_bits=8)
    samples = _demo_samples(groups * samples_per_period)
    raw = pwm_lab.pwm_kind2_fifo_grouped_multichannel(
        samples,
        config,
        samples_per_period=samples_per_period,
        channels=channels,
        normalize_sum=False,
    )
    normalized = raw / float(channels)
    grouped = samples.reshape(groups, samples_per_period)
    group_average = grouped.mean(axis=1)
    period_average = normalized.reshape(groups, period_samples).mean(axis=1)

    periods_to_plot = min(groups, 5)
    raw_points = periods_to_plot * period_samples
    pwm_x = np.arange(raw_points, dtype=np.float64) / float(period_samples)
    sample_x = np.arange(samples.size, dtype=np.float64) / float(samples_per_period)
    group_x = np.arange(groups, dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.0), sharex=False, constrained_layout=True)

    axes[0].stem(sample_x, samples, basefmt=" ", linefmt="C0-", markerfmt="C0o")
    axes[0].step(
        np.r_[group_x, groups],
        np.r_[group_average, group_average[-1]],
        where="post",
        color="C3",
        linewidth=1.8,
        label="group average",
    )
    axes[0].set_title("FIFO samples grouped before summed PWM")
    axes[0].set_ylabel("input")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    axes[1].step(pwm_x, raw[:raw_points], where="post", color="C1")
    axes[1].set_title("Raw summed PWM levels")
    axes[1].set_ylabel("sum")
    axes[1].set_ylim(-0.2, channels + 0.2)
    axes[1].set_yticks(np.arange(channels + 1))
    axes[1].grid(True, alpha=0.25)

    axes[2].step(pwm_x, normalized[:raw_points], where="post", color="C2")
    axes[2].set_title("Normalized summed PWM")
    axes[2].set_ylabel("sum / channels")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(group_x, group_average, "o-", label="FIFO group average")
    axes[3].plot(group_x, period_average, "s--", label="PWM period average")
    axes[3].set_title("Period averages check the low-frequency envelope")
    axes[3].set_xlabel("PWM period")
    axes[3].set_ylabel("average")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.25)

    path = output_dir / "pwm_grouped_waveform.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _grouped_channel_waveforms(
    pwm_lab,
    samples: np.ndarray,
    config,
    *,
    samples_per_period: int,
    channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.clip(np.asarray(samples, dtype=np.float64), 0.0, 1.0)
    groups = values.size // samples_per_period
    grouped = values[: groups * samples_per_period].reshape(groups, samples_per_period)
    traces = np.zeros((channels, groups, config.period_samples), dtype=np.float64)

    for ch in range(channels):
        sample_slot = ch % samples_per_period
        phase = ch / float(channels)
        carrier = pwm_lab.triangle_carrier(config.period_samples, phase=phase)
        channel_values = grouped[:, sample_slot, None]
        traces[ch] = ((channel_values > carrier[None, :]) | (channel_values >= 1.0)).astype(np.float64)

    return traces.reshape(channels, -1), grouped


def _save_time_realization_figure(
    pwm_lab,
    output_dir: Path,
    *,
    samples_per_period: int,
    channels: int,
    groups: int,
    period_samples: int,
) -> Path:
    config = pwm_lab.PwmConfig(f_clk=float(period_samples), f_pwm=1.0, resolution_bits=8)
    samples = _demo_samples(groups * samples_per_period)
    channel_waveforms, grouped = _grouped_channel_waveforms(
        pwm_lab,
        samples,
        config,
        samples_per_period=samples_per_period,
        channels=channels,
    )
    raw = channel_waveforms.sum(axis=0)
    normalized = raw / float(channels)

    periods_to_plot = min(groups, 5)
    points = periods_to_plot * period_samples
    time = np.arange(points, dtype=np.float64) / float(period_samples)
    period_edges = np.arange(periods_to_plot + 1, dtype=np.float64)
    offset_step = 1.35

    fig_height = max(6.5, 0.55 * channels + 4.5)
    fig, axes = plt.subplots(3, 1, figsize=(11.0, fig_height), sharex=True, constrained_layout=True)

    for ch in range(channels):
        offset = ch * offset_step
        axes[0].step(
            time,
            channel_waveforms[ch, :points] + offset,
            where="post",
            linewidth=1.2,
            label=f"ch {ch}, phase {ch / channels:.2f}",
        )
    axes[0].set_title("Per-channel PWM time realization")
    axes[0].set_ylabel("channel output")
    axes[0].set_yticks(
        [ch * offset_step + 0.5 for ch in range(channels)],
        [f"ch {ch}" for ch in range(channels)],
    )
    axes[0].set_ylim(-0.3, channels * offset_step)
    axes[0].grid(True, alpha=0.25)
    if channels <= 6:
        axes[0].legend(loc="upper right", ncols=2, fontsize=8)

    axes[1].step(time, raw[:points], where="post", color="C1")
    axes[1].set_title("Raw summed PWM")
    axes[1].set_ylabel("sum")
    axes[1].set_ylim(-0.2, channels + 0.2)
    axes[1].set_yticks(np.arange(channels + 1))
    axes[1].grid(True, alpha=0.25)

    group_average = grouped.mean(axis=1)
    axes[2].step(time, normalized[:points], where="post", color="C2", label="normalized summed PWM")
    axes[2].step(
        period_edges[: periods_to_plot + 1],
        np.r_[group_average[:periods_to_plot], group_average[periods_to_plot - 1]],
        where="post",
        color="C3",
        linewidth=2.0,
        label="FIFO group average",
    )
    axes[2].set_title("Normalized sum against FIFO group average")
    axes[2].set_xlabel("PWM periods")
    axes[2].set_ylabel("normalized")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.25)

    for ax in axes:
        for edge in period_edges:
            ax.axvline(edge, color="black", alpha=0.12, linewidth=0.8)

    path = output_dir / "pwm_grouped_time_realization.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_spectrum_figure(
    pwm_lab,
    output_dir: Path,
    *,
    samples_per_period: int,
    channels: int,
    groups: int,
    period_samples: int,
) -> Path:
    config = pwm_lab.PwmConfig(f_clk=float(period_samples), f_pwm=1.0, resolution_bits=8)
    samples = _demo_samples(groups * samples_per_period)
    normalized = pwm_lab.pwm_kind2_fifo_grouped_multichannel(
        samples,
        config,
        samples_per_period=samples_per_period,
        channels=channels,
    )
    group_average = samples.reshape(groups, samples_per_period).mean(axis=1)
    period_average = normalized.reshape(groups, period_samples).mean(axis=1)

    group_freqs, group_amps = pwm_lab.amplitude_spectrum(group_average, 1.0, remove_dc=True)
    period_freqs, period_amps = pwm_lab.amplitude_spectrum(period_average, 1.0, remove_dc=True)
    pwm_freqs, pwm_amps = pwm_lab.amplitude_spectrum(normalized, float(period_samples), remove_dc=True)

    eps = 1e-12
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), constrained_layout=True)

    axes[0].semilogy(group_freqs, np.maximum(group_amps, eps), "o-", label="FIFO group average")
    axes[0].semilogy(period_freqs, np.maximum(period_amps, eps), "s--", label="PWM period average")
    axes[0].set_title("Low-frequency envelope spectrum")
    axes[0].set_xlabel("frequency / f_pwm")
    axes[0].set_ylabel("amplitude")
    axes[0].set_xlim(0.0, 0.5)
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="upper right")

    max_freq = min(float(period_samples) / 2.0, 8.0)
    mask = pwm_freqs <= max_freq
    axes[1].semilogy(pwm_freqs[mask], np.maximum(pwm_amps[mask], eps), color="C2")
    for harmonic in range(1, int(max_freq) + 1):
        axes[1].axvline(float(harmonic), color="black", alpha=0.12, linewidth=0.8)
    axes[1].set_title("Summed PWM spectrum")
    axes[1].set_xlabel("frequency / f_pwm")
    axes[1].set_ylabel("amplitude")
    axes[1].set_xlim(0.0, max_freq)
    axes[1].grid(True, which="both", alpha=0.25)

    path = output_dir / "pwm_grouped_spectrum.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate grouped FIFO PWM demo figures.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--samples-per-period", type=int, default=2)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--spectrum-groups", type=int, default=256)
    parser.add_argument("--period-samples", type=int, default=64)
    args = parser.parse_args(argv)

    if args.samples_per_period <= 0:
        raise ValueError("samples_per_period must be positive")
    if args.channels <= 0:
        raise ValueError("channels must be positive")
    if args.channels % args.samples_per_period != 0:
        raise ValueError("channels must be an integer multiple of samples_per_period")
    if args.groups <= 0:
        raise ValueError("groups must be positive")
    if args.spectrum_groups <= 0:
        raise ValueError("spectrum_groups must be positive")
    if args.period_samples < 4:
        raise ValueError("period_samples must be at least 4")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pwm_lab = _load_pwm_lab()

    paths = [
        _save_mapping_figure(
            output_dir,
            samples_per_period=args.samples_per_period,
            channels=args.channels,
            groups=args.groups,
        ),
        _save_waveform_figure(
            pwm_lab,
            output_dir,
            samples_per_period=args.samples_per_period,
            channels=args.channels,
            groups=args.groups,
            period_samples=args.period_samples,
        ),
        _save_time_realization_figure(
            pwm_lab,
            output_dir,
            samples_per_period=args.samples_per_period,
            channels=args.channels,
            groups=args.groups,
            period_samples=args.period_samples,
        ),
        _save_spectrum_figure(
            pwm_lab,
            output_dir,
            samples_per_period=args.samples_per_period,
            channels=args.channels,
            groups=args.spectrum_groups,
            period_samples=args.period_samples,
        ),
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
