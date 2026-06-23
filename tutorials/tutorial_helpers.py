from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]


def load_pwm_lab():
    if "pwm_lab" in sys.modules:
        return sys.modules["pwm_lab"]

    spec = importlib.util.spec_from_file_location(
        "pwm_lab",
        REPO_DIR / "__init__.py",
        submodule_search_locations=[str(REPO_DIR)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pwm_lab package from local checkout")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pwm_lab"] = module
    spec.loader.exec_module(module)
    return module


_PWM_LAB = load_pwm_lab()
amplitude_spectrum = _PWM_LAB.amplitude_spectrum
moving_average_decimate = _PWM_LAB.moving_average_decimate
spectrum_result = _PWM_LAB.spectrum_result
triangle_carrier = _PWM_LAB.triangle_carrier


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (11.0, 5.0),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def time_us(sample_rate: float, n_samples: int) -> np.ndarray:
    return np.arange(n_samples, dtype=np.float64) / float(sample_rate) * 1e6


def plot_pwm_carrier_output(
    input_values: np.ndarray,
    carrier: np.ndarray,
    output: np.ndarray,
    *,
    sample_rate: float,
    max_points: int | None = None,
    input_scale: float = 255.0,
    title: str = "PWM time realization",
) -> tuple[plt.Figure, np.ndarray]:
    values = np.asarray(input_values, dtype=np.float64)
    carr = np.asarray(carrier, dtype=np.float64)
    y = np.asarray(output, dtype=np.float64)
    n = min(values.size, carr.size, y.size)
    if max_points is not None:
        n = min(n, max_points)
    t = time_us(sample_rate, n)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 5.3), sharex=True, constrained_layout=True)
    axes[0].plot(t, carr[:n] * input_scale, color="C0", linewidth=1.4, label="carrier")
    axes[0].plot(
        t,
        values[:n] * input_scale,
        ".-",
        color="C1",
        markersize=3.2,
        linewidth=1.0,
        label="input",
    )
    axes[0].set_title(title)
    axes[0].set_ylabel("code")
    axes[0].legend(loc="upper right")

    axes[1].step(t, y[:n], where="post", color="C0", linewidth=1.2)
    axes[1].set_xlabel("time, us")
    axes[1].set_ylabel("output")
    axes[1].set_ylim(float(np.min(y[:n])) - 0.15, float(np.max(y[:n])) + 0.15)
    return fig, axes


def plot_bitstream(
    input_values: np.ndarray,
    output: np.ndarray,
    *,
    sample_rate: float,
    max_points: int = 400,
    title: str = "One-bit stream",
) -> tuple[plt.Figure, np.ndarray]:
    values = np.asarray(input_values, dtype=np.float64)
    y = np.asarray(output, dtype=np.float64)
    n = min(values.size, y.size, max_points)
    t = time_us(sample_rate, n)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 4.8), sharex=True, constrained_layout=True)
    axes[0].plot(t, values[:n], color="C1", linewidth=1.6)
    axes[0].set_title(title)
    axes[0].set_ylabel("input")
    axes[0].set_ylim(float(np.min(values[:n])) - 0.1, float(np.max(values[:n])) + 0.1)

    axes[1].step(t, y[:n], where="post", color="C0", linewidth=1.0)
    axes[1].set_xlabel("time, us")
    axes[1].set_ylabel("stream")
    axes[1].set_ylim(float(np.min(y[:n])) - 0.2, float(np.max(y[:n])) + 0.2)
    return fig, axes


def plot_channel_stack(
    traces: np.ndarray,
    *,
    sample_rate: float,
    max_points: int = 600,
    summed: np.ndarray | None = None,
    title: str = "Per-channel time realization",
) -> tuple[plt.Figure, np.ndarray]:
    channel_values = np.asarray(traces, dtype=np.float64)
    if channel_values.ndim != 2:
        raise ValueError("traces must have shape (channels, samples)")
    n = min(channel_values.shape[1], max_points)
    t = time_us(sample_rate, n)
    channels = channel_values.shape[0]
    offset_step = 1.35

    rows = 2 if summed is not None else 1
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(11.0, max(4.2, 0.45 * channels + 3.2)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ch in range(channels):
        axes[0].step(
            t,
            channel_values[ch, :n] + ch * offset_step,
            where="post",
            linewidth=1.0,
            label=f"ch {ch}",
        )
    axes[0].set_title(title)
    axes[0].set_ylabel("channels")
    axes[0].set_yticks([ch * offset_step + 0.5 for ch in range(channels)], [f"ch {ch}" for ch in range(channels)])
    axes[0].set_ylim(-0.3, channels * offset_step)
    if channels <= 8:
        axes[0].legend(loc="upper right", ncols=2, fontsize=8)

    if summed is not None:
        s = np.asarray(summed, dtype=np.float64)
        axes[1].step(t, s[:n], where="post", color="C3", linewidth=1.2)
        axes[1].set_ylabel("sum")
        axes[1].set_xlabel("time, us")
    else:
        axes[0].set_xlabel("time, us")
    return fig, axes


def plot_spectra(
    waveforms: dict[str, np.ndarray],
    *,
    sample_rate: float,
    f_max: float | None = None,
    f_scale: float = 1e3,
    f_unit: str = "kHz",
    floor_db: float = -140.0,
    remove_dc: bool = True,
    title: str = "Amplitude spectra",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    for name, waveform in waveforms.items():
        freqs, amps = amplitude_spectrum(np.asarray(waveform, dtype=np.float64), sample_rate, remove_dc=remove_dc)
        if f_max is not None:
            mask = freqs <= f_max
            freqs = freqs[mask]
            amps = amps[mask]
        db = 20.0 * np.log10(np.maximum(amps, 10.0 ** (floor_db / 20.0)))
        ax.plot(freqs / f_scale, db, linewidth=1.2, label=name)
    ax.set_title(title)
    ax.set_xlabel(f"frequency, {f_unit}")
    ax.set_ylabel("amplitude, dBFS")
    ax.set_ylim(floor_db, 6.0)
    ax.legend(loc="best")
    return fig, ax


def print_peak_table(
    waveforms: dict[str, np.ndarray],
    *,
    sample_rate: float,
    f_min: float = 1.0,
    f_max: float | None = None,
    count: int = 5,
    f_scale: float = 1e3,
    f_unit: str = "kHz",
) -> None:
    specs = {
        name: spectrum_result(
            name,
            np.asarray(waveform, dtype=np.float64),
            sample_rate,
            peak_count=count,
            f_min=f_min,
            f_max=f_max,
            remove_dc=True,
        )
        for name, waveform in waveforms.items()
    }
    for name, spec in specs.items():
        print(name)
        for freq, amp in spec.peaks:
            print(f"  {freq / f_scale:10.4f} {f_unit}  amplitude={amp:.6g}")


def plot_moving_average_reconstruction(
    input_values: np.ndarray,
    modulated: np.ndarray,
    *,
    input_sample_rate: float,
    modulated_sample_rate: float,
    average_factor: int,
    title: str = "Moving-average reconstruction",
) -> tuple[plt.Figure, plt.Axes]:
    x = np.asarray(input_values, dtype=np.float64)
    y = np.asarray(modulated, dtype=np.float64)
    recon = moving_average_decimate(y, average_factor)
    t_input = np.arange(x.size, dtype=np.float64) / float(input_sample_rate) * 1e3
    t_recon = (
        np.arange(recon.size, dtype=np.float64) * float(average_factor) / float(modulated_sample_rate) * 1e3
    )

    fig, ax = plt.subplots(figsize=(11.0, 4.5), constrained_layout=True)
    ax.plot(t_input, x, color="C1", linewidth=1.8, label="input")
    ax.plot(t_recon, recon, color="C0", linewidth=1.2, label=f"average by {average_factor}")
    ax.set_title(title)
    ax.set_xlabel("time, ms")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right")
    return fig, ax


def pwm_kind2_channel_waveforms(
    samples: np.ndarray,
    config,
    *,
    channels: int,
    phase_offsets: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if phase_offsets is None:
        phases = np.arange(channels, dtype=np.float64) / float(channels)
    else:
        phases = np.asarray(phase_offsets, dtype=np.float64)
        if phases.size != channels:
            raise ValueError("phase_offsets must have one value per channel")
    traces = np.zeros((channels, values.size, config.period_samples), dtype=np.float64)
    for ch, phase in enumerate(phases):
        carrier = triangle_carrier(config.period_samples, phase=float(phase))
        traces[ch] = ((values[:, None] > carrier[None, :]) | (values[:, None] >= 1.0)).astype(np.float64)
    return traces.reshape(channels, -1)


def grouped_fifo_channel_waveforms(
    samples: np.ndarray,
    config,
    *,
    samples_per_period: int,
    channels: int,
    phase_offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=np.float64)
    groups = values.size // samples_per_period
    grouped = values[: groups * samples_per_period].reshape(groups, samples_per_period)
    if phase_offsets is None:
        phases = np.arange(channels, dtype=np.float64) / float(channels)
    else:
        phases = np.asarray(phase_offsets, dtype=np.float64)
        if phases.size != channels:
            raise ValueError("phase_offsets must have one value per channel")

    traces = np.zeros((channels, groups, config.period_samples), dtype=np.float64)
    for ch, phase in enumerate(phases):
        sample_slot = ch % samples_per_period
        carrier = triangle_carrier(config.period_samples, phase=float(phase))
        traces[ch] = ((grouped[:, sample_slot, None] > carrier[None, :]) | (grouped[:, sample_slot, None] >= 1.0)).astype(
            np.float64
        )
    return traces.reshape(channels, -1), grouped


def plot_grouped_fifo_channel_comparison(
    samples: np.ndarray,
    config,
    *,
    samples_per_period: int,
    channels: int,
    phase_offsets: np.ndarray | None = None,
    max_periods: int = 8,
    input_scale: float = 255.0,
    title: str = "Grouped FIFO PWM per-channel comparison",
) -> tuple[plt.Figure, np.ndarray]:
    values = np.asarray(samples, dtype=np.float64)
    groups = values.size // samples_per_period
    grouped = values[: groups * samples_per_period].reshape(groups, samples_per_period)
    periods = min(groups, max_periods)
    if periods <= 0:
        raise ValueError("at least one complete FIFO group is required")

    if phase_offsets is None:
        phases = np.arange(channels, dtype=np.float64) / float(channels)
    else:
        phases = np.asarray(phase_offsets, dtype=np.float64)
        if phases.size != channels:
            raise ValueError("phase_offsets must have one value per channel")

    points = periods * config.period_samples
    t = time_us(config.f_clk, points)
    period_edges = np.arange(periods + 1, dtype=np.float64) * config.period_samples / float(config.f_clk) * 1e6

    fig, axes = plt.subplots(
        channels * 2,
        1,
        figsize=(11.0, max(5.8, 1.85 * channels + 1.4)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    fig.suptitle(title)

    for ch, phase in enumerate(phases):
        sample_slot = ch % samples_per_period
        sample_indices = np.arange(periods) * samples_per_period + sample_slot
        channel_samples = grouped[:periods, sample_slot]
        carrier_one = triangle_carrier(config.period_samples, phase=float(phase))
        carrier = np.tile(carrier_one, periods)
        latched = np.repeat(channel_samples, config.period_samples)
        output = ((latched > carrier) | (latched >= 1.0)).astype(np.float64)

        top = axes[2 * ch]
        bottom = axes[2 * ch + 1]

        top.plot(t, carrier * input_scale, color="C0", linewidth=1.1, label="carrier")
        top.step(
            period_edges,
            np.r_[channel_samples, channel_samples[-1]] * input_scale,
            where="post",
            color="C1",
            linewidth=1.4,
            label="latched FIFO sample",
        )
        top.plot(
            period_edges[:-1],
            channel_samples * input_scale,
            "o",
            color="C1",
            markersize=3.5,
            label="FIFO read",
        )
        for x_pos, sample_index, sample_value in zip(period_edges[:-1], sample_indices, channel_samples):
            top.annotate(
                f"x{sample_index}",
                xy=(x_pos, sample_value * input_scale),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7,
                color="C1",
            )

        top.set_ylabel(f"ch {ch}\ncode")
        top.set_ylim(-0.05 * input_scale, 1.08 * input_scale)
        top.set_title(f"ch {ch}: FIFO slot {sample_slot}, carrier phase {phase:.2f}", fontsize=10)
        if ch == 0:
            top.legend(loc="upper right", ncols=3, fontsize=8)

        bottom.step(t, output, where="post", color=f"C{ch % 10}", linewidth=1.1)
        bottom.set_ylabel("output")
        bottom.set_ylim(-0.15, 1.15)
        bottom.set_yticks([0.0, 1.0])

        for ax in (top, bottom):
            for edge in period_edges:
                ax.axvline(edge, color="black", alpha=0.12, linewidth=0.8)

    axes[-1].set_xlabel("time, us")
    return fig, axes


def show_grouped_mapping(samples_per_period: int, channels: int, groups: int = 4) -> tuple[plt.Figure, plt.Axes]:
    sample_index = np.zeros((channels, groups), dtype=int)
    sample_slot = np.zeros((channels, groups), dtype=int)
    for ch in range(channels):
        slot = ch % samples_per_period
        for group in range(groups):
            sample_index[ch, group] = group * samples_per_period + slot
            sample_slot[ch, group] = slot

    fig, ax = plt.subplots(figsize=(8.5, 0.45 * channels + 2.0), constrained_layout=True)
    ax.imshow(sample_slot, aspect="auto", cmap=plt.get_cmap("tab10", samples_per_period))
    for ch in range(channels):
        for group in range(groups):
            ax.text(group, ch, f"x{sample_index[ch, group]}", ha="center", va="center", color="white", weight="bold")
    ax.set_title("Grouped FIFO mapping")
    ax.set_xlabel("PWM period")
    ax.set_ylabel("physical channel")
    ax.set_xticks(np.arange(groups), [str(i) for i in range(groups)])
    ax.set_yticks(np.arange(channels), [f"ch {i}" for i in range(channels)])
    ax.set_xticks(np.arange(-0.5, groups, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, channels, 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    return fig, ax
