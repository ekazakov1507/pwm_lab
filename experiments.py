from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import SpectrumResult, moving_average_decimate, peak_frequency, spectra_for_waveforms
from .delta_sigma import (
    delta_sigma_first_order,
    delta_sigma_first_order_bipolar,
    delta_sigma_first_order_bipolar_fifo_parallel,
    delta_sigma_first_order_bipolar_fifo_round_robin,
    delta_sigma_first_order_bipolar_multichannel,
    delta_sigma_first_order_fifo_parallel,
    delta_sigma_first_order_fifo_round_robin,
    delta_sigma_first_order_multichannel,
    delta_sigma_first_order_signed,
    delta_sigma_first_order_signed_fifo_parallel,
    delta_sigma_first_order_signed_fifo_round_robin,
    delta_sigma_first_order_signed_multichannel,
    delta_sigma_second_order,
    delta_sigma_second_order_bipolar,
    delta_sigma_second_order_bipolar_fifo_parallel,
    delta_sigma_second_order_bipolar_fifo_round_robin,
    delta_sigma_second_order_bipolar_multichannel,
    delta_sigma_second_order_fifo_parallel,
    delta_sigma_second_order_fifo_round_robin,
    delta_sigma_second_order_multichannel,
    delta_sigma_second_order_signed,
    delta_sigma_second_order_signed_fifo_parallel,
    delta_sigma_second_order_signed_fifo_round_robin,
    delta_sigma_second_order_signed_multichannel,
)
from .pdm import (
    pdm_first_order,
    pdm_first_order_bipolar,
    pdm_first_order_bipolar_multichannel,
    pdm_first_order_multichannel,
    pdm_second_order,
    pdm_second_order_bipolar,
    pdm_second_order_bipolar_multichannel,
    pdm_second_order_multichannel,
)
from .pwm import PwmConfig, pwm_kind2_latched, pwm_kind2_phase_interleaved, pwm_kind2_same_phase_parallel
from .signals import sine_samples


@dataclass(frozen=True)
class FifoRatePlan:
    f_data: float
    f_pwm: float
    channels: int
    write_to_read_ratio: float
    effective_read_rate: float
    residual_decimation: float
    min_channels_without_decimation: int


def plan_fifo_rates(f_data: float, f_pwm: float, channels: int = 1) -> FifoRatePlan:
    """Summarize FIFO write/read rates for PWM kind 2."""
    if f_data <= 0 or f_pwm <= 0:
        raise ValueError("rates must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")
    effective_read_rate = channels * f_pwm
    return FifoRatePlan(
        f_data=float(f_data),
        f_pwm=float(f_pwm),
        channels=int(channels),
        write_to_read_ratio=float(f_data / f_pwm),
        effective_read_rate=float(effective_read_rate),
        residual_decimation=float(f_data / effective_read_rate),
        min_channels_without_decimation=int(np.ceil(f_data / f_pwm)),
    )


def delta_sigma_spectra_for_samples(
    samples: np.ndarray,
    sample_rate: float,
    *,
    signed: bool = False,
    bipolar: bool = False,
    channels: int = 1,
    orders: tuple[int, ...] = (1, 2),
    strategy: str = "same_input",
    include_input: bool = True,
    peak_count: int = 8,
    f_min: float = 0.0,
    f_max: float | None = None,
    min_spacing_hz: float = 0.0,
    remove_dc: bool = True,
) -> dict[str, SpectrumResult]:
    """Compute spectra for one-bit delta-sigma outputs.

    Delta-sigma emits one output sample per input sample, so sample_rate must be
    the modulator data/output sample rate, not a PWM carrier clock. For
    ``strategy="fifo_parallel"``, the grouped output sample rate is
    ``sample_rate / channels``.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")
    if signed and bipolar:
        raise ValueError("signed and bipolar modes are mutually exclusive")
    if strategy not in {"same_input", "fifo_parallel", "fifo_round_robin"}:
        raise ValueError("strategy must be 'same_input', 'fifo_parallel', or 'fifo_round_robin'")
    selected_orders = tuple(orders)
    if not selected_orders:
        raise ValueError("orders must contain at least one order")
    if any(order not in (1, 2) for order in selected_orders):
        raise ValueError("orders must contain only 1 and/or 2")

    spectra: dict[str, SpectrumResult] = {}
    if include_input:
        spectra.update(
            spectra_for_waveforms(
                {"input": values},
                sample_rate,
                peak_count=peak_count,
                f_min=f_min,
                f_max=f_max,
                min_spacing_hz=min_spacing_hz,
                remove_dc=remove_dc,
            )
        )

    waveforms: dict[str, object] = {}
    for order in selected_orders:
        order_name = "first" if order == 1 else "second"
        base_name = f"delta_sigma_{order_name}_order"

        if bipolar:
            if strategy == "fifo_parallel":
                name = f"{base_name}_bipolar_fifo_parallel"
                value = (
                    delta_sigma_first_order_bipolar_fifo_parallel(values, channels)
                    if order == 1
                    else delta_sigma_second_order_bipolar_fifo_parallel(values, channels)
                )
            elif strategy == "fifo_round_robin":
                name = f"{base_name}_bipolar_fifo_round_robin"
                value = (
                    delta_sigma_first_order_bipolar_fifo_round_robin(values, channels)
                    if order == 1
                    else delta_sigma_second_order_bipolar_fifo_round_robin(values, channels)
                )
            elif channels == 1:
                name = f"{base_name}_bipolar"
                value = (
                    delta_sigma_first_order_bipolar(values)
                    if order == 1
                    else delta_sigma_second_order_bipolar(values)
                )
            else:
                name = f"{base_name}_bipolar_multichannel"
                value = (
                    delta_sigma_first_order_bipolar_multichannel(values, channels)
                    if order == 1
                    else delta_sigma_second_order_bipolar_multichannel(values, channels)
                )
        elif signed:
            if strategy == "fifo_parallel":
                name = f"{base_name}_signed_fifo_parallel"
                value = (
                    delta_sigma_first_order_signed_fifo_parallel(values, channels)
                    if order == 1
                    else delta_sigma_second_order_signed_fifo_parallel(values, channels)
                )
            elif strategy == "fifo_round_robin":
                name = f"{base_name}_signed_fifo_round_robin"
                value = (
                    delta_sigma_first_order_signed_fifo_round_robin(values, channels)
                    if order == 1
                    else delta_sigma_second_order_signed_fifo_round_robin(values, channels)
                )
            elif channels == 1:
                name = f"{base_name}_signed"
                value = (
                    delta_sigma_first_order_signed(values)
                    if order == 1
                    else delta_sigma_second_order_signed(values)
                )
            else:
                name = f"{base_name}_signed_multichannel"
                value = (
                    delta_sigma_first_order_signed_multichannel(values, channels)
                    if order == 1
                    else delta_sigma_second_order_signed_multichannel(values, channels)
                )
        else:
            if strategy == "fifo_parallel":
                name = f"{base_name}_fifo_parallel"
                value = (
                    delta_sigma_first_order_fifo_parallel(values, channels)
                    if order == 1
                    else delta_sigma_second_order_fifo_parallel(values, channels)
                )
            elif strategy == "fifo_round_robin":
                name = f"{base_name}_fifo_round_robin"
                value = (
                    delta_sigma_first_order_fifo_round_robin(values, channels)
                    if order == 1
                    else delta_sigma_second_order_fifo_round_robin(values, channels)
                )
            elif channels == 1:
                name = base_name
                value = delta_sigma_first_order(values) if order == 1 else delta_sigma_second_order(values)
            else:
                name = f"{base_name}_multichannel"
                value = (
                    delta_sigma_first_order_multichannel(values, channels)
                    if order == 1
                    else delta_sigma_second_order_multichannel(values, channels)
                )
        waveforms[name] = value

    output_sample_rate = sample_rate / float(channels) if strategy == "fifo_parallel" else sample_rate
    spectra.update(
        spectra_for_waveforms(
            waveforms,
            output_sample_rate,
            peak_count=peak_count,
            f_min=f_min,
            f_max=f_max,
            min_spacing_hz=min_spacing_hz,
            remove_dc=remove_dc,
        )
    )
    return spectra


def pdm_spectra_for_samples(
    samples: np.ndarray,
    sample_rate: float,
    *,
    bipolar: bool = False,
    channels: int = 1,
    include_input: bool = True,
    peak_count: int = 8,
    f_min: float = 0.0,
    f_max: float | None = None,
    min_spacing_hz: float = 0.0,
    remove_dc: bool = True,
) -> dict[str, SpectrumResult]:
    """Compute spectra for first-order and second-order PDM outputs.

    PDM emits one output sample per input sample, so sample_rate must be the PDM
    data/output sample rate, not a PWM carrier clock.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")

    waveforms: dict[str, object] = {}
    if include_input:
        waveforms["input"] = values

    if bipolar:
        if channels == 1:
            waveforms["pdm_first_order"] = pdm_first_order_bipolar(values)
            waveforms["pdm_second_order"] = pdm_second_order_bipolar(values)
        else:
            waveforms["pdm_first_order_multichannel"] = pdm_first_order_bipolar_multichannel(values, channels)
            waveforms["pdm_second_order_multichannel"] = pdm_second_order_bipolar_multichannel(values, channels)
    else:
        if channels == 1:
            waveforms["pdm_first_order"] = pdm_first_order(values)
            waveforms["pdm_second_order"] = pdm_second_order(values)
        else:
            waveforms["pdm_first_order_multichannel"] = pdm_first_order_multichannel(values, channels)
            waveforms["pdm_second_order_multichannel"] = pdm_second_order_multichannel(values, channels)

    return spectra_for_waveforms(
        waveforms,
        sample_rate,
        peak_count=peak_count,
        f_min=f_min,
        f_max=f_max,
        min_spacing_hz=min_spacing_hz,
        remove_dc=remove_dc,
    )


def simulate_fifo_parallel_idea(
    *,
    f_data: float = 1e6,
    f_pwm: float = 200e3,
    f_clk: float = 100e6,
    f_signal: float = 20e3,
    channels: int | None = None,
    n_data_samples: int = 4096,
    resolution_bits: int = 8,
) -> dict[str, object]:
    """Compare one-channel, same-phase parallel, and phase-interleaved PWM kind 2."""
    if channels is None:
        channels = int(np.ceil(f_data / f_pwm))
    config = PwmConfig(f_clk=f_clk, f_pwm=f_pwm, resolution_bits=resolution_bits)
    _, samples = sine_samples(f_signal, f_data, n_data_samples)

    one_channel = pwm_kind2_latched(samples, config)
    same_phase = pwm_kind2_same_phase_parallel(samples, config, channels)
    interleaved = pwm_kind2_phase_interleaved(samples, config, channels)
    averaged_reference = moving_average_decimate(samples, channels)

    one_peak = peak_frequency(one_channel, f_clk, f_min=1.0, f_max=min(0.45 * f_pwm, 10 * f_signal))
    same_peak = peak_frequency(same_phase, f_clk, f_min=1.0, f_max=min(0.45 * f_pwm, 10 * f_signal))
    interleaved_peak = peak_frequency(
        interleaved,
        f_clk,
        f_min=1.0,
        f_max=min(0.45 * channels * f_pwm, 10 * f_signal),
    )
    reference_peak = peak_frequency(
        averaged_reference,
        f_data / channels,
        f_min=1.0,
        f_max=min(0.45 * f_data / channels, 10 * f_signal),
    )

    return {
        "config": config,
        "plan": plan_fifo_rates(f_data, f_pwm, channels),
        "samples": samples,
        "one_channel": one_channel,
        "same_phase": same_phase,
        "interleaved": interleaved,
        "averaged_reference": averaged_reference,
        "peaks_hz": {
            "one_channel": one_peak,
            "same_phase": same_peak,
            "phase_interleaved": interleaved_peak,
            "averaged_reference": reference_peak,
        },
    }
