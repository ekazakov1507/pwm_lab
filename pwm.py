from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._modulation import (
    BipolarOutput,
    as_unit_values as _as_unit_values,
    phase_offsets as _phase_offsets,
    split_signed_magnitude as _split_signed_magnitude,
    validate_channels as _validate_channels,
)


@dataclass(frozen=True)
class PwmConfig:
    """Clock-level PWM configuration."""

    f_clk: float
    f_pwm: float
    resolution_bits: int

    def __post_init__(self) -> None:
        if self.f_clk <= 0:
            raise ValueError("f_clk must be positive")
        if self.f_pwm <= 0:
            raise ValueError("f_pwm must be positive")
        if self.resolution_bits <= 0:
            raise ValueError("resolution_bits must be positive")

    @property
    def period_samples(self) -> int:
        period = int(round(float(self.f_clk) / float(self.f_pwm)))
        if period < 4:
            raise ValueError("PWM period must contain at least four clock samples")
        return period

    @property
    def levels(self) -> int:
        return 1 << int(self.resolution_bits)

    @property
    def actual_f_pwm(self) -> float:
        """Actual PWM frequency after rounding the clock period to samples."""
        return float(self.f_clk) / float(self.period_samples)

    @property
    def relative_frequency_error(self) -> float:
        """Relative difference between requested and actual PWM frequency."""
        return (self.actual_f_pwm - float(self.f_pwm)) / float(self.f_pwm)


class BipolarPwm(BipolarOutput):
    """Two-output PWM representation for a signed signal."""


def triangle_carrier(period_samples: int, *, phase: float = 0.0) -> np.ndarray:
    """Generate one normalized triangular carrier period in [0, 1]."""
    if period_samples < 4:
        raise ValueError("period_samples must be at least 4")
    n = np.arange(period_samples, dtype=np.float64)
    phase_pos = (n / period_samples + phase) % 1.0
    carrier = 1.0 - np.abs(2.0 * phase_pos - 1.0)
    return carrier


def _compare_pwm(values: np.ndarray, carrier: np.ndarray) -> np.ndarray:
    """PWM compare where 0 duty is always off and 1 duty is always on."""
    return ((values > carrier) | (values >= 1.0)).astype(np.float64)


def pwm_kind1(samples_at_clk: np.ndarray, config: PwmConfig, *, phase: float = 0.0) -> np.ndarray:
    """PWM kind 1: compare the current signal value with the current carrier value."""
    values = _as_unit_values(samples_at_clk)
    carrier = triangle_carrier(config.period_samples, phase=phase)
    tiled = np.resize(carrier, values.size)
    return _compare_pwm(values, tiled)


def pwm_kind1_multichannel(
    samples_at_clk: np.ndarray,
    config: PwmConfig,
    channels: int,
    *,
    phase_offsets: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """PWM kind 1 with phase-shifted carriers summed across channels."""
    phases = _phase_offsets(channels, phase_offsets)
    summed = np.zeros_like(np.asarray(samples_at_clk, dtype=np.float64))
    for phase in phases:
        summed += pwm_kind1(samples_at_clk, config, phase=float(phase))
    if normalize_sum:
        summed /= float(channels)
    return summed


def pwm_kind1_bipolar(samples_at_clk: np.ndarray, config: PwmConfig, *, phase: float = 0.0) -> BipolarPwm:
    """Bipolar PWM kind 1 for signed values in [-1, 1]."""
    positive, negative = _split_signed_magnitude(samples_at_clk)
    return BipolarPwm(
        positive=pwm_kind1(positive, config, phase=phase),
        negative=pwm_kind1(negative, config, phase=phase),
    )


def pwm_kind1_bipolar_multichannel(
    samples_at_clk: np.ndarray,
    config: PwmConfig,
    channels: int,
    *,
    phase_offsets: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> BipolarPwm:
    """Bipolar PWM kind 1 with phase-shifted carriers summed across channels."""
    positive, negative = _split_signed_magnitude(samples_at_clk)
    return BipolarPwm(
        positive=pwm_kind1_multichannel(
            positive,
            config,
            channels,
            phase_offsets=phase_offsets,
            normalize_sum=normalize_sum,
        ),
        negative=pwm_kind1_multichannel(
            negative,
            config,
            channels,
            phase_offsets=phase_offsets,
            normalize_sum=normalize_sum,
        ),
    )


def pwm_kind2_latched(
    samples: np.ndarray,
    config: PwmConfig,
    *,
    phase: float = 0.0,
    input_bits: int | None = None,
) -> np.ndarray:
    """PWM kind 2: one input sample is latched for one full PWM period."""
    values = _as_unit_values(samples, input_bits)
    carrier = triangle_carrier(config.period_samples, phase=phase)
    return _compare_pwm(values[:, None], carrier[None, :]).reshape(-1)


def pwm_kind2_multichannel_latched(
    samples: np.ndarray,
    config: PwmConfig,
    channels: int,
    *,
    phase_offsets: np.ndarray | None = None,
    input_bits: int | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """PWM kind 2 with several phase-shifted channels reading the same sample."""
    values = _as_unit_values(samples, input_bits)
    phases = _phase_offsets(channels, phase_offsets)
    summed = np.zeros((values.size, config.period_samples), dtype=np.float64)
    for phase in phases:
        carrier = triangle_carrier(config.period_samples, phase=float(phase))
        summed += _compare_pwm(values[:, None], carrier[None, :])
    if normalize_sum:
        summed /= float(channels)
    return summed.reshape(-1)


def pwm_kind2_bipolar_latched(
    samples: np.ndarray,
    config: PwmConfig,
    *,
    phase: float = 0.0,
) -> BipolarPwm:
    """Bipolar PWM kind 2 for signed values in [-1, 1]."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarPwm(
        positive=pwm_kind2_latched(positive, config, phase=phase),
        negative=pwm_kind2_latched(negative, config, phase=phase),
    )


def pwm_kind2_bipolar_multichannel_latched(
    samples: np.ndarray,
    config: PwmConfig,
    channels: int,
    *,
    phase_offsets: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> BipolarPwm:
    """Bipolar PWM kind 2 with channels reading the same latched sample."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarPwm(
        positive=pwm_kind2_multichannel_latched(
            positive,
            config,
            channels,
            phase_offsets=phase_offsets,
            normalize_sum=normalize_sum,
        ),
        negative=pwm_kind2_multichannel_latched(
            negative,
            config,
            channels,
            phase_offsets=phase_offsets,
            normalize_sum=normalize_sum,
        ),
    )


def pwm_kind2_same_phase_parallel(
    samples: np.ndarray,
    config: PwmConfig,
    channels: int,
    *,
    input_bits: int | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """Read several FIFO samples per PWM period and modulate them with the same carrier.

    The summed low-frequency value is the average of a group of consecutive samples.
    This behaves like a simple hardware decimator, not like a time-ordered replay of
    every sample.
    """
    _validate_channels(channels)
    values = _as_unit_values(samples, input_bits)
    groups = values.size // channels
    if groups == 0:
        return np.array([], dtype=np.float64)
    grouped = values[: groups * channels].reshape(groups, channels)
    carrier = triangle_carrier(config.period_samples)
    summed = np.zeros((groups, config.period_samples), dtype=np.float64)
    for ch in range(channels):
        summed += _compare_pwm(grouped[:, ch, None], carrier[None, :])
    if normalize_sum:
        summed /= float(channels)
    return summed.reshape(-1)


def pwm_kind2_bipolar_same_phase_parallel(
    samples: np.ndarray,
    config: PwmConfig,
    channels: int,
    *,
    normalize_sum: bool = True,
) -> BipolarPwm:
    """Bipolar multi-sample PWM kind 2 with same-phase carriers."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarPwm(
        positive=pwm_kind2_same_phase_parallel(positive, config, channels, normalize_sum=normalize_sum),
        negative=pwm_kind2_same_phase_parallel(negative, config, channels, normalize_sum=normalize_sum),
    )


def pwm_kind2_fifo_grouped_multichannel(
    samples: np.ndarray,
    config: PwmConfig,
    samples_per_period: int,
    channels: int,
    *,
    phase_offsets: np.ndarray | None = None,
    input_bits: int | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """Read several FIFO samples per PWM period and sum them across channels.

    ``samples_per_period`` controls FIFO read throughput. ``channels`` controls
    how many physical PWM channels are summed. Each FIFO sample gets the same
    number of channels, so ``channels`` must be an integer multiple of
    ``samples_per_period``.

    The default channel mapping is ``sample_slot = channel % samples_per_period``.
    Default phases are evenly distributed across physical channels. The returned
    waveform is only the summed PWM channel model; it does not include an analog
    filter, transformer, load, or hardware timing model.
    """
    _validate_channels(samples_per_period)
    _validate_channels(channels)
    if channels % samples_per_period != 0:
        raise ValueError("channels must be an integer multiple of samples_per_period")
    values = _as_unit_values(samples, input_bits)
    groups = values.size // samples_per_period
    if groups == 0:
        return np.array([], dtype=np.float64)
    grouped = values[: groups * samples_per_period].reshape(groups, samples_per_period)
    phases = _phase_offsets(channels, phase_offsets)
    summed = np.zeros((groups, config.period_samples), dtype=np.float64)
    for ch, phase in enumerate(phases):
        sample_slot = ch % samples_per_period
        carrier = triangle_carrier(config.period_samples, phase=float(phase))
        summed += _compare_pwm(grouped[:, sample_slot, None], carrier[None, :])
    if normalize_sum:
        summed /= float(channels)
    return summed.reshape(-1)


def pwm_kind2_bipolar_fifo_grouped_multichannel(
    samples: np.ndarray,
    config: PwmConfig,
    samples_per_period: int,
    channels: int,
    *,
    phase_offsets: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> BipolarPwm:
    """Bipolar grouped FIFO PWM kind 2 summed across physical channels."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarPwm(
        positive=pwm_kind2_fifo_grouped_multichannel(
            positive,
            config,
            samples_per_period,
            channels,
            phase_offsets=phase_offsets,
            normalize_sum=normalize_sum,
        ),
        negative=pwm_kind2_fifo_grouped_multichannel(
            negative,
            config,
            samples_per_period,
            channels,
            phase_offsets=phase_offsets,
            normalize_sum=normalize_sum,
        ),
    )


def pwm_kind2_phase_interleaved(
    samples: np.ndarray,
    config: PwmConfig,
    channels: int,
    *,
    input_bits: int | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """Read several FIFO samples per PWM period using phase-shifted carriers."""
    _validate_channels(channels)
    values = _as_unit_values(samples, input_bits)
    groups = values.size // channels
    if groups == 0:
        return np.array([], dtype=np.float64)
    grouped = values[: groups * channels].reshape(groups, channels)
    carriers = np.stack(
        [triangle_carrier(config.period_samples, phase=ch / channels) for ch in range(channels)],
        axis=0,
    )
    summed = np.zeros((groups, config.period_samples), dtype=np.float64)
    for ch in range(channels):
        summed += _compare_pwm(grouped[:, ch, None], carriers[ch][None, :])
    if normalize_sum:
        summed /= float(channels)
    return summed.reshape(-1)


def pwm_kind2_bipolar_phase_interleaved(
    samples: np.ndarray,
    config: PwmConfig,
    channels: int,
    *,
    normalize_sum: bool = True,
) -> BipolarPwm:
    """Bipolar multi-sample PWM kind 2 with phase-interleaved carriers."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarPwm(
        positive=pwm_kind2_phase_interleaved(positive, config, channels, normalize_sum=normalize_sum),
        negative=pwm_kind2_phase_interleaved(negative, config, channels, normalize_sum=normalize_sum),
    )
