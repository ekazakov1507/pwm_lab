from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BipolarOutput:
    """Shared two-output representation for signed modulation results."""

    positive: np.ndarray
    negative: np.ndarray

    @property
    def differential(self) -> np.ndarray:
        return self.positive - self.negative


def as_unit_values(samples: np.ndarray, bits: int | None = None) -> np.ndarray:
    """Convert samples to clipped normalized values in [0, 1]."""
    values = np.asarray(samples, dtype=np.float64)
    if bits is None:
        return np.clip(values, 0.0, 1.0)
    if bits <= 0:
        raise ValueError("bits must be positive")
    max_code = float((1 << bits) - 1)
    return np.clip(values / max_code, 0.0, 1.0)


def as_signed_values(samples: np.ndarray) -> np.ndarray:
    """Convert samples to clipped signed values in [-1, 1]."""
    return np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)


def split_signed_magnitude(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split signed samples into positive and negative unipolar magnitudes."""
    values = as_signed_values(samples)
    positive = np.clip(values, 0.0, 1.0)
    negative = np.clip(-values, 0.0, 1.0)
    return positive, negative


def validate_channels(channels: int) -> None:
    if channels <= 0:
        raise ValueError("channels must be positive")


def phase_offsets(channels: int, offsets: np.ndarray | None = None) -> np.ndarray:
    """Return per-channel carrier phases."""
    validate_channels(channels)
    if offsets is None:
        return np.arange(channels, dtype=np.float64) / float(channels)
    phases = np.asarray(offsets, dtype=np.float64)
    if phases.size != channels:
        raise ValueError("phase_offsets length must match channels")
    return phases


def staggered_states(channels: int, states: np.ndarray | None = None) -> np.ndarray:
    """Return per-channel accumulator states in [0, 1)."""
    validate_channels(channels)
    if states is None:
        return np.arange(channels, dtype=np.float64) / float(channels)
    values = np.asarray(states, dtype=np.float64)
    if values.size != channels:
        raise ValueError("initial_states length must match channels")
    return np.mod(values, 1.0)


def state_array(
    channels: int,
    values: np.ndarray | None,
    *,
    default: float = 0.0,
    name: str,
    flatten: bool = False,
) -> np.ndarray:
    """Return a per-channel state array with shared validation."""
    validate_channels(channels)
    if values is None:
        return np.full(channels, default, dtype=np.float64)
    states = np.asarray(values, dtype=np.float64)
    if states.size != channels:
        raise ValueError(f"{name} length must match channels")
    if flatten:
        return states.reshape(-1)
    return states


def one_bit_output_states(channels: int, values: np.ndarray | None, *, name: str) -> np.ndarray:
    """Return per-channel previous outputs quantized to 0 or 1."""
    states = state_array(channels, values, name=name, flatten=True)
    return (states >= 0.5).astype(np.float64)


def one_bit_first_order(values: np.ndarray, *, initial_state: float = 0.0) -> np.ndarray:
    """First-order one-bit accumulator loop for normalized values in [0, 1]."""
    unit_values = np.asarray(values, dtype=np.float64)
    state = float(initial_state) % 1.0
    out = np.empty(unit_values.size, dtype=np.float64)
    for idx, value in enumerate(unit_values):
        state += float(value)
        if state >= 1.0:
            out[idx] = 1.0
            state -= 1.0
        else:
            out[idx] = 0.0
    return out


def one_bit_second_order(
    values: np.ndarray,
    *,
    initial_state1: float = 0.0,
    initial_state2: float = 0.0,
    initial_output: float = 0.0,
) -> np.ndarray:
    """Second-order one-bit error-feedback loop for normalized values in [0, 1]."""
    unit_values = np.asarray(values, dtype=np.float64)
    state1 = float(initial_state1)
    state2 = float(initial_state2)
    previous_output = 1.0 if float(initial_output) >= 0.5 else 0.0
    out = np.empty(unit_values.size, dtype=np.float64)
    for idx, value in enumerate(unit_values):
        if value <= 0.0:
            out[idx] = 0.0
            state1 = 0.0
            state2 = 0.0
            previous_output = 0.0
            continue
        if value >= 1.0:
            out[idx] = 1.0
            state1 = 0.0
            state2 = 0.0
            previous_output = 1.0
            continue

        state1 += float(value) - previous_output
        state2 += state1 - previous_output
        if state2 > 0.0:
            out[idx] = 1.0
        else:
            out[idx] = 0.0
        previous_output = out[idx]
    return out
