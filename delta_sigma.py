from __future__ import annotations

import numpy as np

from ._modulation import (
    BipolarOutput,
    as_signed_values as _as_signed_values,
    as_unit_values as _as_unit_values,
    one_bit_first_order as _one_bit_first_order,
    one_bit_output_states as _output_states,
    one_bit_second_order as _one_bit_second_order,
    split_signed_magnitude as _split_signed_magnitude,
    staggered_states as _initial_states,
    state_array as _state_array,
)


class BipolarDeltaSigma(BipolarOutput):
    """Two-output delta-sigma representation for a signed signal."""


def delta_sigma_first_order(
    samples: np.ndarray,
    *,
    input_bits: int | None = None,
    initial_state: float = 0.0,
) -> np.ndarray:
    """First-order one-bit delta-sigma modulation for values in [0, 1].

    The modulator emits one ``0`` or ``1`` sample for each input sample. The
    accumulator keeps the quantization error, so the long-term output density
    approaches the input value.
    """
    values = _as_unit_values(samples, input_bits)
    return _one_bit_first_order(values, initial_state=initial_state)


def delta_sigma_second_order(
    samples: np.ndarray,
    *,
    input_bits: int | None = None,
    initial_state1: float = 0.0,
    initial_state2: float = 0.0,
    initial_output: float = 0.0,
) -> np.ndarray:
    """Second-order one-bit delta-sigma modulation for values in [0, 1].

    This is a two-integrator single-bit error-feedback model. Exact 0 and 1
    inputs are saturated to deterministic all-zero/all-one output and reset
    the loop state.
    """
    values = _as_unit_values(samples, input_bits)
    return _one_bit_second_order(
        values,
        initial_state1=initial_state1,
        initial_state2=initial_state2,
        initial_output=initial_output,
    )


def delta_sigma_first_order_signed(
    samples: np.ndarray,
    *,
    initial_state: float = 0.0,
) -> np.ndarray:
    """First-order one-bit delta-sigma modulation for signed values in [-1, 1].

    The returned stream uses the common signed one-bit representation ``-1``
    and ``+1``. Its average approaches the signed input value.
    """
    values = _as_signed_values(samples)
    unit_values = 0.5 * (values + 1.0)
    bits = delta_sigma_first_order(unit_values, initial_state=initial_state)
    return 2.0 * bits - 1.0


def delta_sigma_second_order_signed(
    samples: np.ndarray,
    *,
    initial_state1: float = 0.0,
    initial_state2: float = 0.0,
    initial_output: float = 0.0,
) -> np.ndarray:
    """Second-order one-bit delta-sigma modulation for signed values in [-1, 1]."""
    values = _as_signed_values(samples)
    unit_values = 0.5 * (values + 1.0)
    bits = delta_sigma_second_order(
        unit_values,
        initial_state1=initial_state1,
        initial_state2=initial_state2,
        initial_output=initial_output,
    )
    return 2.0 * bits - 1.0


def delta_sigma_first_order_multichannel(
    samples: np.ndarray,
    channels: int,
    *,
    input_bits: int | None = None,
    initial_states: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """First-order delta-sigma summed across staggered one-bit channels."""
    values = _as_unit_values(samples, input_bits)
    states = _initial_states(channels, initial_states)
    summed = np.zeros(values.size, dtype=np.float64)
    for state in states:
        summed += delta_sigma_first_order(values, initial_state=float(state))
    if normalize_sum:
        summed /= float(channels)
    return summed


def delta_sigma_second_order_multichannel(
    samples: np.ndarray,
    channels: int,
    *,
    input_bits: int | None = None,
    initial_state1s: np.ndarray | None = None,
    initial_state2s: np.ndarray | None = None,
    initial_outputs: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """Second-order delta-sigma summed across staggered one-bit channels."""
    values = _as_unit_values(samples, input_bits)
    state1s = _state_array(channels, initial_state1s, name="initial_state1s", flatten=True)
    state2s = _initial_states(channels, initial_state2s)
    outputs = _output_states(channels, initial_outputs, name="initial_outputs")
    summed = np.zeros(values.size, dtype=np.float64)
    for state1, state2, output in zip(state1s, state2s, outputs, strict=True):
        summed += delta_sigma_second_order(
            values,
            initial_state1=float(state1),
            initial_state2=float(state2),
            initial_output=float(output),
        )
    if normalize_sum:
        summed /= float(channels)
    return summed


def delta_sigma_first_order_signed_multichannel(
    samples: np.ndarray,
    channels: int,
    *,
    initial_states: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """Signed first-order delta-sigma summed across staggered one-bit channels."""
    values = _as_signed_values(samples)
    unit_values = 0.5 * (values + 1.0)
    bits = delta_sigma_first_order_multichannel(
        unit_values,
        channels,
        initial_states=initial_states,
        normalize_sum=normalize_sum,
    )
    if normalize_sum:
        return 2.0 * bits - 1.0
    return 2.0 * bits - float(channels)


def delta_sigma_second_order_signed_multichannel(
    samples: np.ndarray,
    channels: int,
    *,
    initial_state1s: np.ndarray | None = None,
    initial_state2s: np.ndarray | None = None,
    initial_outputs: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """Signed second-order delta-sigma summed across staggered one-bit channels."""
    values = _as_signed_values(samples)
    unit_values = 0.5 * (values + 1.0)
    bits = delta_sigma_second_order_multichannel(
        unit_values,
        channels,
        initial_state1s=initial_state1s,
        initial_state2s=initial_state2s,
        initial_outputs=initial_outputs,
        normalize_sum=normalize_sum,
    )
    if normalize_sum:
        return 2.0 * bits - 1.0
    return 2.0 * bits - float(channels)


def delta_sigma_first_order_bipolar(
    samples: np.ndarray,
    *,
    initial_state: float = 0.0,
) -> BipolarDeltaSigma:
    """Bipolar first-order delta-sigma for signed values in [-1, 1]."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarDeltaSigma(
        positive=delta_sigma_first_order(positive, initial_state=initial_state),
        negative=delta_sigma_first_order(negative, initial_state=initial_state),
    )


def delta_sigma_second_order_bipolar(
    samples: np.ndarray,
    *,
    initial_state1: float = 0.0,
    initial_state2: float = 0.0,
    initial_output: float = 0.0,
) -> BipolarDeltaSigma:
    """Bipolar second-order delta-sigma for signed values in [-1, 1]."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarDeltaSigma(
        positive=delta_sigma_second_order(
            positive,
            initial_state1=initial_state1,
            initial_state2=initial_state2,
            initial_output=initial_output,
        ),
        negative=delta_sigma_second_order(
            negative,
            initial_state1=initial_state1,
            initial_state2=initial_state2,
            initial_output=initial_output,
        ),
    )


def delta_sigma_first_order_bipolar_multichannel(
    samples: np.ndarray,
    channels: int,
    *,
    initial_states: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> BipolarDeltaSigma:
    """Bipolar first-order delta-sigma summed across staggered channels."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarDeltaSigma(
        positive=delta_sigma_first_order_multichannel(
            positive,
            channels,
            initial_states=initial_states,
            normalize_sum=normalize_sum,
        ),
        negative=delta_sigma_first_order_multichannel(
            negative,
            channels,
            initial_states=initial_states,
            normalize_sum=normalize_sum,
        ),
    )


def delta_sigma_second_order_bipolar_multichannel(
    samples: np.ndarray,
    channels: int,
    *,
    initial_state1s: np.ndarray | None = None,
    initial_state2s: np.ndarray | None = None,
    initial_outputs: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> BipolarDeltaSigma:
    """Bipolar second-order delta-sigma summed across staggered channels."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarDeltaSigma(
        positive=delta_sigma_second_order_multichannel(
            positive,
            channels,
            initial_state1s=initial_state1s,
            initial_state2s=initial_state2s,
            initial_outputs=initial_outputs,
            normalize_sum=normalize_sum,
        ),
        negative=delta_sigma_second_order_multichannel(
            negative,
            channels,
            initial_state1s=initial_state1s,
            initial_state2s=initial_state2s,
            initial_outputs=initial_outputs,
            normalize_sum=normalize_sum,
        ),
    )


def delta_sigma_first_order_fifo_parallel(
    samples: np.ndarray,
    channels: int,
    *,
    input_bits: int | None = None,
    initial_states: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """Process consecutive FIFO samples through parallel first-order channels.

    Each output sample combines one group of ``channels`` input samples. This is
    a grouped buffer-processing model, so the output length is truncated to
    ``len(samples) // channels``.
    """
    values = _as_unit_values(samples, input_bits).reshape(-1)
    states = _initial_states(channels, initial_states)
    groups = values.size // channels
    if groups == 0:
        return np.array([], dtype=np.float64)
    grouped = values[: groups * channels].reshape(groups, channels)
    summed = np.zeros(groups, dtype=np.float64)
    for ch in range(channels):
        summed += delta_sigma_first_order(grouped[:, ch], initial_state=float(states[ch]))
    if normalize_sum:
        summed /= float(channels)
    return summed


def delta_sigma_second_order_fifo_parallel(
    samples: np.ndarray,
    channels: int,
    *,
    input_bits: int | None = None,
    initial_state1s: np.ndarray | None = None,
    initial_state2s: np.ndarray | None = None,
    initial_outputs: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """Process consecutive FIFO samples through parallel second-order channels."""
    values = _as_unit_values(samples, input_bits).reshape(-1)
    state1s = _state_array(channels, initial_state1s, name="initial_state1s", flatten=True)
    state2s = _initial_states(channels, initial_state2s)
    outputs = _output_states(channels, initial_outputs, name="initial_outputs")
    groups = values.size // channels
    if groups == 0:
        return np.array([], dtype=np.float64)
    grouped = values[: groups * channels].reshape(groups, channels)
    summed = np.zeros(groups, dtype=np.float64)
    for ch in range(channels):
        summed += delta_sigma_second_order(
            grouped[:, ch],
            initial_state1=float(state1s[ch]),
            initial_state2=float(state2s[ch]),
            initial_output=float(outputs[ch]),
        )
    if normalize_sum:
        summed /= float(channels)
    return summed


def delta_sigma_first_order_signed_fifo_parallel(
    samples: np.ndarray,
    channels: int,
    *,
    initial_states: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """Signed first-order delta-sigma with grouped FIFO parallel channels."""
    values = _as_signed_values(samples).reshape(-1)
    unit_values = 0.5 * (values + 1.0)
    bits = delta_sigma_first_order_fifo_parallel(
        unit_values,
        channels,
        initial_states=initial_states,
        normalize_sum=normalize_sum,
    )
    if normalize_sum:
        return 2.0 * bits - 1.0
    return 2.0 * bits - float(channels)


def delta_sigma_second_order_signed_fifo_parallel(
    samples: np.ndarray,
    channels: int,
    *,
    initial_state1s: np.ndarray | None = None,
    initial_state2s: np.ndarray | None = None,
    initial_outputs: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """Signed second-order delta-sigma with grouped FIFO parallel channels."""
    values = _as_signed_values(samples).reshape(-1)
    unit_values = 0.5 * (values + 1.0)
    bits = delta_sigma_second_order_fifo_parallel(
        unit_values,
        channels,
        initial_state1s=initial_state1s,
        initial_state2s=initial_state2s,
        initial_outputs=initial_outputs,
        normalize_sum=normalize_sum,
    )
    if normalize_sum:
        return 2.0 * bits - 1.0
    return 2.0 * bits - float(channels)


def delta_sigma_first_order_bipolar_fifo_parallel(
    samples: np.ndarray,
    channels: int,
    *,
    initial_states: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> BipolarDeltaSigma:
    """Bipolar first-order delta-sigma with grouped FIFO parallel channels."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarDeltaSigma(
        positive=delta_sigma_first_order_fifo_parallel(
            positive,
            channels,
            initial_states=initial_states,
            normalize_sum=normalize_sum,
        ),
        negative=delta_sigma_first_order_fifo_parallel(
            negative,
            channels,
            initial_states=initial_states,
            normalize_sum=normalize_sum,
        ),
    )


def delta_sigma_second_order_bipolar_fifo_parallel(
    samples: np.ndarray,
    channels: int,
    *,
    initial_state1s: np.ndarray | None = None,
    initial_state2s: np.ndarray | None = None,
    initial_outputs: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> BipolarDeltaSigma:
    """Bipolar second-order delta-sigma with grouped FIFO parallel channels."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarDeltaSigma(
        positive=delta_sigma_second_order_fifo_parallel(
            positive,
            channels,
            initial_state1s=initial_state1s,
            initial_state2s=initial_state2s,
            initial_outputs=initial_outputs,
            normalize_sum=normalize_sum,
        ),
        negative=delta_sigma_second_order_fifo_parallel(
            negative,
            channels,
            initial_state1s=initial_state1s,
            initial_state2s=initial_state2s,
            initial_outputs=initial_outputs,
            normalize_sum=normalize_sum,
        ),
    )


def delta_sigma_first_order_fifo_round_robin(
    samples: np.ndarray,
    channels: int,
    *,
    input_bits: int | None = None,
    initial_states: np.ndarray | None = None,
) -> np.ndarray:
    """Process FIFO samples in order while rotating first-order channel state."""
    values = _as_unit_values(samples, input_bits).reshape(-1)
    states = _initial_states(channels, initial_states)
    out = np.empty(values.size, dtype=np.float64)
    for idx, value in enumerate(values):
        ch = idx % channels
        states[ch] += float(value)
        if states[ch] >= 1.0:
            out[idx] = 1.0
            states[ch] -= 1.0
        else:
            out[idx] = 0.0
    return out


def delta_sigma_second_order_fifo_round_robin(
    samples: np.ndarray,
    channels: int,
    *,
    input_bits: int | None = None,
    initial_state1s: np.ndarray | None = None,
    initial_state2s: np.ndarray | None = None,
    initial_outputs: np.ndarray | None = None,
) -> np.ndarray:
    """Process FIFO samples in order while rotating second-order channel state."""
    values = _as_unit_values(samples, input_bits).reshape(-1)
    state1s = _state_array(channels, initial_state1s, name="initial_state1s", flatten=True)
    state2s = _initial_states(channels, initial_state2s)
    outputs = _output_states(channels, initial_outputs, name="initial_outputs")
    out = np.empty(values.size, dtype=np.float64)
    for idx, value in enumerate(values):
        ch = idx % channels
        if value <= 0.0:
            out[idx] = 0.0
            state1s[ch] = 0.0
            state2s[ch] = 0.0
            outputs[ch] = 0.0
            continue
        if value >= 1.0:
            out[idx] = 1.0
            state1s[ch] = 0.0
            state2s[ch] = 0.0
            outputs[ch] = 1.0
            continue

        state1s[ch] += float(value) - outputs[ch]
        state2s[ch] += state1s[ch] - outputs[ch]
        if state2s[ch] > 0.0:
            out[idx] = 1.0
        else:
            out[idx] = 0.0
        outputs[ch] = out[idx]
    return out


def delta_sigma_first_order_signed_fifo_round_robin(
    samples: np.ndarray,
    channels: int,
    *,
    initial_states: np.ndarray | None = None,
) -> np.ndarray:
    """Signed first-order delta-sigma with FIFO round-robin channel state."""
    values = _as_signed_values(samples).reshape(-1)
    unit_values = 0.5 * (values + 1.0)
    bits = delta_sigma_first_order_fifo_round_robin(
        unit_values,
        channels,
        initial_states=initial_states,
    )
    return 2.0 * bits - 1.0


def delta_sigma_second_order_signed_fifo_round_robin(
    samples: np.ndarray,
    channels: int,
    *,
    initial_state1s: np.ndarray | None = None,
    initial_state2s: np.ndarray | None = None,
    initial_outputs: np.ndarray | None = None,
) -> np.ndarray:
    """Signed second-order delta-sigma with FIFO round-robin channel state."""
    values = _as_signed_values(samples).reshape(-1)
    unit_values = 0.5 * (values + 1.0)
    bits = delta_sigma_second_order_fifo_round_robin(
        unit_values,
        channels,
        initial_state1s=initial_state1s,
        initial_state2s=initial_state2s,
        initial_outputs=initial_outputs,
    )
    return 2.0 * bits - 1.0


def delta_sigma_first_order_bipolar_fifo_round_robin(
    samples: np.ndarray,
    channels: int,
    *,
    initial_states: np.ndarray | None = None,
) -> BipolarDeltaSigma:
    """Bipolar first-order delta-sigma with FIFO round-robin channel state."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarDeltaSigma(
        positive=delta_sigma_first_order_fifo_round_robin(
            positive,
            channels,
            initial_states=initial_states,
        ),
        negative=delta_sigma_first_order_fifo_round_robin(
            negative,
            channels,
            initial_states=initial_states,
        ),
    )


def delta_sigma_second_order_bipolar_fifo_round_robin(
    samples: np.ndarray,
    channels: int,
    *,
    initial_state1s: np.ndarray | None = None,
    initial_state2s: np.ndarray | None = None,
    initial_outputs: np.ndarray | None = None,
) -> BipolarDeltaSigma:
    """Bipolar second-order delta-sigma with FIFO round-robin channel state."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarDeltaSigma(
        positive=delta_sigma_second_order_fifo_round_robin(
            positive,
            channels,
            initial_state1s=initial_state1s,
            initial_state2s=initial_state2s,
            initial_outputs=initial_outputs,
        ),
        negative=delta_sigma_second_order_fifo_round_robin(
            negative,
            channels,
            initial_state1s=initial_state1s,
            initial_state2s=initial_state2s,
            initial_outputs=initial_outputs,
        ),
    )
