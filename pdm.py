from __future__ import annotations

import numpy as np

from ._modulation import (
    BipolarOutput,
    as_unit_values as _as_unit_values,
    one_bit_first_order as _one_bit_first_order,
    one_bit_second_order as _one_bit_second_order,
    split_signed_magnitude as _split_signed_magnitude,
    staggered_states as _initial_states,
    state_array as _state_array,
)


class BipolarPdm(BipolarOutput):
    """Two-output PDM representation for a signed signal."""


def pdm_first_order(
    samples: np.ndarray,
    *,
    input_bits: int | None = None,
    initial_state: float = 0.0,
) -> np.ndarray:
    """First-order pulse-density modulation for normalized values in [0, 1].

    The accumulator keeps the fractional pulse error between samples. For a
    constant input, the average output pulse density approaches the input value.
    """
    values = _as_unit_values(samples, input_bits)
    return _one_bit_first_order(values, initial_state=initial_state)


def pdm_second_order(
    samples: np.ndarray,
    *,
    input_bits: int | None = None,
    initial_state1: float = 0.0,
    initial_state2: float = 0.0,
    initial_output: float = 0.0,
) -> np.ndarray:
    """Second-order pulse-density modulation for normalized values in [0, 1].

    This is a two-integrator single-bit error-feedback model. Exact 0 and 1
    inputs are saturated to all-zero/all-one output and reset the loop state,
    which keeps edge cases deterministic for architectural experiments.
    """
    values = _as_unit_values(samples, input_bits)
    return _one_bit_second_order(
        values,
        initial_state1=initial_state1,
        initial_state2=initial_state2,
        initial_output=initial_output,
    )


def pdm_first_order_multichannel(
    samples: np.ndarray,
    channels: int,
    *,
    input_bits: int | None = None,
    initial_states: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """First-order PDM summed across channels with staggered accumulator states."""
    values = _as_unit_values(samples, input_bits)
    states = _initial_states(channels, initial_states)
    summed = np.zeros(values.size, dtype=np.float64)
    for state in states:
        summed += pdm_first_order(values, initial_state=float(state))
    if normalize_sum:
        summed /= float(channels)
    return summed


def pdm_second_order_multichannel(
    samples: np.ndarray,
    channels: int,
    *,
    input_bits: int | None = None,
    initial_state1s: np.ndarray | None = None,
    initial_state2s: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> np.ndarray:
    """Second-order PDM summed across channels with staggered second states."""
    values = _as_unit_values(samples, input_bits)
    state2s = _initial_states(channels, initial_state2s)
    state1s = _state_array(channels, initial_state1s, name="initial_state1s")
    summed = np.zeros(values.size, dtype=np.float64)
    for state1, state2 in zip(state1s, state2s, strict=True):
        summed += pdm_second_order(
            values,
            initial_state1=float(state1),
            initial_state2=float(state2),
        )
    if normalize_sum:
        summed /= float(channels)
    return summed


def pdm_first_order_bipolar(
    samples: np.ndarray,
    *,
    initial_state: float = 0.0,
) -> BipolarPdm:
    """Bipolar first-order PDM for signed values in [-1, 1]."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarPdm(
        positive=pdm_first_order(positive, initial_state=initial_state),
        negative=pdm_first_order(negative, initial_state=initial_state),
    )


def pdm_second_order_bipolar(
    samples: np.ndarray,
    *,
    initial_state1: float = 0.0,
    initial_state2: float = 0.0,
    initial_output: float = 0.0,
) -> BipolarPdm:
    """Bipolar second-order PDM for signed values in [-1, 1]."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarPdm(
        positive=pdm_second_order(
            positive,
            initial_state1=initial_state1,
            initial_state2=initial_state2,
            initial_output=initial_output,
        ),
        negative=pdm_second_order(
            negative,
            initial_state1=initial_state1,
            initial_state2=initial_state2,
            initial_output=initial_output,
        ),
    )


def pdm_first_order_bipolar_multichannel(
    samples: np.ndarray,
    channels: int,
    *,
    initial_states: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> BipolarPdm:
    """Bipolar first-order PDM summed across staggered channels."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarPdm(
        positive=pdm_first_order_multichannel(
            positive,
            channels,
            initial_states=initial_states,
            normalize_sum=normalize_sum,
        ),
        negative=pdm_first_order_multichannel(
            negative,
            channels,
            initial_states=initial_states,
            normalize_sum=normalize_sum,
        ),
    )


def pdm_second_order_bipolar_multichannel(
    samples: np.ndarray,
    channels: int,
    *,
    initial_state1s: np.ndarray | None = None,
    initial_state2s: np.ndarray | None = None,
    normalize_sum: bool = True,
) -> BipolarPdm:
    """Bipolar second-order PDM summed across staggered channels."""
    positive, negative = _split_signed_magnitude(samples)
    return BipolarPdm(
        positive=pdm_second_order_multichannel(
            positive,
            channels,
            initial_state1s=initial_state1s,
            initial_state2s=initial_state2s,
            normalize_sum=normalize_sum,
        ),
        negative=pdm_second_order_multichannel(
            negative,
            channels,
            initial_state1s=initial_state1s,
            initial_state2s=initial_state2s,
            normalize_sum=normalize_sum,
        ),
    )
