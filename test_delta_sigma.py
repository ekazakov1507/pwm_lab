from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "pwm_lab",
    PACKAGE_DIR / "__init__.py",
    submodule_search_locations=[str(PACKAGE_DIR)],
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load pwm_lab package")

pwm_lab = importlib.util.module_from_spec(SPEC)
sys.modules["pwm_lab"] = pwm_lab
SPEC.loader.exec_module(pwm_lab)

delta_sigma_first_order = pwm_lab.delta_sigma_first_order
delta_sigma_first_order_bipolar = pwm_lab.delta_sigma_first_order_bipolar
delta_sigma_first_order_bipolar_fifo_parallel = pwm_lab.delta_sigma_first_order_bipolar_fifo_parallel
delta_sigma_first_order_bipolar_fifo_round_robin = pwm_lab.delta_sigma_first_order_bipolar_fifo_round_robin
delta_sigma_first_order_bipolar_multichannel = pwm_lab.delta_sigma_first_order_bipolar_multichannel
delta_sigma_first_order_fifo_parallel = pwm_lab.delta_sigma_first_order_fifo_parallel
delta_sigma_first_order_fifo_round_robin = pwm_lab.delta_sigma_first_order_fifo_round_robin
delta_sigma_first_order_multichannel = pwm_lab.delta_sigma_first_order_multichannel
delta_sigma_first_order_signed = pwm_lab.delta_sigma_first_order_signed
delta_sigma_first_order_signed_fifo_parallel = pwm_lab.delta_sigma_first_order_signed_fifo_parallel
delta_sigma_first_order_signed_fifo_round_robin = pwm_lab.delta_sigma_first_order_signed_fifo_round_robin
delta_sigma_first_order_signed_multichannel = pwm_lab.delta_sigma_first_order_signed_multichannel
delta_sigma_second_order = pwm_lab.delta_sigma_second_order
delta_sigma_second_order_bipolar = pwm_lab.delta_sigma_second_order_bipolar
delta_sigma_second_order_bipolar_fifo_parallel = pwm_lab.delta_sigma_second_order_bipolar_fifo_parallel
delta_sigma_second_order_bipolar_fifo_round_robin = pwm_lab.delta_sigma_second_order_bipolar_fifo_round_robin
delta_sigma_second_order_bipolar_multichannel = pwm_lab.delta_sigma_second_order_bipolar_multichannel
delta_sigma_second_order_fifo_parallel = pwm_lab.delta_sigma_second_order_fifo_parallel
delta_sigma_second_order_fifo_round_robin = pwm_lab.delta_sigma_second_order_fifo_round_robin
delta_sigma_second_order_multichannel = pwm_lab.delta_sigma_second_order_multichannel
delta_sigma_second_order_signed = pwm_lab.delta_sigma_second_order_signed
delta_sigma_second_order_signed_fifo_parallel = pwm_lab.delta_sigma_second_order_signed_fifo_parallel
delta_sigma_second_order_signed_fifo_round_robin = pwm_lab.delta_sigma_second_order_signed_fifo_round_robin
delta_sigma_second_order_signed_multichannel = pwm_lab.delta_sigma_second_order_signed_multichannel
delta_sigma_spectra_for_samples = pwm_lab.delta_sigma_spectra_for_samples
pdm_first_order = pwm_lab.pdm_first_order
pdm_first_order_multichannel = pwm_lab.pdm_first_order_multichannel
pdm_second_order = pwm_lab.pdm_second_order
pdm_second_order_multichannel = pwm_lab.pdm_second_order_multichannel
sine_samples = pwm_lab.sine_samples


def assert_close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected}, got {actual}")


def assert_values_in(actual: np.ndarray, allowed: set[float]) -> None:
    values = set(float(value) for value in np.unique(actual))
    if not values.issubset(allowed):
        raise AssertionError(f"expected only {allowed}, got {values}")


def main() -> None:
    zeros = delta_sigma_first_order(np.zeros(32))
    ones = delta_sigma_first_order(np.ones(32))
    if np.any(zeros):
        raise AssertionError("zero input must produce all-zero delta-sigma")
    if not np.all(ones == 1.0):
        raise AssertionError("one input must produce all-one delta-sigma")
    assert_values_in(zeros, {0.0})
    assert_values_in(ones, {1.0})

    quarter = delta_sigma_first_order(np.full(400, 0.25))
    half = delta_sigma_first_order(np.full(400, 0.5))
    three_quarters = delta_sigma_first_order(np.full(400, 0.75))
    assert_close(float(quarter.mean()), 0.25)
    assert_close(float(half.mean()), 0.5)
    assert_close(float(three_quarters.mean()), 0.75)
    assert_values_in(three_quarters, {0.0, 1.0})

    comparison = np.linspace(0.0, 1.0, 257)
    if not np.array_equal(delta_sigma_first_order(comparison), pdm_first_order(comparison)):
        raise AssertionError("first-order delta-sigma must match first-order PDM for unipolar input")
    if not np.array_equal(delta_sigma_second_order(comparison), pdm_second_order(comparison)):
        raise AssertionError("second-order delta-sigma must match second-order PDM for unipolar input")
    if not np.array_equal(
        delta_sigma_first_order_multichannel(comparison, channels=4),
        pdm_first_order_multichannel(comparison, channels=4),
    ):
        raise AssertionError("first-order multichannel delta-sigma must match PDM for unipolar input")
    if not np.array_equal(
        delta_sigma_second_order_multichannel(comparison, channels=4),
        pdm_second_order_multichannel(comparison, channels=4),
    ):
        raise AssertionError("second-order multichannel delta-sigma must match PDM for unipolar input")

    second_order_zeros = delta_sigma_second_order(np.zeros(32))
    second_order_ones = delta_sigma_second_order(np.ones(32))
    if np.any(second_order_zeros):
        raise AssertionError("zero input must produce all-zero second-order delta-sigma")
    if not np.all(second_order_ones == 1.0):
        raise AssertionError("one input must produce all-one second-order delta-sigma")

    second_order_quarter = delta_sigma_second_order(np.full(400, 0.25))
    second_order_half = delta_sigma_second_order(np.full(400, 0.5))
    assert_close(float(second_order_quarter.mean()), 0.25)
    assert_close(float(second_order_half.mean()), 0.5)
    assert_values_in(second_order_quarter, {0.0, 1.0})

    signed_negative_full = delta_sigma_first_order_signed(np.full(32, -1.0))
    signed_positive_full = delta_sigma_first_order_signed(np.full(32, 1.0))
    if not np.all(signed_negative_full == -1.0):
        raise AssertionError("-1 signed input must produce all -1 output")
    if not np.all(signed_positive_full == 1.0):
        raise AssertionError("+1 signed input must produce all +1 output")

    signed_zero = delta_sigma_first_order_signed(np.zeros(400))
    signed_positive = delta_sigma_first_order_signed(np.full(400, 0.4))
    signed_negative = delta_sigma_first_order_signed(np.full(400, -0.4))
    assert_close(float(signed_zero.mean()), 0.0)
    assert_close(float(signed_positive.mean()), 0.4, tolerance=1e-2)
    assert_close(float(signed_negative.mean()), -0.4, tolerance=1e-2)
    assert_values_in(signed_positive, {-1.0, 1.0})

    signed_second_positive = delta_sigma_second_order_signed(np.full(400, 0.4))
    signed_second_negative = delta_sigma_second_order_signed(np.full(400, -0.4))
    assert_close(float(signed_second_positive.mean()), 0.4, tolerance=1e-2)
    assert_close(float(signed_second_negative.mean()), -0.4, tolerance=1e-2)
    assert_values_in(signed_second_negative, {-1.0, 1.0})

    half_two_channel = delta_sigma_first_order_multichannel(np.full(32, 0.5), channels=2)
    if not np.allclose(half_two_channel, 0.5):
        raise AssertionError("staggered two-channel delta-sigma should average to 0.5 per sample")

    second_order_multi = delta_sigma_second_order_multichannel(np.full(400, 0.5), channels=4)
    assert_close(float(second_order_multi.mean()), 0.5)

    signed_zero_two_channel = delta_sigma_first_order_signed_multichannel(np.zeros(32), channels=2)
    if not np.allclose(signed_zero_two_channel, 0.0):
        raise AssertionError("staggered two-channel signed delta-sigma should average to 0 per sample")

    signed_second_zero_two_channel = delta_sigma_second_order_signed_multichannel(np.zeros(400), channels=4)
    assert_close(float(signed_second_zero_two_channel.mean()), 0.0)

    bipolar_zero = delta_sigma_first_order_bipolar(np.zeros(32))
    if np.any(bipolar_zero.positive) or np.any(bipolar_zero.negative):
        raise AssertionError("zero bipolar input must keep both branches inactive")

    bipolar_positive = delta_sigma_first_order_bipolar(np.full(400, 0.4))
    bipolar_negative = delta_sigma_first_order_bipolar(np.full(400, -0.4))
    assert_close(float(bipolar_positive.differential.mean()), 0.4, tolerance=1e-2)
    assert_close(float(bipolar_negative.differential.mean()), -0.4, tolerance=1e-2)

    bipolar_second_positive = delta_sigma_second_order_bipolar(np.full(400, 0.4))
    bipolar_second_negative = delta_sigma_second_order_bipolar(np.full(400, -0.4))
    assert_close(float(bipolar_second_positive.differential.mean()), 0.4, tolerance=1e-2)
    assert_close(float(bipolar_second_negative.differential.mean()), -0.4, tolerance=1e-2)

    bipolar_multi = delta_sigma_first_order_bipolar_multichannel(np.full(32, -0.5), channels=2)
    if not np.allclose(bipolar_multi.differential, -0.5):
        raise AssertionError("staggered bipolar two-channel delta-sigma should average to -0.5 per sample")

    bipolar_second_multi = delta_sigma_second_order_bipolar_multichannel(np.full(400, -0.5), channels=4)
    assert_close(float(bipolar_second_multi.differential.mean()), -0.5)

    fifo_parallel = delta_sigma_first_order_fifo_parallel(np.full(40, 0.5), channels=4)
    if fifo_parallel.size != 10:
        raise AssertionError("FIFO parallel output length must be len(samples) // channels")
    assert_close(float(fifo_parallel.mean()), 0.5)

    second_fifo_parallel = delta_sigma_second_order_fifo_parallel(np.full(400, 0.5), channels=4)
    if second_fifo_parallel.size != 100:
        raise AssertionError("second-order FIFO parallel output length must be len(samples) // channels")
    assert_close(float(second_fifo_parallel.mean()), 0.5)

    signed_fifo_parallel = delta_sigma_first_order_signed_fifo_parallel(np.full(40, -0.5), channels=4)
    assert_close(float(signed_fifo_parallel.mean()), -0.5)

    signed_second_fifo_parallel = delta_sigma_second_order_signed_fifo_parallel(np.full(400, -0.5), channels=4)
    assert_close(float(signed_second_fifo_parallel.mean()), -0.5)

    bipolar_fifo_parallel = delta_sigma_first_order_bipolar_fifo_parallel(np.full(40, -0.5), channels=4)
    assert_close(float(bipolar_fifo_parallel.differential.mean()), -0.5)

    bipolar_second_fifo_parallel = delta_sigma_second_order_bipolar_fifo_parallel(np.full(400, -0.5), channels=4)
    assert_close(float(bipolar_second_fifo_parallel.differential.mean()), -0.5)

    fifo_round_robin = delta_sigma_first_order_fifo_round_robin(np.full(40, 0.5), channels=4)
    if fifo_round_robin.size != 40:
        raise AssertionError("FIFO round-robin output length must match input length")
    assert_close(float(fifo_round_robin.mean()), 0.5)

    second_fifo_round_robin = delta_sigma_second_order_fifo_round_robin(np.full(400, 0.5), channels=4)
    if second_fifo_round_robin.size != 400:
        raise AssertionError("second-order FIFO round-robin output length must match input length")
    assert_close(float(second_fifo_round_robin.mean()), 0.5)

    signed_fifo_round_robin = delta_sigma_first_order_signed_fifo_round_robin(np.full(40, -0.5), channels=4)
    assert_close(float(signed_fifo_round_robin.mean()), -0.5)

    signed_second_fifo_round_robin = delta_sigma_second_order_signed_fifo_round_robin(np.full(400, -0.5), channels=4)
    assert_close(float(signed_second_fifo_round_robin.mean()), -0.5)

    bipolar_fifo_round_robin = delta_sigma_first_order_bipolar_fifo_round_robin(np.full(40, -0.5), channels=4)
    assert_close(float(bipolar_fifo_round_robin.differential.mean()), -0.5)

    bipolar_second_fifo_round_robin = delta_sigma_second_order_bipolar_fifo_round_robin(np.full(400, -0.5), channels=4)
    assert_close(float(bipolar_second_fifo_round_robin.differential.mean()), -0.5)

    try:
        delta_sigma_first_order_multichannel(np.full(4, 0.5), channels=0)
    except ValueError:
        pass
    else:
        raise AssertionError("channels=0 must be rejected")

    try:
        delta_sigma_first_order_multichannel(np.full(4, 0.5), channels=2, initial_states=np.array([0.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("initial_states length mismatch must be rejected")

    _, sine = sine_samples(freq=1000.0, sample_rate=100_000.0, n_samples=10_000)
    spectra = delta_sigma_spectra_for_samples(
        sine,
        100_000.0,
        peak_count=3,
        f_min=100.0,
        f_max=5_000.0,
        min_spacing_hz=100.0,
    )
    for name in ("input", "delta_sigma_first_order", "delta_sigma_second_order"):
        if name not in spectra:
            raise AssertionError(f"missing delta-sigma spectrum {name}")
        assert_close(float(spectra[name].sample_rate), 100_000.0)

    signed_spectra = delta_sigma_spectra_for_samples(
        np.full(400, -0.4),
        100_000.0,
        signed=True,
        peak_count=2,
    )
    for name in ("delta_sigma_first_order_signed", "delta_sigma_second_order_signed"):
        if name not in signed_spectra:
            raise AssertionError(f"missing signed delta-sigma spectrum {name}")

    bipolar_spectra = delta_sigma_spectra_for_samples(
        np.full(400, -0.4),
        100_000.0,
        bipolar=True,
        peak_count=2,
    )
    for name in (
        "delta_sigma_first_order_bipolar.positive",
        "delta_sigma_first_order_bipolar.negative",
        "delta_sigma_first_order_bipolar.differential",
        "delta_sigma_second_order_bipolar.positive",
        "delta_sigma_second_order_bipolar.negative",
        "delta_sigma_second_order_bipolar.differential",
    ):
        if name not in bipolar_spectra:
            raise AssertionError(f"missing bipolar delta-sigma spectrum {name}")

    fifo_parallel_spectra = delta_sigma_spectra_for_samples(
        np.full(400, 0.5),
        100_000.0,
        channels=4,
        strategy="fifo_parallel",
        peak_count=2,
    )
    for name in ("delta_sigma_first_order_fifo_parallel", "delta_sigma_second_order_fifo_parallel"):
        if name not in fifo_parallel_spectra:
            raise AssertionError(f"missing FIFO parallel delta-sigma spectrum {name}")
        assert_close(float(fifo_parallel_spectra[name].sample_rate), 25_000.0)

    fifo_round_robin_spectra = delta_sigma_spectra_for_samples(
        np.full(400, 0.5),
        100_000.0,
        channels=4,
        strategy="fifo_round_robin",
        peak_count=2,
    )
    for name in ("delta_sigma_first_order_fifo_round_robin", "delta_sigma_second_order_fifo_round_robin"):
        if name not in fifo_round_robin_spectra:
            raise AssertionError(f"missing FIFO round-robin delta-sigma spectrum {name}")
        assert_close(float(fifo_round_robin_spectra[name].sample_rate), 100_000.0)

    first_order_only = delta_sigma_spectra_for_samples(
        np.full(400, 0.5),
        100_000.0,
        orders=(1,),
        peak_count=2,
    )
    if "delta_sigma_first_order" not in first_order_only:
        raise AssertionError("missing first-order-only delta-sigma spectrum")
    if "delta_sigma_second_order" in first_order_only:
        raise AssertionError("orders=(1,) must not include second-order output")

    try:
        delta_sigma_spectra_for_samples(np.zeros(4), 100_000.0, signed=True, bipolar=True)
    except ValueError:
        pass
    else:
        raise AssertionError("signed and bipolar spectra modes must be mutually exclusive")

    try:
        delta_sigma_spectra_for_samples(np.zeros(4), 100_000.0, strategy="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown delta-sigma strategy must be rejected")

    try:
        delta_sigma_spectra_for_samples(np.zeros(4), 100_000.0, orders=(3,))
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported delta-sigma order must be rejected")

    print("Delta-sigma checks passed")


if __name__ == "__main__":
    main()
