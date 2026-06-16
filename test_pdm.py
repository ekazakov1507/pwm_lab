from __future__ import annotations

import sys
import importlib.util
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

pdm_first_order = pwm_lab.pdm_first_order
pdm_first_order_bipolar = pwm_lab.pdm_first_order_bipolar
pdm_first_order_bipolar_multichannel = pwm_lab.pdm_first_order_bipolar_multichannel
pdm_first_order_multichannel = pwm_lab.pdm_first_order_multichannel
pdm_second_order = pwm_lab.pdm_second_order
pdm_second_order_bipolar = pwm_lab.pdm_second_order_bipolar
pdm_second_order_bipolar_multichannel = pwm_lab.pdm_second_order_bipolar_multichannel
pdm_second_order_multichannel = pwm_lab.pdm_second_order_multichannel
pdm_spectra_for_samples = pwm_lab.pdm_spectra_for_samples
sine_samples = pwm_lab.sine_samples


def assert_close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected}, got {actual}")


def main() -> None:
    zeros = pdm_first_order(np.zeros(32))
    ones = pdm_first_order(np.ones(32))
    if np.any(zeros):
        raise AssertionError("zero input must produce all-zero PDM")
    if not np.all(ones == 1.0):
        raise AssertionError("one input must produce all-one PDM")

    quarter = pdm_first_order(np.full(400, 0.25))
    assert_close(float(quarter.mean()), 0.25)

    second_order_zeros = pdm_second_order(np.zeros(32))
    second_order_ones = pdm_second_order(np.ones(32))
    if np.any(second_order_zeros):
        raise AssertionError("zero input must produce all-zero second-order PDM")
    if not np.all(second_order_ones == 1.0):
        raise AssertionError("one input must produce all-one second-order PDM")

    second_order_quarter = pdm_second_order(np.full(400, 0.25))
    second_order_half = pdm_second_order(np.full(400, 0.5))
    assert_close(float(second_order_quarter.mean()), 0.25)
    assert_close(float(second_order_half.mean()), 0.5)

    half_two_channel = pdm_first_order_multichannel(np.full(32, 0.5), channels=2)
    if not np.allclose(half_two_channel, 0.5):
        raise AssertionError("staggered two-channel PDM should average to 0.5 per sample")

    second_order_multi = pdm_second_order_multichannel(np.full(400, 0.5), channels=4)
    assert_close(float(second_order_multi.mean()), 0.5)

    bipolar_positive = pdm_first_order_bipolar(np.full(400, 0.4))
    bipolar_negative = pdm_first_order_bipolar(np.full(400, -0.4))
    assert_close(float(bipolar_positive.differential.mean()), 0.4)
    assert_close(float(bipolar_negative.differential.mean()), -0.4)

    bipolar_second_positive = pdm_second_order_bipolar(np.full(400, 0.4))
    bipolar_second_negative = pdm_second_order_bipolar(np.full(400, -0.4))
    assert_close(float(bipolar_second_positive.differential.mean()), 0.4)
    assert_close(float(bipolar_second_negative.differential.mean()), -0.4)

    bipolar_multi = pdm_first_order_bipolar_multichannel(np.full(32, -0.5), channels=2)
    if not np.allclose(bipolar_multi.differential, -0.5):
        raise AssertionError("staggered bipolar two-channel PDM should average to -0.5 per sample")

    bipolar_second_multi = pdm_second_order_bipolar_multichannel(np.full(400, -0.5), channels=4)
    assert_close(float(bipolar_second_multi.differential.mean()), -0.5)

    _, sine = sine_samples(freq=1000.0, sample_rate=100_000.0, n_samples=10_000)
    spectra = pdm_spectra_for_samples(
        sine,
        100_000.0,
        peak_count=3,
        f_min=100.0,
        f_max=5_000.0,
        min_spacing_hz=100.0,
    )
    for name in ("input", "pdm_first_order", "pdm_second_order"):
        if name not in spectra:
            raise AssertionError(f"missing PDM spectrum {name}")
        assert_close(float(spectra[name].sample_rate), 100_000.0)

    bipolar_spectra = pdm_spectra_for_samples(
        np.full(400, -0.4),
        100_000.0,
        bipolar=True,
        peak_count=2,
    )
    for name in (
        "pdm_first_order.positive",
        "pdm_first_order.negative",
        "pdm_first_order.differential",
        "pdm_second_order.positive",
        "pdm_second_order.negative",
        "pdm_second_order.differential",
    ):
        if name not in bipolar_spectra:
            raise AssertionError(f"missing bipolar PDM spectrum {name}")

    print("PDM checks passed")


if __name__ == "__main__":
    main()
