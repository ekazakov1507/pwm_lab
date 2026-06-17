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

PwmConfig = pwm_lab.PwmConfig
pwm_kind2_bipolar_fifo_grouped_multichannel = pwm_lab.pwm_kind2_bipolar_fifo_grouped_multichannel
pwm_kind2_fifo_grouped_multichannel = pwm_lab.pwm_kind2_fifo_grouped_multichannel
pwm_kind2_multichannel_latched = pwm_lab.pwm_kind2_multichannel_latched
pwm_kind2_phase_interleaved = pwm_lab.pwm_kind2_phase_interleaved
pwm_kind2_same_phase_parallel = pwm_lab.pwm_kind2_same_phase_parallel


def assert_close_array(actual: np.ndarray, expected: np.ndarray, tolerance: float = 1e-12) -> None:
    if not np.allclose(actual, expected, atol=tolerance, rtol=0.0):
        raise AssertionError(f"expected {expected}, got {actual}")


def main() -> None:
    config = PwmConfig(f_clk=16.0, f_pwm=1.0, resolution_bits=8)
    samples = np.array([0.25, 0.75, 0.5, 0.0, 1.0, 0.25], dtype=np.float64)

    grouped_as_latched = pwm_kind2_fifo_grouped_multichannel(
        samples,
        config,
        samples_per_period=1,
        channels=4,
    )
    latched = pwm_kind2_multichannel_latched(samples, config, channels=4)
    assert_close_array(grouped_as_latched, latched)

    grouped_as_interleaved = pwm_kind2_fifo_grouped_multichannel(
        samples,
        config,
        samples_per_period=4,
        channels=4,
    )
    interleaved = pwm_kind2_phase_interleaved(samples, config, channels=4)
    assert_close_array(grouped_as_interleaved, interleaved)

    same_phase_grouped = pwm_kind2_fifo_grouped_multichannel(
        samples,
        config,
        samples_per_period=2,
        channels=4,
        phase_offsets=np.zeros(4),
    )
    same_phase_reference = pwm_kind2_same_phase_parallel(samples, config, channels=2)
    assert_close_array(same_phase_grouped, same_phase_reference)

    raw = pwm_kind2_fifo_grouped_multichannel(
        np.array([0.25, 0.75]),
        config,
        samples_per_period=2,
        channels=4,
        normalize_sum=False,
    )
    if raw.size != config.period_samples:
        raise AssertionError("grouped FIFO PWM output must have one PWM period per sample group")
    raw_values = set(float(value) for value in np.unique(raw))
    allowed = {0.0, 1.0, 2.0, 3.0, 4.0}
    if not raw_values.issubset(allowed):
        raise AssertionError(f"raw summed PWM levels must be in {allowed}, got {raw_values}")

    bipolar = pwm_kind2_bipolar_fifo_grouped_multichannel(
        np.array([-0.5, 0.25, 0.75, -0.25]),
        config,
        samples_per_period=2,
        channels=4,
    )
    if bipolar.positive.shape != bipolar.negative.shape:
        raise AssertionError("bipolar grouped FIFO PWM branches must have matching shapes")

    try:
        pwm_kind2_fifo_grouped_multichannel(samples, config, samples_per_period=3, channels=4)
    except ValueError:
        pass
    else:
        raise AssertionError("channels must be a multiple of samples_per_period")

    print("PWM checks passed")


if __name__ == "__main__":
    main()
