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
BridgePwm = pwm_lab.BridgePwm
pwm_kind1_bridge = pwm_lab.pwm_kind1_bridge
pwm_kind1_bridge_multichannel = pwm_lab.pwm_kind1_bridge_multichannel
pwm_kind2_bridge_fifo_grouped_multichannel = pwm_lab.pwm_kind2_bridge_fifo_grouped_multichannel
pwm_kind2_bridge_latched = pwm_lab.pwm_kind2_bridge_latched
pwm_kind2_bridge_multichannel_latched = pwm_lab.pwm_kind2_bridge_multichannel_latched
pwm_kind2_bridge_phase_interleaved = pwm_lab.pwm_kind2_bridge_phase_interleaved
pwm_kind2_bipolar_fifo_grouped_multichannel = pwm_lab.pwm_kind2_bipolar_fifo_grouped_multichannel
pwm_kind2_fifo_grouped_multichannel = pwm_lab.pwm_kind2_fifo_grouped_multichannel
pwm_kind2_multichannel_latched = pwm_lab.pwm_kind2_multichannel_latched
pwm_kind2_phase_interleaved = pwm_lab.pwm_kind2_phase_interleaved
pwm_kind2_same_phase_parallel = pwm_lab.pwm_kind2_same_phase_parallel


def assert_close_array(actual: np.ndarray, expected: np.ndarray, tolerance: float = 1e-12) -> None:
    if not np.allclose(actual, expected, atol=tolerance, rtol=0.0):
        raise AssertionError(f"expected {expected}, got {actual}")


def assert_bridge_close(actual: BridgePwm, expected: BridgePwm, tolerance: float = 1e-12) -> None:
    assert_close_array(actual.plus, expected.plus, tolerance)
    assert_close_array(actual.minus, expected.minus, tolerance)


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

    signed = np.array([0.1, -0.1, 0.0], dtype=np.float64)
    bridge_config = PwmConfig(f_clk=128.0, f_pwm=1.0, resolution_bits=8)
    bridge = pwm_kind2_bridge_latched(signed, bridge_config, mode="three_level")
    if not isinstance(bridge, BridgePwm):
        raise AssertionError("bridge PWM functions must return BridgePwm")
    assert_close_array(bridge.plus, bridge.positive)
    assert_close_array(bridge.minus, bridge.negative)
    assert_close_array(bridge.differential, bridge.plus - bridge.minus)

    regular = pwm_kind2_bridge_latched(np.array([-1.0, 0.0, 1.0]), bridge_config, mode="regular")
    regular_plus = regular.plus.reshape(3, bridge_config.period_samples)
    regular_minus = regular.minus.reshape(3, bridge_config.period_samples)
    if np.any(regular_plus + regular_minus != 1.0):
        raise AssertionError("regular bridge PWM must select exactly one output per clock")
    if regular_plus[0].mean() != 0.0 or regular_minus[0].mean() != 1.0:
        raise AssertionError("regular bridge PWM must map -1 to the minus output")
    if regular_plus[2].mean() != 1.0 or regular_minus[2].mean() != 0.0:
        raise AssertionError("regular bridge PWM must map +1 to the plus output")

    bipolar_bridge = pwm_kind2_bridge_latched(signed, bridge_config, mode="bipolar")
    bipolar_plus = bipolar_bridge.plus.reshape(3, bridge_config.period_samples).mean(axis=1)
    bipolar_minus = bipolar_bridge.minus.reshape(3, bridge_config.period_samples).mean(axis=1)
    if not (0.45 < bipolar_plus[0] < 0.65 and bipolar_minus[0] == 0.0):
        raise AssertionError("bipolar bridge PWM must gate small positive input to plus near 50% duty")
    if not (0.45 < bipolar_minus[1] < 0.65 and bipolar_plus[1] == 0.0):
        raise AssertionError("bipolar bridge PWM must gate small negative input to minus near 50% duty")
    if bipolar_plus[2] != 0.0 or bipolar_minus[2] != 0.0:
        raise AssertionError("bipolar bridge PWM must keep both outputs off at zero input")

    three_level = pwm_kind2_bridge_latched(signed, bridge_config, mode="three_level")
    three_plus = three_level.plus.reshape(3, bridge_config.period_samples).mean(axis=1)
    three_minus = three_level.minus.reshape(3, bridge_config.period_samples).mean(axis=1)
    if not (0.05 < three_plus[0] < 0.15 and three_minus[0] == 0.0):
        raise AssertionError("three-level bridge PWM must use abs(input) as plus duty")
    if not (0.05 < three_minus[1] < 0.15 and three_plus[1] == 0.0):
        raise AssertionError("three-level bridge PWM must use abs(input) as minus duty")
    if three_plus[2] != 0.0 or three_minus[2] != 0.0:
        raise AssertionError("three-level bridge PWM must keep both outputs off at zero input")

    kind1_bridge = pwm_kind1_bridge(np.full(bridge_config.period_samples, 0.25), bridge_config)
    if kind1_bridge.plus.shape != (bridge_config.period_samples,):
        raise AssertionError("kind-1 bridge PWM must keep the clock-rate shape")

    kind1_multi = pwm_kind1_bridge_multichannel(
        np.full(bridge_config.period_samples, -0.25),
        bridge_config,
        channels=4,
    )
    if np.any(kind1_multi.plus < 0.0) or np.any(kind1_multi.plus > 1.0):
        raise AssertionError("normalized bridge sums must stay in [0, 1]")
    if np.any(kind1_multi.minus < 0.0) or np.any(kind1_multi.minus > 1.0):
        raise AssertionError("normalized bridge sums must stay in [0, 1]")

    bridge_grouped_as_latched = pwm_kind2_bridge_fifo_grouped_multichannel(
        signed,
        bridge_config,
        samples_per_period=1,
        channels=4,
        mode="three_level",
    )
    bridge_latched = pwm_kind2_bridge_multichannel_latched(
        signed,
        bridge_config,
        channels=4,
        mode="three_level",
    )
    assert_bridge_close(bridge_grouped_as_latched, bridge_latched)

    bridge_grouped_as_interleaved = pwm_kind2_bridge_fifo_grouped_multichannel(
        np.array([0.25, -0.25, 0.75, -0.75]),
        bridge_config,
        samples_per_period=4,
        channels=4,
        mode="three_level",
    )
    bridge_interleaved = pwm_kind2_bridge_phase_interleaved(
        np.array([0.25, -0.25, 0.75, -0.75]),
        bridge_config,
        channels=4,
        mode="three_level",
    )
    assert_bridge_close(bridge_grouped_as_interleaved, bridge_interleaved)

    try:
        pwm_kind2_fifo_grouped_multichannel(samples, config, samples_per_period=3, channels=4)
    except ValueError:
        pass
    else:
        raise AssertionError("channels must be a multiple of samples_per_period")

    print("PWM checks passed")


if __name__ == "__main__":
    main()
