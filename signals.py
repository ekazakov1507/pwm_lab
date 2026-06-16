from __future__ import annotations

import numpy as np


def time_axis(sample_rate: float, n_samples: int) -> np.ndarray:
    """Return a zero-based time axis."""
    return np.arange(n_samples, dtype=np.float64) / float(sample_rate)


def sine_samples(
    freq: float,
    sample_rate: float,
    n_samples: int,
    *,
    amplitude: float = 0.9,
    offset: float = 0.5,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a normalized sine wave in the [0, 1] PWM-code range."""
    t = time_axis(sample_rate, n_samples)
    x = offset + 0.5 * amplitude * np.sin(2.0 * np.pi * freq * t + phase)
    return t, np.clip(x, 0.0, 1.0)


def sine_signed(
    freq: float,
    sample_rate: float,
    n_samples: int,
    *,
    amplitude: float = 0.9,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a signed sine wave in [-amplitude, amplitude]."""
    t = time_axis(sample_rate, n_samples)
    x = amplitude * np.sin(2.0 * np.pi * freq * t + phase)
    return t, np.clip(x, -1.0, 1.0)


def lfm_samples(
    f_start: float,
    f_end: float,
    sample_rate: float,
    n_samples: int,
    *,
    amplitude: float = 0.9,
    offset: float = 0.5,
    phase0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a normalized linear-frequency-modulated sine wave."""
    t = time_axis(sample_rate, n_samples)
    duration = n_samples / float(sample_rate)
    slope = (f_end - f_start) / duration
    phase = phase0 + 2.0 * np.pi * (f_start * t + 0.5 * slope * t * t)
    x = offset + 0.5 * amplitude * np.sin(phase)
    return t, np.clip(x, 0.0, 1.0)


def lfm_signed(
    f_start: float,
    f_end: float,
    sample_rate: float,
    n_samples: int,
    *,
    amplitude: float = 0.9,
    phase0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a signed linear-frequency-modulated sine wave."""
    t = time_axis(sample_rate, n_samples)
    duration = n_samples / float(sample_rate)
    slope = (f_end - f_start) / duration
    phase = phase0 + 2.0 * np.pi * (f_start * t + 0.5 * slope * t * t)
    x = amplitude * np.sin(phase)
    return t, np.clip(x, -1.0, 1.0)


def normalize_signed(x: np.ndarray, *, amplitude: float = 0.9) -> np.ndarray:
    """Center and scale a signal to a signed range."""
    values = np.asarray(x, dtype=np.float64)
    centered = values - np.mean(values)
    scale = np.max(np.abs(centered))
    if scale == 0:
        return np.zeros_like(centered)
    return np.clip(amplitude * centered / scale, -1.0, 1.0)


def lorenz_components(
    n_samples: int,
    *,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    initial: tuple[float, float, float] = (1.0, 1.0, 1.0),
    discard: int = 200,
) -> np.ndarray:
    """Generate Lorenz-system samples using a fixed-step RK4 integrator."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    def f(state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array(
            [
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z,
            ],
            dtype=np.float64,
        )

    total = n_samples + max(0, discard)
    state = np.array(initial, dtype=np.float64)
    out = np.empty((total, 3), dtype=np.float64)
    for idx in range(total):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[idx] = state
    return out[max(0, discard) :]


def lorenz_samples(
    sample_rate: float,
    n_samples: int,
    *,
    component: str = "x",
    dt: float = 0.01,
    amplitude: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one normalized signed Lorenz component for PWM experiments."""
    components = lorenz_components(n_samples, dt=dt)
    mapping = {"x": 0, "y": 1, "z": 2}
    key = component.lower()
    if key not in mapping:
        raise ValueError("component must be 'x', 'y', or 'z'")
    t = time_axis(sample_rate, n_samples)
    return t, normalize_signed(components[:, mapping[key]], amplitude=amplitude)


def quantize_unit(x: np.ndarray, bits: int) -> np.ndarray:
    """Quantize normalized [0, 1] values to integer PWM codes."""
    if bits <= 0:
        raise ValueError("bits must be positive")
    levels = (1 << bits) - 1
    return np.rint(np.clip(x, 0.0, 1.0) * levels).astype(np.int64)
