from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SpectrumResult:
    """One named amplitude spectrum with selected dominant peaks."""

    name: str
    sample_rate: float
    freqs: np.ndarray
    amplitudes: np.ndarray
    peaks: list[tuple[float, float]]

    @property
    def dominant_frequency(self) -> float | None:
        return self.peaks[0][0] if self.peaks else None


def amplitude_spectrum(
    x: np.ndarray,
    sample_rate: float,
    *,
    remove_dc: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided amplitude spectrum for a real signal."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if values.size == 0:
        raise ValueError("x must contain at least one sample")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if remove_dc:
        values = values - np.mean(values)
    freqs = np.fft.rfftfreq(values.size, d=1.0 / float(sample_rate))
    amps = np.abs(np.fft.rfft(values)) * 2.0 / values.size
    if amps.size:
        amps[0] *= 0.5
    return freqs, amps


def _select_peaks(
    freqs: np.ndarray,
    amps: np.ndarray,
    *,
    count: int,
    f_min: float,
    f_max: float,
    min_spacing_hz: float,
) -> list[tuple[float, float]]:
    if count <= 0:
        return []
    candidates = np.flatnonzero((freqs >= f_min) & (freqs <= f_max))
    order = candidates[np.argsort(amps[candidates])[::-1]]
    peaks: list[tuple[float, float]] = []
    for idx in order:
        freq = float(freqs[idx])
        if any(abs(freq - old_freq) < min_spacing_hz for old_freq, _ in peaks):
            continue
        peaks.append((freq, float(amps[idx])))
        if len(peaks) >= count:
            break
    return peaks


def spectrum_result(
    name: str,
    x: np.ndarray,
    sample_rate: float,
    *,
    peak_count: int = 8,
    f_min: float = 0.0,
    f_max: float | None = None,
    min_spacing_hz: float = 0.0,
    remove_dc: bool = True,
) -> SpectrumResult:
    """Compute a named amplitude spectrum and its dominant peaks."""
    freqs, amps = amplitude_spectrum(x, sample_rate, remove_dc=remove_dc)
    if f_max is None:
        f_max = sample_rate / 2.0
    peaks = _select_peaks(
        freqs,
        amps,
        count=peak_count,
        f_min=f_min,
        f_max=f_max,
        min_spacing_hz=min_spacing_hz,
    )
    return SpectrumResult(
        name=name,
        sample_rate=float(sample_rate),
        freqs=freqs,
        amplitudes=amps,
        peaks=peaks,
    )


def spectra_for_waveforms(
    waveforms: Mapping[str, object],
    sample_rate: float,
    *,
    peak_count: int = 8,
    f_min: float = 0.0,
    f_max: float | None = None,
    min_spacing_hz: float = 0.0,
    remove_dc: bool = True,
) -> dict[str, SpectrumResult]:
    """Compute spectra for plain arrays and bipolar PWM-like objects."""
    spectra: dict[str, SpectrumResult] = {}
    for name, value in waveforms.items():
        if all(hasattr(value, attr) for attr in ("positive", "negative", "differential")):
            parts = {
                f"{name}.positive": getattr(value, "positive"),
                f"{name}.negative": getattr(value, "negative"),
                f"{name}.differential": getattr(value, "differential"),
            }
        else:
            parts = {name: value}
        for part_name, part_value in parts.items():
            spectra[part_name] = spectrum_result(
                part_name,
                np.asarray(part_value, dtype=np.float64),
                sample_rate,
                peak_count=peak_count,
                f_min=f_min,
                f_max=f_max,
                min_spacing_hz=min_spacing_hz,
                remove_dc=remove_dc,
            )
    return spectra


def peak_frequency(
    x: np.ndarray,
    sample_rate: float,
    *,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> tuple[float, float]:
    """Return the strongest spectral peak in a selected frequency band."""
    freqs, amps = amplitude_spectrum(x, sample_rate, remove_dc=True)
    if f_max is None:
        f_max = sample_rate / 2.0
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mask):
        raise ValueError("empty frequency band")
    local = np.flatnonzero(mask)
    idx = local[np.argmax(amps[local])]
    return float(freqs[idx]), float(amps[idx])


def dominant_peaks(
    x: np.ndarray,
    sample_rate: float,
    *,
    count: int = 8,
    f_min: float = 0.0,
    f_max: float | None = None,
    min_spacing_hz: float = 0.0,
) -> list[tuple[float, float]]:
    """Return dominant spectral peaks, with optional spacing between peaks."""
    freqs, amps = amplitude_spectrum(x, sample_rate, remove_dc=True)
    if f_max is None:
        f_max = sample_rate / 2.0
    return _select_peaks(
        freqs,
        amps,
        count=count,
        f_min=f_min,
        f_max=f_max,
        min_spacing_hz=min_spacing_hz,
    )


def moving_average_decimate(x: np.ndarray, factor: int) -> np.ndarray:
    """Average groups of samples and keep one value per group."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    values = np.asarray(x, dtype=np.float64)
    n_groups = values.size // factor
    if n_groups == 0:
        return np.array([], dtype=np.float64)
    return values[: n_groups * factor].reshape(n_groups, factor).mean(axis=1)
