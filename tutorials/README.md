# PWM Lab Jupyter Tutorials

These notebooks are a plotting-oriented tutorial layer over the existing
`pwm_lab` package. Open them in order when learning the models, or jump to the
specific modulation family you need.

## Notebook Order

1. `00_setup_and_plot_helpers.ipynb` - imports, shared plotting helpers, basic PWM plot and spectrum.
2. `01_pwm_kind1_kind2.ipynb` - PWM kind 1 versus kind 2, including the FIFO-rate effect.
3. `02_pdm_delta_sigma.ipynb` - first-order and second-order PDM/delta-sigma, spectra, signed and bipolar forms.
4. `03_multichannel_pwm.ipynb` - same-phase and phase-interleaved multi-channel PWM.
5. `04_grouped_fifo_pwm.ipynb` - grouped FIFO samples mapped to summed PWM channels.
6. `05_bipolar_bridge_pwm.ipynb` - bipolar branches and bridge plus/minus modes.
7. `06_comparisons.ipynb` - side-by-side spectra and simple reconstruction-error checks.

## Running

From the repository root:

```powershell
jupyter lab tutorials
```

The notebooks can also be executed from the repository root or from the
`tutorials` folder. The first code cell in each notebook adjusts `sys.path` so
that it imports the local checkout of `pwm_lab`.

## Notes

- PWM output spectra use `sample_rate=config.f_clk`.
- PDM and direct delta-sigma spectra use the data/output rate `f_data`.
- Grouped FIFO PWM separates FIFO read throughput from the number of summed
  physical channels; its period-average signal represents the average of each
  FIFO group.
- The examples are numerical architecture checks. They do not model power
  stages, transformers, analog filters, loads, dead time, or bit-accurate HDL
  timing.
