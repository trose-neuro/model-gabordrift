# Gaze-Induced Tuning Changes in Mouse V1

This project is a reproducible first-pass null model for the question:

**How much change in inferred PO should be expected from gaze shifts alone in mouse V1 L2/3, and are natural-image responses more sensitive to gaze than grating tuning?**

The simulation builds a 500 x 500 um two-photon field of view, samples mouse-V1-like Gabor receptive fields, maps cells into visual coordinates with local retinotopy plus jitter, and compares grating tuning with natural-image population responses under global gaze shifts.

## Conda Environment

All scripts, notebooks, and tests are expected to run after activating the dedicated conda environment named `gaze_v1_sim`.

```bash
conda env create -f environment.yml
conda activate gaze_v1_sim
```

The `environment.yml` file includes Python, NumPy, SciPy, pandas, Matplotlib, seaborn, PyYAML, Jupyter, pytest, tqdm, scikit-image, xarray, and an editable install of this project.

To update an existing environment after dependency changes:

```bash
conda env update -f environment.yml --prune
conda activate gaze_v1_sim
```

## Quickstart

Run the fast sanity-check pipeline:

```bash
conda activate gaze_v1_sim
python scripts/run_main_simulation.py --debug
```

Run tests:

```bash
conda activate gaze_v1_sim
pytest
```

Run compact sensitivity sweeps:

```bash
conda activate gaze_v1_sim
python scripts/run_sensitivity_sweeps.py
```

Run a focused 1-10 degree drift sanity check without the full 2D grid:

```bash
conda activate gaze_v1_sim
python scripts/run_drift_sanity_check.py --debug --include-natural
```

Regenerate final figures from saved result tables:

```bash
conda activate gaze_v1_sim
python scripts/make_all_figures.py --debug
```

The full default config uses 1000 neurons and a 0 to 10 degree gaze grid in 0.1 degree steps for both azimuth and elevation. That is intentionally much heavier than debug mode:

```bash
conda activate gaze_v1_sim
python scripts/run_main_simulation.py --config configs/default.yaml
```

## Model Summary

Each neuron has:

- cortical position in a 500 x 500 um FOV
- RF center from a configurable local retinotopic map plus Gaussian jitter
- PO in [0, 180) degrees
- preferred spatial frequency
- RF envelope widths
- Gabor phase
- gain, baseline, and static output exponent

The grating pipeline uses a Gabor-inspired analytic response model. It separates orientation/SF matching from the phase shifts introduced when gaze translates the retinal image relative to RFs. The natural-image pipeline builds cosine and sine Gabor kernel banks on a visual grid and applies them to translated grayscale images.

## Decomposition

The central decomposition separates:

1. **Gaze only, perfect mapping, noiseless**: deterministic geometry.
2. **Gaze + mapping error, noiseless**: effects from approximate projection, wrong origin, wrong scale, anisotropy, rotation, or nonlinear distortion.
3. **Gaze + noise, perfect mapping**: SNR-limited estimation variability.
4. **Gaze + mapping error + noise**: combined experimental failure modes.

This distinction is important because apparent preferred-orientation drift can come from actual deterministic phase/geometry, coordinate mismatch, or noisy estimates from limited repeats.

## Natural Images

Set `stimuli.natural_images.folder` in a config YAML to a folder of grayscale or RGB images:

```yaml
stimuli:
  natural_images:
    folder: /path/to/images
```

Images are converted to grayscale, resized to `image_size_px`, normalized, and treated as spanning `extent_deg` visual degrees. If no folder is supplied, the project generates procedural naturalistic images so every script remains runnable.

## Configs

The main config is `configs/default.yaml`. It controls:

- neuron count and FOV size
- retinotopy transform, rotation, anisotropy, and jitter
- exact versus approximate spherical-to-plane mapping
- RF parameter distributions
- grating orientation/SF/phase grids
- natural-image size, extent, and count
- gaze-shift ranges
- noise model, SNR, and repeats
- thresholds and bootstrap sample counts

Additional configs override default assumptions:

- `broad_rf.yaml`
- `narrow_rf.yaml`
- `low_jitter.yaml`
- `high_jitter.yaml`
- `low_snr.yaml`
- `high_snr.yaml`
- `mapping_error_none.yaml`
- `mapping_error_small.yaml`
- `mapping_error_medium.yaml`

## Outputs

Generated outputs are written to:

- `results/figures/`: validation, heat maps, comparison panels, secondary plots
- `results/arrays/`: compressed arrays for selected responses/images
- `results/tables/`: CSV summaries plus `methods_summary.md` and `interpretation_summary.md`

Primary figures include heat maps for median and 90th percentile `|ΔPO|`, fraction of neurons above the `|ΔPO|` threshold, circular-variance change, tuning-strength change, natural-image response correlation, RDM similarity, response-change magnitude, grating versus natural-image summaries, decomposition summaries, and simple-cell versus energy-model summaries.

## Notebooks

Run notebooks only after activating `gaze_v1_sim`.

- `01_model_setup_and_sampling.ipynb`: cortical sampling, retinotopy, mapping, RF distributions, RF examples
- `02_grating_tuning_and_po_estimation.ipynb`: grating responses, PO estimation, circular variance, simple versus energy model
- `03_gaze_shift_heatmaps.ipynb`: grating gaze-shift heat maps and example tuning changes
- `04_natural_images_comparison.ipynb`: natural-image responses, correlations, RDMs, grating comparison
- `05_noise_and_mapping_error_effects.ipynb`: decomposition of geometry, mapping error, and SNR
- `06_parameter_sweeps_and_interpretation.ipynb`: RF, SF, jitter, mapping, and SNR sweeps

## Interpretation Targets

The generated interpretation summary addresses:

1. when gaze causes substantial `ΔPO`
2. when gaze mainly flattens tuning rather than rotating PO
3. how much instability mapping error explains alone
4. how much apparent PO instability low SNR explains alone
5. whether natural images are more sensitive than gratings
6. which assumptions make gaze effects large enough to matter for mouse V1 experiments

This is not a full biological model of mouse V1 L2/3. A single Gabor RF is a deliberate simplification used to make geometry, mapping error, and estimation variability explicit and testable.
