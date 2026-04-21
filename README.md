# Gaze-Induced Tuning Changes in Mouse V1

This project is a reproducible first-pass null model for the question:

**How much change in inferred PO should be expected from gaze shifts alone in mouse V1 L2/3, and are natural-image responses more sensitive to gaze than grating tuning?**

The simulation builds a 500 x 500 um two-photon field of view, samples mouse-V1-like Gabor receptive fields, maps cells into visual coordinates with local retinotopy plus jitter, and compares grating tuning with natural-image population responses under global gaze shifts.

## What This Project Tests

The project is designed to separate three mechanisms that can otherwise be conflated:

1. **True deterministic gaze geometry**: what changes when the same RFs view the same stimuli after a global retinal translation.
2. **Coordinate/mapping error**: what changes when the analysis uses an imperfect visual-coordinate projection.
3. **SNR-limited estimation**: what changes when noisy or sparse responses make PO estimates unstable.

The central output is `ΔPO`, the circular change in inferred PO relative to the zero-shift baseline. Because orientation is 180-degree periodic, `|ΔPO|` is always the shortest orientation-axis change. For example, a change from 175 degrees to 5 degrees is `|ΔPO| = 10` degrees, not 170 degrees.

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

Recommended first pass:

1. Run `python scripts/run_main_simulation.py --debug`.
2. Open `results/tables/interpretation_summary.md`.
3. Inspect `results/figures/validation_grating_phase_sanity.png`.
4. Run notebooks 01, 02, and 05 before interpreting the full heat maps.

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

For full-field gratings, this distinction is critical. A gaze shift translates the grating, so it changes the phase at an RF:

```text
Δφ = 2π f (Δa cosθ + Δe sinθ)
```

It does not geometrically rotate the grating. Therefore, with perfect mapping and no noise, a phase-invariant energy model or a densely phase-averaged simple-cell model should show little to no deterministic `ΔPO`. Large `ΔPO` for gratings is interpreted as phase-sensitive estimation, finite phase sampling, mapping error, or SNR-driven instability rather than literal stimulus rotation.

The debug run includes this explicit sanity check:

| Condition | Median `|ΔPO|` at 10 degree diagonal drift |
| --- | ---: |
| energy model | 0.00 deg |
| simple cell, 24 phases | about 0.02 deg |
| simple cell, 4 phases | about 0.94 deg |
| simple cell, 2 phases | about 11.58 deg |
| simple cell, 1 phase | about 19.26 deg |

This is the intended behavior. It shows that the model is not claiming a translated full-field grating rotates. Instead, apparent grating `ΔPO` comes from phase-sensitive response sampling and estimation.

The schematic figure `results/figures/scheme_grating_gaze_rf_delta_po.png` visualizes the mechanism: baseline grating over an RF, gaze-shifted grating over the same RF, orientation-dependent phase advance, and the resulting apparent PO shift from sparse simple-cell phase sampling.

Moving gratings are handled with the same logic, but temporal samples replace static phase samples:

```text
I(x, y, t) = cos(2π f (x cosθ + y sinθ) - 2π TF t + ψ)
```

For a full-field moving grating, gaze still adds only a fixed spatial phase offset. Motion sweeps phase through time. Therefore, a sparse time sample can show apparent `ΔPO` in a phase-sensitive simple-cell model, while dense sampling over a drift cycle or an energy model should strongly suppress deterministic `ΔPO`. The moving-grating sanity outputs are:

- `results/tables/moving_grating_temporal_sanity_summary.csv`
- `results/figures/validation_moving_grating_temporal_sanity.png`
- `results/figures/scheme_moving_grating_gaze_rf_delta_po.png`

## Decomposition

The central decomposition separates:

1. **Gaze only, perfect mapping, noiseless**: deterministic geometry.
2. **Gaze + mapping error, noiseless**: effects from approximate projection, wrong origin, wrong scale, anisotropy, rotation, or nonlinear distortion.
3. **Gaze + noise, perfect mapping**: SNR-limited estimation variability.
4. **Gaze + mapping error + noise**: combined experimental failure modes.

This distinction is important because apparent preferred-orientation drift can come from actual deterministic phase/geometry, coordinate mismatch, or noisy estimates from limited repeats.

The decomposition table is saved at:

```text
results/tables/decomposition_summary.csv
```

The generated summary text is saved at:

```text
results/tables/interpretation_summary.md
```

The weekly hypothesis comparison that cleanly separates gaze drift, mapping error, and circuit drift is saved at:

```text
results/tables/longitudinal_hypothesis_summary.csv
```

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
- moving-grating orientation/SF/temporal-frequency/time-sampling grids
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

Primary figures include heat maps for median and 90th percentile `|ΔPO|`, fraction of neurons above the `|ΔPO|` threshold, circular-variance change, tuning-strength change, natural-image response correlation, RDM similarity, response-change magnitude, grating versus natural-image summaries, decomposition summaries, simple-cell versus energy-model summaries, and a full-field grating phase-sanity check.

Additional explanatory figures compare weekly trajectories for static gratings, moving gratings, and natural images under separate gaze-only, mapping-error, circuit-drift-only, and combined hypotheses.

Important result files:

- `results/tables/grating_phase_sanity_summary.csv`
- `results/tables/moving_grating_temporal_sanity_summary.csv`
- `results/tables/moving_grating_drift_sanity_summary.csv`
- `results/tables/drift_sanity_summary.csv`
- `results/tables/grating_shift_summary.csv`
- `results/tables/natural_image_shift_summary.csv`
- `results/tables/decomposition_summary.csv`
- `results/tables/longitudinal_hypothesis_summary.csv`
- `results/figures/scheme_grating_gaze_rf_delta_po.png`
- `results/figures/scheme_moving_grating_gaze_rf_delta_po.png`
- `results/figures/validation_grating_phase_sanity.png`
- `results/figures/validation_moving_grating_temporal_sanity.png`
- `results/figures/drift_sanity_median_abs_delta_po.png`
- `results/figures/drift_sanity_moving_grating_median_abs_delta_po.png`
- `results/figures/primary_06_natural_population_correlation.png`
- `results/figures/primary_07_natural_rdm_similarity.png`
- `results/figures/primary_12_longitudinal_hypothesis_panels.png`
- `results/figures/primary_13_longitudinal_gaze_vs_similarity.png`

## Notebooks

Run notebooks only after activating `gaze_v1_sim`.

- `01_model_setup_and_sampling.ipynb`: explains the simulated FOV, sampled RF parameters, retinotopy, coordinate mapping, and RF examples.
- `02_grating_tuning_and_po_estimation.ipynb`: explains static and moving grating phase advance, PO estimation, tuning curves, simple-cell versus energy-model behavior, phase-sampling checks, and moving-grating temporal averaging checks.
- `03_gaze_shift_heatmaps.ipynb`: shows static and moving grating `ΔPO`, circular-variance, and tuning-strength heat maps, plus neuron-level sensitivity examples.
- `04_natural_images_comparison.ipynb`: shows image inputs, natural-image population correlations, RDM similarity, response matrices, and grating-versus-natural bridge plots.
- `05_noise_and_mapping_error_effects.ipynb`: shows the four-way decomposition, mapping-error fields, and bootstrap PO uncertainty under different SNR regimes.
- `06_parameter_sweeps_and_interpretation.ipynb`: ranks sensitivity across RF size, jitter, mapping error, and SNR assumptions.

Notebook 06 also includes a weekly hypothesis section that overlays static gratings, moving gratings, and natural images against measured cumulative gaze drift so optical and circuit-drift explanations can be compared directly.

## Interpretation Targets

The generated interpretation summary addresses:

1. when gaze causes substantial `ΔPO`
2. when gaze mainly flattens tuning rather than rotating PO
3. how much instability mapping error explains alone
4. how much apparent PO instability low SNR explains alone
5. whether natural images are more sensitive than gratings
6. which assumptions make gaze effects large enough to matter for mouse V1 experiments

This is not a full biological model of mouse V1 L2/3. A single Gabor RF is a deliberate simplification used to make geometry, mapping error, and estimation variability explicit and testable.

## Bottom Line

Under this null model, natural-image responses can be highly gaze-sensitive because translating an image changes the local structure falling on each RF. Full-field gratings are different: a gaze shift changes phase, not orientation. Grating `ΔPO` should therefore be interpreted as an estimation consequence of phase sensitivity, sparse phase sampling, mapping error, or low SNR unless it survives the phase-invariant and dense-phase sanity checks.
