# Methods Summary

This first-pass null model simulates 160 L2/3-like units sampled uniformly in a 500 x 500 um V1 imaging field. Cortical positions are mapped to visual RF centers by a configurable local linear retinotopic transform with Gaussian cell-to-cell scatter. Each unit has a single Gabor-like receptive field with sampled PO, spatial-frequency preference, RF envelope widths, phase, gain, baseline, and static nonlinearity.

Full-field static and moving gratings are evaluated with a Gabor-inspired analytic response model that separates orientation/SF matching from gaze-dependent phase. Moving gratings add a sampled temporal phase axis so sparse versus dense drift-cycle averaging can be compared. Natural-image responses are evaluated by applying sampled Gabor kernels to preprocessed grayscale images. If no image folder is provided, the pipeline uses procedural naturalistic images so the project remains runnable.

Gaze is modeled as a global translation of the retinal image relative to the RFs. The configured grid spans azimuth 0.0 to 5.0 deg and elevation 0.0 to 5.0 deg in 1.0 deg azimuth steps and 1.0 deg elevation steps. Exact spherical projection and parameterized approximate mappings can be compared. Output noise can be disabled or simulated as Gaussian, Poisson-like, multiplicative gain noise, or combined Gaussian plus multiplicative noise.

All scripts, notebooks, and tests are intended to run after activating the dedicated conda environment `gaze_v1_sim`.
