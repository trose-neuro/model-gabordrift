# Interpretation Summary

At the largest simulated offset in this run (5.00, 5.00) deg, the median absolute inferred OP shift was 9.49 deg and the 90th percentile was 37.28 deg.
The median tuning-strength change was -0.007; this distinguishes OP rotation from tuning flattening or sharpening.
For natural images at the same edge of the grid, population response correlation to baseline was -0.009, RDM similarity was 0.411, and the median relative per-neuron response change was 1.061.

Under this model, gaze matters little when RFs are broad, phase is averaged or an energy model is used, mapping is accurate, and SNR is high. Gaze matters more when RFs are narrow, high-SF units are common, phase-sensitive simple-cell responses are estimated from limited phases, or natural images contain local structure that translates across RF subfields.

Mapping errors matter when approximate projection, wrong scale, origin offsets, rotation mismatch, or nonlinear distortion produce systematic phase and position errors comparable to the RF subfield scale. Low SNR dominates when repeated-trial averaging and bootstrap intervals show large OP uncertainty even at zero or small gaze offsets.

Decomposition at the largest analyzed offsets:
- gaze_plus_mapping_error_noiseless: median |dOP| 9.72 deg, natural-image correlation -0.026, RDM similarity 0.403.
- gaze_only_perfect_noiseless: median |dOP| 9.49 deg, natural-image correlation -0.009, RDM similarity 0.411.
- gaze_plus_noise_perfect: median |dOP| 9.42 deg, natural-image correlation -0.010, RDM similarity 0.411.
- gaze_plus_mapping_error_plus_noise: median |dOP| 9.11 deg, natural-image correlation -0.026, RDM similarity 0.404.

Natural images are considered more gaze-sensitive than gratings here when their population correlation or RDM similarity drops substantially while grating OP shifts remain small. That pattern is expected because translating a structured image can change RF drive without necessarily changing the orientation that best fits grating responses.
