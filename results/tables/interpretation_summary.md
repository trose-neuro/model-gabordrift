# Interpretation Summary

At the largest simulated offset in this run (5.00, 5.00) deg, the median absolute inferred ΔPO was 9.49 deg and the 90th percentile was 37.28 deg.
The median tuning-strength change was -0.007; this distinguishes PO rotation from tuning flattening or sharpening.
For natural images at the same edge of the grid, population response correlation to baseline was -0.009, RDM similarity was 0.411, and the median relative per-neuron response change was 1.061.

Why can a gaze shift affect PO for full-field gratings at all? Translating an infinite grating does not rotate it; it advances stimulus phase at each RF by Δφ = 2π f (Δa cosθ + Δe sinθ). Therefore a perfectly phase-invariant cell, or a simple-cell response averaged over dense stimulus phases, should show little to no deterministic ΔPO under perfect mapping and no noise. Apparent ΔPO appears when finite phase sampling and rectification let that phase advance modulate different orientations unevenly, or when mapping error and noise perturb the estimated tuning curve.

Moving gratings add a temporal phase term, φtime(t) = -2πTFt. The gaze shift still contributes only a fixed spatial phase offset for a full-field grating. If responses are averaged over enough samples from a drift cycle, that fixed offset should cancel for the simple-cell model; the energy model should be stable even without dense temporal sampling. Sparse time samples remain phase-sensitive and can therefore mimic PO shifts.

Under this model, gaze matters little when RFs are broad, phase is averaged or an energy model is used, mapping is accurate, and SNR is high. Gaze matters more when RFs are narrow, high-SF units are common, phase-sensitive simple-cell responses are estimated from limited phases, or natural images contain local structure that translates across RF subfields.

Mapping errors matter when approximate projection, wrong scale, origin offsets, rotation mismatch, or nonlinear distortion produce systematic phase and position errors comparable to the RF subfield scale. Low SNR dominates when repeated-trial averaging and bootstrap intervals show large PO uncertainty even at zero or small gaze offsets.

Decomposition at the largest analyzed offsets:
- gaze_plus_mapping_error_noiseless: median |ΔPO| 9.72 deg, natural-image correlation -0.026, RDM similarity 0.403.
- gaze_only_perfect_noiseless: median |ΔPO| 9.49 deg, natural-image correlation -0.009, RDM similarity 0.411.
- gaze_plus_noise_perfect: median |ΔPO| 9.26 deg, natural-image correlation -0.009, RDM similarity 0.410.
- gaze_plus_mapping_error_plus_noise: median |ΔPO| 9.10 deg, natural-image correlation -0.026, RDM similarity 0.406.

Full-field grating phase sanity check at the largest tested drift:
- energy_model: median |ΔPO| 0.00 deg.
- simple_1_phase: median |ΔPO| 19.26 deg.
- simple_24_phases: median |ΔPO| 0.02 deg.
- simple_2_phases: median |ΔPO| 11.58 deg.
- simple_4_phases: median |ΔPO| 0.94 deg.

Moving-grating temporal sanity check at the largest tested drift:
- energy_model: median |ΔPO| 0.00 deg.
- simple_1_time: median |ΔPO| 19.26 deg.
- simple_24_times: median |ΔPO| 0.02 deg.
- simple_2_times: median |ΔPO| 6.22 deg.
- simple_4_times: median |ΔPO| 0.94 deg.

Longitudinal weekly decomposition at the latest simulated session:
- circuit_drift_only: measured gaze 4.00 deg, static-grating median |ΔPO| 4.13 deg, moving-grating median |ΔPO| 1.93 deg, natural-image correlation 0.981, natural-image RDM similarity 0.998.
- gaze_only: measured gaze 4.00 deg, static-grating median |ΔPO| 6.36 deg, moving-grating median |ΔPO| 0.94 deg, natural-image correlation 0.409, natural-image RDM similarity 0.822.
- gaze_plus_circuit_drift: measured gaze 4.00 deg, static-grating median |ΔPO| 7.40 deg, moving-grating median |ΔPO| 2.17 deg, natural-image correlation 0.404, natural-image RDM similarity 0.833.
- gaze_plus_mapping_error: measured gaze 4.00 deg, static-grating median |ΔPO| 6.72 deg, moving-grating median |ΔPO| 0.98 deg, natural-image correlation 0.398, natural-image RDM similarity 0.824.

This weekly comparison is the main test of the biological question: if the data mainly follow the gaze-only or gaze-plus-mapping-error curves, the model supports an optical or registration explanation. If the data require circuit-drift-only or gaze-plus-circuit-drift trajectories, then a cumulative neural-state change is needed in addition to gaze.

Natural images are considered more gaze-sensitive than gratings here when their population correlation or RDM similarity drops substantially while grating ΔPO remains small. That pattern is expected because translating a structured image can change RF drive without necessarily changing the PO that best fits grating responses.
