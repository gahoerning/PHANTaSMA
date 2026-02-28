# PHANTaSMA

**P**latform of **H**igh-level **A**nalysis, **N**umerical **T**echniques **a**nd **S**ky **M**ap **A**pplications

PHANTaSMA is a precision-oriented framework for astrophysical sky-map analysis.
Its purpose is to integrate established libraries (e.g. healpy, astropy, numpy, scipy) into rigorously controlled pipelines that minimize numerical inconsistencies and preprocessing biases.

The project does not aim to replace existing libraries. It enforces correct usage patterns, explicit assumptions, and reproducible workflows.

Planned Features
	•	RMS map smoothing (HEALPix and Cartesian)
	•	Safe HEALPix resolution changes (ud_grade with variance and pixel window control)
	•	Map smoothing and interpolation with beam consistency
	•	Simple SED MCMC (synchrotron, dust, free–free)
	•	Gaussian fitting via MCMC
	•	Aperture photometry utilities
	•	Template fitting (weighted linear regression)
	•	High-performance modules in C/C++ for heavy simulations

⸻

Principles
	•	Numerical exactness
	•	Explicit beam and resolution handling
	•	Reproducibility
	•	Modular design

⸻

Status

Early development.
