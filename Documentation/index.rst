.. CalIPE documentation master file, created by
   sphinx-quickstart on Fri Jul  7 17:54:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CalIPE's documentation!
======================================

The CalIPE package is a python open-source package with tools to calibrate intensity prediction equation
and to explore some epistemic uncertainties inherent to intensity prediction equation calibration.

The mathematical formulation available in CalIPE are:
      - I = C1 + C2.M + beta.log10(Hypo) + gamma.Hypo
      - I = C1 + C2.M + beta.log10(Hypo)
      - I = I0 + beta.log10(Hypo/H) + gamma.(Hypo/H)
      - I = I0 + beta.log10(Hypo/H)

Where I is the intensity at hypocentral distance Hypo, M the magnitude of the earthquake, H the hypocentral
depth of the earthquake, I0 the epicentral intensity, C1 and C2 magnitude coefficients, beta and gamma the
geometric and intresic attenuation coefficients respectively.

All inversion processes were with the optimize.curve_fit() function in SciPy, that uses non-linear least squares
to fit a function to data.

Calibration of the two first mathematical formulation can be done in one step or in two steps (first step,
calibration of attenuation, and second step, calibration of the magnitude coefficients).

The following epistemic uncertainty can be explored:
	- choice of the calibration dataset,
	- choice of the weighting scheme,
	- choice of the coefficient initial values.

The package provides also tools to analyze the calibration outputs.

This package is presented in Provost (in prep., 2023).

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   Install
   input_files
   examples
   output_files
   inventory



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
