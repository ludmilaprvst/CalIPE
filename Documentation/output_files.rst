.. CalIPE_doc documentation master file, created by
   sphinx-quickstart on Fri Jul  7 17:54:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Output data files
=================

After a calibration, the output results will be saved into files. The format of the output file
depends on the choice of the mathematical formulation chosen by the user. The different output
files will be described inthe following.
Some post-processing functions have also output files, also described here.

Output files of the mathematical formulation I = I0 + beta.log10(Hypo/H)
------------------------------------------------------------------------

Three different output files are produced for the calibration of this mathematical formulation,
done with the Kovbeta_onedataset() function.
The first file, called betaFinal file, is the summary of the calibration. Its name is :
betaFinal_(name of the input Evt file)_w(name of the chosen weighting scheme)_betaini(absolute value of initial value of beta used x 10).txt
It contains the output value of the beta coefficient, its associated standard deviation, the value of the initial value of beta, the number
of iteration used to converge to the output solution, the maximal and the minimal number of iteration allowed, the number of calibration
earthquake used, the number of intensity data used, the path and name of the Evt and Obs input files used and the path and name of the python
script used to launch the calibration.
Here an example::

	Beta:-3.29813743678502
	Beta Std:0.08168377779199379
	Beta ini:-3.0
	Nbre iteration:17 
	Nbre iteration max:100 
	Nbre iteration min:3 
	Nbre evt:31
	Nbre data:164
	Fichier Evt:Data/FRinstru01.txt
	Fichier Obs:Data/Obs_FRextended.txt
	Nom fichier execute: CalIPE_launchers_example\calibration_AttKov_launch_FRinstru.py

The second file, called betaWay file, is a file with the value of coefficient beta after each iteration. Its name is:
betaWay_(name of the input Evt file)_w(name of the chosen weighting scheme)_betaini(absolute value of initial value of beta used x 10).txt
This file has two columns, separated by ;. First column is the number of iteration, 0 representing the initial value used. The second columns
corresponds to the value of coefficient beta after the concerned iteration. Here the first six lines of the betaWay file associated to the previous 
betaFinal file presented in example::

	;Beta values during inversion
	0;-3.0
	1;-3.0659111272544592
	2;-3.1168399538635403
	3;-3.1570590615729452
	4;-3.1886428826593454
	5;-3.213723752423889

The thrird file, called obsbinEnd file, contains the data used in the inversion after the inversion process, with the final values of depth and epicentral intensity for
each calibration earthquake, which are inverted successively with the beta coefficient. The name of the file is:
obsbinEnd_(name of the input Evt file)_w(name of the chosen weighting scheme)_betaini(absolute value of initial value of beta used x 10).csv

It is a .csv file and the data are stored in a table, with the following columns:
	- EVID: ID of the earthquake,
	- I: value of the intensity bin,
	- StdI: uncertainty associated to I,
	- Io: value of the inverted epicentral intensity
	- Io_std: uncertainty associated to the initial value of epicentral intensity,
	- Io_ini: value of the initial value of epicentral intensity,
	- Depi: epicentral distance associated to I,
	- Ndata: number of data used to compute the intensity bin,
	- Mag: magnitude of the earthquake,
	- StdM: uncertainty assocaited to the magnitude,
	- Depth: hypocentral depth of the earthquake,
	- Hmin: minimal (shallow) limit used to invert depth,
	- Hmax: maximal (deep) limit used to invert depth,
	- Hmin_ini: initial value of Hmin. In this calibration case, Hmin=Hmin_ini,
	- Hmax_ini: initial value of Hmin. In this calibration case, Hmax=Hmax_ini,
	- StdIo_inv: uncertainty associated to the inverted value of epicentral intensity,
	- RegID: ID of the region where the earthquake is located (defined by the region input file),
	- Poids: weight given to the intensity bin in the inversion,
	- eqStd: equivalent standard deviation of Poids, input of the scipy.optimize curve_fit() function

Additional columns can appear, depending on the chosen weighting scheme and the number of weight layer used (see the weighting
functions documentation):
	- Poids int: weight associated to the I, based on StdI,
	- Poids_int_norm: normalized by earthquake poids_int,
	- Poids_evt: weight of each earthquake,
	- Poids_evt_norm: normalized Poids_evt, 
	- poids_gp : weight of the group to which the earthquake belong,


Output files of the mathematical formulation I = I0 + beta.log10(Hypo/H) + gamma.(Hypo/H)
-----------------------------------------------------------------------------------------

Three different output files are produced for the calibration of this mathematical formulation,
done with the Kovbetagamma_onedataset() function.

The first file is the summary of the calibration. Its name is :
betagammaFinal_(name of the input Evt file)_w(name of the chosen weighting scheme)_betaini(absolute value of initial value of beta used x 10).txt
It contains the output value of the beta coefficient, its associated standard deviation, the value of the initial value of beta,
the output value of the gamma coefficient, its associated standard deviation, the value of the initial value of gamma, the number
of iteration used to converge to the output solution, the maximal and the minimal number of iteration allowed, the number of calibration
earthquake used, the number of intensity data used, the path and name of the Evt and Obs input files used and the path and name of the python
script used to launch the calibration.
Here an example::

	Beta:-3.2958604621345904
	Beta Std:0.16938930506645936
	Beta ini:-3.0
	Gamma:-9.07844740022615e-07
	Gamma Std:0.0029515662825366803
	Gamma ini:-0.001
	Nbre iteration:19 
	Nbre iteration max:100 
	Nbre iteration min:3 
	Nbre evt:31
	Nbre data:164
	Fichier Evt:Data/FRinstru01.txt
	Fichier Obs:Data/Obs_FRextended.txt
	Nom fichier execute:CalIPE_launchers_example\calibration_AttKov_launch_FRinstru.py

The second file is a file with the value of coefficient beta after each iteration. It is the same betaWay file described before.
The third file contains the data used in the inversion after the inversion process, with the final values of depth and epicentral intensity for
each calibration earthquake, which are inverted successively with the beta coefficient. It is the same obsbinEnd file described before.

Output files of the mathematical formulation I = C1 + C2 + beta.log10(Hypo)  and I = C1 + C2 + beta.log10(Hypo) + gamma.Hypo
----------------------------------------------------------------------------------------------------------------------------

The functions calib_C1C2(), calib_C1C2H, calib_C1C2betaH and calib_C1C2betagammaH() don't produce output files. However, their outputs
can be easly stored in output files as suggested in the launcher examples provided in the github repository xxxx.

.. toctree::
   :maxdepth: 2
   :caption: Contents:




