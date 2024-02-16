.. CalIPE_doc documentation master file, created by
   sphinx-quickstart on Fri Jul  7 17:54:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tutorial
========

This chapter presents some examples of the use of CalIPE. More examples, with the example data used in Provost (in prep, 2023), are available in the github repository xxxx. 

Launch a calibration
--------------------

Let's start with the calibration of the following mathematical formulation:
	- I = I0 + beta.log10(Hypo/H)

First, you have to add the path to your CalIPE repository to the Python paths:: 

	import sys
	sys.path.append('path_to_your_CalIPE_package/calib_fc')


Then you can import the CalIPE function adapted to the chosen mathematical formulation::

	from attKov_onedataset import Kovbeta_onedataset

Enter the Evt file, Obs file and Region file names. If you don't need the Region file, just put a blank 
name as following::
	
	evt_name = 'path_to_evtfile/evtfilename.txt'
	obs_name = 'path_to_obsfile/obsfilename.txt'
	regiondata_name = ''

The calibration dataset is given by the Evt file. All calibration functions will calibrate the chosen
mathematical formulation with all calibration earthquakes in the Evt file. 
Enter the name of the output folder::

	outputfolder = 'path_to_the_outputfolder'

The output folder should be created before launching the CalIPE function.
Then enter the name of the chosen weighting scheme::

	weighting_scheme = 'Ponderation evt-uniforme'

Choose the intensity binning method::

	binning_type = 'ROBS'

Choose the beta initial values::

	list_beta_ini = [-2.5, -3.0, -3.5]

You are ready to launch the calibration::

	Kovbeta_onedataset(evt_name, obs_name,
                           outputfolder=outputfolder,
                           liste_beta_ini=liste_beta_ini,
                           ponderation=ponderation,
                           binning_type=binning_type,
                           regiondata_name=regiondata_name,
                           NminIter=3, NmaxIter=100)


The result are saved in the output folder. Example python script calibrating this mathematical formulation
are available in the github repository (adress). Other calibration examples are also available in this repository.

Creating subsets
----------------


When using one of the calibration function, it is easy to use a for loop to explore different
weighting scheme or calibration dataset.

The CalIPE package provides a tool to create subsets from a given calibration dataset, based on filters on the metadata. Here is one example.
First, you have to add the path to your CalIPE repository to the Python paths:: 

	import sys
	sys.path.append('path_to_your_CalIPE_package/calib_fc')

Then import the needed functions::

	from create_subsets import create_liste_subset, filter_by_nevt, check_duplicate, create_basicdb_criteria
	from create_subsets import create_subsets

Create a table with the metadata used to create the subsets. Excel file can be created/modified manually::

	basic_db_name = 'path_to_evtfile/evtfilename.txt'
	obsdata_name = 'path_to_obsfile/obsfilename.txt'
	subset_folder = 'path_to_the_folder_where_the_subset_Evtfiles_will_be_saved'
	criteria = create_basicdb_criteria(basic_db_name, obsdata_name,
	                                   binning_type='ROBS',
	                                   outputfolder='../../Data',
	                                   regiondata_name='',
	                                   ponderation='IStdI_evtUni',
	                                   )

The table will be saved in the subset_folder. This folder should be created before lauching the create_basicdb_criteria function.
Then create the different lists of calibration earthquakes, corresponding to the different subsets.
Before that, an additional column QH should be added in the criteria excel file, corresponding to the depth
value quality, ranked from A (best quality) to E (Unknown depth). An example of the completed criteria Excel file is
available in the github repository (adress).
In this example, the metadata used to create subsets are the year of occurence of the earthquake, the depth quality,
the number of intensity class of the calibration earthquake macroseismic field, the number of intensity data points
of each calibration earthquake macroseismic field and the distance of completeness::

	global_liste, criteres = create_liste_subset(criteria,
                                             	     year_inf=[1980],
                                             	     year_sup=[2020, 2006],
                                             	     QH=[['A'], ['A', 'B'], ['A', 'B', 'C']],
                                                     NClass=[2, 3, 4, 5, 6],
                                                     Nobs=[10, 50, 100, 200],
                                                     Dc=[10, 25, 50, 100])

Once done, you can filter all calibration earthquake list with a number of earthquakes smaller than a certyain amount,
in this example 10 earthquakes::

	new_liste_nmin, new_critere_nmin = filter_by_nevt(global_liste, criteres, nmin=10)


To ensure that no subsets are identical, run this function::

	new_liste, new_critere = check_duplicate(new_liste_nmin, new_critere_nmin)

Once the list of the subsets is ready, the corresponding Evt files should be written::

	create_subsets(new_liste, new_critere, basic_db_name, folder=subset_folder)


The new Evt files, which correspond to each subset, are saved in the subset_folder defined previously.
The name of the subsets Evt files are Datasubsetxx.csv, where xx is the number of the subset.

Post-processing a calibration run
---------------------------------

After a calibration, some tests can be performed. One of them is the intensity residual analysis.
CalIPE provides tools to perform intensity residual analysis on the calibration outputs. In this example,
the intensity residual analysis will be done on the outputs of the calibration of the following
mathematical formulation::

	- I = I0 + beta.log10(Hypo/H)

First, you have to add the path to your CalIPE repository to the Python paths:: 

	import sys
	sys.path.append('path_to_your_CalIPE_package/calib_fc')


Then you can import the CalIPE function adapted to the chosen mathematical formulation::

	from CalIPE.postprocessing_fc.postprocessing_Kovbeta import plot_dIMag, plot_dII0


Do not forget to import the matplotlib library to plot the analysis::

	import matplotlib.pyplot as plt

Enter the name of the output folder where the outputs files are stored::

	path_subsets =  'Outputs/FRinstru01/Beta'

And the base name of the targeted inversion result (see repo xxx for a concrete example)::

	runname_basedb = 'basename'

Initialize the plots::

	figbdb_resMI0 = plt.figure(figsize=(10, 8))
	ax_resM = figbdb_resMI0.add_subplot(223)
	ax_resI0= figbdb_resMI0.add_subplot(224)
	ax_resM.grid(which='both')
	ax_resM.legend()
	ax_resI0.grid(which='both')
	ax_resI0.legend()
	ax_resI0.text(2, -1.25, '(c)', fontsize=15)
	ax_resM.text(2.8, -1.25, '(d)', fontsize=15)
	ax_resM.set_ylim([-1, 1])
	ax_resM.set_xlim([3, 5.5])
	ax_resI0.set_ylim([-1, 1])
	ax_resI0.set_xlim([3, 9])

And call the plot residual functions of the CalIPE library::
	
	plot_dIMag(runname_basedb, path_subsets, ax_resM, color='#1f77b4')
	plot_dII0(runname_basedb, path_subsets, ax_resI0, color='#1f77b4')



.. toctree::
   :maxdepth: 2
   :caption: Contents:

