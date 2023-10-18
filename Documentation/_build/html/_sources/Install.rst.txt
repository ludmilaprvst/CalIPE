.. CalIPE_doc documentation master file, created by
   sphinx-quickstart on Fri Jul  7 17:54:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
============
Requirements
------------

To run CalIPE, python 3 is needed with the following python libraries:
 * matplotlib 3.5.1
 * basemap 1.2.2
 * numpy 1.21.5
 * pandas 1.4.2
 * scipy 1.7.3

The Anaconda distribution of python includes all these libraries except for the Basemap library.
The open-source Basemap library can be downloaded and installed from the Anaconda cloud.

CalIPE has been tested with Python 3.9.12 and the python library versions previously listed on a windows environment.

Install
-------

Clone the repository on your chosen local repository or download it.

Use CalIPE
----------
To use CalIPE, please import in your python script the needed functions.
To be sure that python find CalIPE, please write the following lines
at the beginning of your python script::

	import sys
	sys.path.append('path_to_your_CalIPE_package')


.. toctree::
   :maxdepth: 2
   :caption: Contents:

