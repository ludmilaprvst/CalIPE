# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:50:57 2021

@author: PROVOST-LUD
"""
import sys
sys.path.append('../calib_fc')
sys.path.append('../postprocessing_fc')

from prepa_data import fichier_input
import EvtIntensityObject as eio
import pandas as pd

obsdata_name = 'O:\ENV\SCAN\BERSSIN\R4\Projet 1.4.4\2 - Macrosismicité\2022 stage master profondeurs Vallée Rhône\Travail\inputevtcalibFRinstru02.txt'
evtdata_name = 'O:\ENV\SCAN\BERSSIN\R4\Projet 1.4.4\2 - Macrosismicité\2022 stage master profondeurs Vallée Rhône\Travail\ObsCalibrationFrextendedfiltered.txt'

depth = 2
Ic = 3
obsdata = pd.read_csv(obsdata_name, sep=';')
evtdata = pd.read_csv(evtdata_name, sep=';')
fichiers = fichier_input(obsdata, evtdata)
data = eio.Evt(fichiers)
data.build(2019111102)
data.Binning_Obs(depth, Ic)
print(data.ObsBinn)

