# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 09:27:38 2021

@author: PROVOST-LUD
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('../calib_fc')
sys.path.append('../postprocessing_fc')
from combinaison_calib import calib_C1C2
from prepa_data import prepare_input4calibration, add_Mweigths

# FR_instu
obsdata_name = '../../Data/ObsData/ObsCalibration_Fr_Instru02_filtoutsiders.txt'
evtdata_name = 'input_evt_calib_FRinstru_sansAlsace2018.txt'
subset_folder = '../../Data/FR_instru_01/subsets_01'
#subset_folder = '../../Data/FR_instru_01/bootstrap_1evt_FRinstru'
evtcalib_folder = '../../Data/FR_instru_01/'
beta_distrib_name= '../../Data/Att_distributions/FR_extended_01_beta.csv'
outputfolder = '../../Outputs/FR_instru_01/Subsets_01/IPE'
regiondata_name = '../../Data/Regions/region2_FRinstru.txt'


dummy_ponderation = 'Ponderation dI'
binning_type = 'RAVG'
gamma_option = False
ponderation_list = ['Ponderation evt-stdM', 'Ponderation mag_class']
ponderation_list = ['Ponderation mag_class']
poids_ponderation = [0.5, 0.5]


# Preparation des donnees
nom_evt_complet = evtcalib_folder + '/' + evtdata_name
obsbin_plus = prepare_input4calibration(obsdata_name, nom_evt_complet,
                                        dummy_ponderation,
                                        regiondata_name, binning_type)
liste_evt = np.unique(obsbin_plus.EVID.values)
beta_data = pd.read_csv(beta_distrib_name, sep=';')
for ind, row in beta_data.iterrows():
    beta = row['beta']
    proba = row['proba']
    print(beta)
    for ponderation, wp in zip(ponderation_list, poids_ponderation):
        obsbin_plus = add_Mweigths(obsbin_plus, ponderation)
        if gamma_option:
            pass
        else:
            ObsBin_plus, C1, C2 = calib_C1C2(liste_evt, obsbin_plus, beta, 0, NmaxIter=50)
            print(proba*wp, C1, C2, beta)
