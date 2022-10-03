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
#from combinaison_calib import calib_C1C2beta0, calib_C1C2beta1, calib_C1C2beta2
from combinaison_calib import calib_C1C2betaH, calib_C1C2betaHb
from prepa_data import prepare_input4calibration, add_Mweigths

# FR_instu
obsdata_name = '../../Data/ObsData/ObsCalibration_Fr_Instru02_filtoutsiders.txt'
evtdata_name = 'input_evt_calib_FRinstru_sansAlsace2018_HSihex.txt'
evtdata_name = 'input_evt_calib_FRinstru_sansAlsace2018.txt'
subset_folder = '../../Data/FR_instru_01/subsets_01'
#subset_folder = '../../Data/FR_instru_01/bootstrap_1evt_FRinstru'
evtcalib_folder = '../../Data/FR_instru_01/'
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

beta_liste = [-3]
#beta_liste = [-3.5]
C1_liste = [1]#, 2, 3]
C2_liste = [1]


for beta in beta_liste:
    print(beta)
    for C1 in C1_liste:
        print(C1)
        for C2 in C2_liste:
            print(C2)
            for ponderation, wp in zip(ponderation_list, poids_ponderation):
                print(ponderation)
                obsbin_plus = add_Mweigths(obsbin_plus, ponderation)
                if gamma_option:
                    pass
                else:
                    ObsBin_plus, result = calib_C1C2betaH(liste_evt, obsbin_plus,
                                                               C1, C2, beta,
                                                               inverse_depth=False,
                                                               inverse_I0=False,
                                                               NmaxIter=50)
                    
#                    resultb, ObsBin_plus_endb = calib_C1C2betaHb(liste_evt, obsbin_plus,
#                                                               C1, C2, beta,
#                                                               inverse_depth=False,
#                                                               inverse_I0=False,
#                                                               NmaxIter=50)
                    
                    print(result[0][:3])
                    print(result[0][3:])
                    #(C1C2BetaH2[0][:3])
                    
#                    (ObsBin_plus, C1, C2, beta,
#                     suivi_beta, suivi_C1, suivi_C2) = calib_C1C2beta0(liste_evt, obsbin_plus,
#                                                               C1, C2, beta,
#                                                               inverse_depth=False,
#                                                               inverse_I0=False,
#                                                               NmaxIter=50)
#                    stock_wrms, stock_C1, stock_C2, stock_beta = calib_C1C2beta1(liste_evt, obsbin_plus,
#                                                                   C1, C2, beta,
#                                                                   inverse_depth=False,
#                                                                   inverse_I0=False,
#                                                                   NmaxIter=50)
#                    ObsBin_plusb, wrms, C1b, C2b, betab = calib_C1C2beta2(liste_evt, obsbin_plus,
#                                                                   C1, C2, beta,
#                                                                   inverse_depth=False,
#                                                                   inverse_I0=False,
#                                                                   NmaxIter=50)
#
#                    print(C1, C2, beta)