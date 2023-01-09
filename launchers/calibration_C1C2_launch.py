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
from combinaison_calib import calib_C1C2, calib_C1C2H
from prepa_data import prepare_input4calibration, add_Mweigths, add_I0as_datapoint

# FR_instu
obsdata_nameFR = '../../Data/ObsData/ObsCalibration_Frextended_filtered.txt'
evtcalib_folderFR = '../../Data/FR_instru_01/'
evtdata_nameFR = 'input_evt_calib_FRinstru01.txt'
# IT_instu
obsdata_nameIT = '../../Data/ObsData/obs_IT.txt'
evtcalib_folderIT = '../../Data/IT_instru01/'
evtdata_nameIT = 'IT_instru_01c.txt'
# beta distrib
beta_distrib_nameFR = '../../Outputs/FR_instru_01/Hist_beta_FRinstru01_Subsets02.txt'
beta_distrib_nameIT = '../../Outputs/IT_instru_01/Hist_beta_ITinstru01_Subsets01c.txt'
##### Dessiner la zone FR et IT ##############
regiondata_name = '../../Data/Regions/region_FRIT.txt'
##### Dessiner la zone FR et IT ##############
outputfolder = '../../Outputs/FR_instru_01/IPE/TwoStep'



dummy_ponderation = 'Ponderation dI'
binning_type = 'ROBS'
invDepth_option = True
addI0 = True
ponderation_list = ['Ponderation evt-stdM', 'Ponderation mag_class']
#ponderation_list = ['Ponderation mag_class']
poids_ponderation = [0.5, 0.5]


# Preparation des donnees
nom_evt_completFR = evtcalib_folderFR + '/' + evtdata_nameFR
obsbin_plus_FR = prepare_input4calibration(obsdata_nameFR, nom_evt_completFR,
                                        dummy_ponderation,
                                        regiondata_name, binning_type)
#obsbin_plus_FR = add_Mweigths(obsbin_plus_FR, 'Ponderation mag_class')

nom_evt_completIT = evtcalib_folderIT + '/' + evtdata_nameIT
obsbin_plus_IT = prepare_input4calibration(obsdata_nameIT, nom_evt_completIT,
                                        dummy_ponderation,
                                        regiondata_name, binning_type)
frames = [obsbin_plus_FR, obsbin_plus_IT]
obsbin_plus = pd.concat(frames)
obsbin_plus = obsbin_plus.astype({'EVID': 'string'})

obsbin_plus = add_Mweigths(obsbin_plus, 'Ponderation mag_class')
# Function calib_C1C2 need a column gamma in obsbin_plus

liste_evt = np.unique(obsbin_plus.EVID.values)
if addI0:
    obsbin_plus = add_I0as_datapoint(obsbin_plus, liste_evt)
    obsbin_plus.sort_values(by=['EVID', 'I'], inplace=True)
    
obsbin_plus.loc[:, 'gamma'] = 0
beta_dataFR = pd.read_csv(beta_distrib_nameFR, sep=';', header=3)
beta_dataFR = beta_dataFR[beta_dataFR.Probability!=0]
beta_dataIT = pd.read_csv(beta_distrib_nameIT, sep=';', header=3)
beta_dataIT = beta_dataIT[beta_dataIT.Probability!=0]

compt = 0
coeffs = pd.DataFrame(columns=['C1_FR', 'C1_IT', 'C2', 'beta_FR', 'beta_IT', 'proba', 'ponderation'])
for indFR, rowFR in beta_dataFR.iterrows():
    # if indFR>1:
    #     break
    beta_FR = rowFR['Beta value']
    proba_FR = rowFR['Probability']
    print(beta_FR, proba_FR)
    obsbin_plus.loc[obsbin_plus.RegID==101, 'beta'] = beta_FR
    for indIT, rowIT in beta_dataIT.iterrows():
        # if indIT>1:
        #     break
        beta_IT = rowIT['Beta value']
        proba_IT = rowIT['Probability']
        print(beta_IT, proba_IT)
        obsbin_plus.loc[obsbin_plus.RegID==201, 'beta'] = beta_IT
        for ponderation, wp in zip(ponderation_list, poids_ponderation):
            obsbin_plus = add_Mweigths(obsbin_plus, ponderation)
            if invDepth_option:
                ObsBin_plus, resC1regC2 = calib_C1C2H(liste_evt, obsbin_plus, 0, 0, NmaxIter=50)
                coeffs.loc[compt, :] = [resC1regC2[0][0], resC1regC2[0][1], resC1regC2[0][2], beta_FR, beta_IT, proba_FR*proba_IT*wp, ponderation]
                compt += 1
            else:
                # This function need a column beta in obsbin_plus
                ObsBin_plus, resC1regC2 = calib_C1C2(liste_evt, obsbin_plus, 0, 0, NmaxIter=50)
                #print(resC1regC2)
                print('C1_FR', 'C1_IT', 'C2')
                print(resC1regC2[0][0], resC1regC2[0][1], resC1regC2[0][2])
                coeffs.loc[compt, :] = [resC1regC2[0][0], resC1regC2[0][1], resC1regC2[0][2], beta_FR, beta_IT, proba_FR*proba_IT*wp, ponderation]
                compt += 1
            #print(proba*wp, C1, C2, beta)
print(coeffs)