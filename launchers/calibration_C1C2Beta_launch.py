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
from combinaison_calib import calib_C1C2betaH
from prepa_data import prepare_input4calibration, add_Mweigths

# FR_instu
obsdata_name = '../../Data/ObsData/ObsCalibration_Frextended_filtered.txt'
evtdata_name = 'input_evt_calib_FRinstru01.txt'
evtcalib_folder = '../../Data/FR_instru_01/'
outputfolder = '../../Outputs/FR_instru_01/IPE/OneStep'
regiondata_name = '../../Data/Regions/region_FRIT.txt'


dummy_ponderation = 'Ponderation dI'
binning_type = 'ROBS'
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

beta_liste = [-2.5, -3, -3.5]
#beta_liste = [-3.5]
C1_liste = [1.2, 1.7, 2.2, 2.7]
C2_liste = [1.35]

output_df = pd.DataFrame(columns=['beta_ini', 'C1_ini', 'C2_ini', 'ponderation', 'C1', 'C2', 'beta','std_C1', 'std_C2', 'std_beta'])
ind = 0
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
                                                               NmaxIter=100)
                    
                    pcov = result[1]
                    std = np.sqrt(np.diag(pcov))
                    output_df.loc[ind, :] = [beta, C1, C2, ponderation, result[0][0], result[0][1], result[0][2],
                                             std[0], std[1], std[2]]
                    ind += 1
                    #print(result)
                    print(result[0][:3])
                    # print(result[0][3:])
                    
                    # print(std[:3])
                    
output_df.to_excel(outputfolder + '/FRinstru01_diffvaliniC1beta.xlsx')