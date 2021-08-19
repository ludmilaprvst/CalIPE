# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:50:57 2021

@author: PROVOST-LUD
"""
import sys
sys.path.append('../calib_fc')
sys.path.append('../postprocessing_fc')

#from prepa_data import prepare_input4calibration
#from combinaison_calib import calib_attBeta_Kov
from attKov_onedataset import Kovbeta_onedataset, Kovbetagamma_onedataset
from create_subsets import same_values_2array
import numpy as np
import pandas as pd

import os
#import matplotlib.pyplot as plt

# FR_instu
obsdata_name = '../../Data/ObsData/ObsCalibration_Fr_Instru02_filtoutsiders.txt'
evtdata_name = 'input_evt_calib_FRinstru_sansAlsace2018.txt'
subset_folder = '../../Data/FR_instru_01/subsets_01'
#subset_folder = '../../Data/FR_instru_01/bootstrap_1evt_FRinstru'
evtcalib_folder = '../../Data/FR_instru_01/'
outputfolder = '../../Outputs/FR_instru_01/Subsets_01/Beta'

"""
# FR_extended
obsdata_name = '../../Data/ObsData/ObsCalibration_Frextended.txt'
evtcalib_folder = '../../Data/FR_extended_01/'
evtdata_name = 'FR_extended_01.txt'
#subset_folder = '../../Data/FR_extended_01/Subsets_01'
subset_folder =''
outputfolder = '../../Outputs/FR_extended_01/BetaGamma/basicdb'
"""
"""
# FR_extended run by regions
obsdata_name = '../../Data/ObsData/ObsCalibration_Frextended.txt'
evtcalib_folder = '../../Data/FR_extended_01/'
evtdata_name = 'FR_extended_01.txt'
subset_folder = '../../Data/FR_extended_01/Regions_01'
outputfolder = '../../Outputs/FR_extended_01/Regions_01'

# FR_extended bootstrap
obsdata_name = '../../Data/ObsData/ObsCalibration_Frextended.txt'
evtcalib_folder = '../../Data/FR_extended_01/'
evtdata_name = 'FR_extended_01.txt'
subset_folder = '../../Data/FR_extended_01/bootstrap_1evt'
outputfolder = '../../Outputs/FR_extended_01/bootstrap_1evt'
"""

#obsdata_name = '../Data/ObsCalibration_BS2006_PROV.txt'
#evtdata_name = 'BS2006_PROV.txt'
#data_folder = '../Data/BS2006/'
#subset_folder = ''


regiondata_name = '../../Data/Regions/region2_FRinstru.txt'
binning_type = 'RAVG'
#♠ponderation = 'Ponderation evt-reg'
ponderation_list = ['Ponderation evt-uniforme', 'Ponderation evt-reg']
#ponderation_list = ['Ponderation evt-uniforme']

option_gamma = False

liste_beta_ini = [-2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0]
#liste_gamma_ini = [0, -0.01, -0.001]
#liste_beta_ini = [-3.0, -3.5, -4.0]
#liste_beta_ini = [-3.5]

if not subset_folder == '':
    liste_subset = pd.read_excel(subset_folder +'/dataset_list.xlsx')
    liste_subset = liste_subset.Datasubsetname.values
    liste_subset = np.insert(liste_subset, 0, evtdata_name)
    for subset in liste_subset:
        if subset == evtdata_name:
            complete_subset = evtcalib_folder + subset
        else:
            complete_subset = evtcalib_folder + subset_folder + '/'+subset + '.csv'
        print(complete_subset)
        print(os.path.isfile(complete_subset))
        if not os.path.isfile(complete_subset):
            print('No ' + complete_subset + ' exists')
        else:
            basedataset = pd.read_csv(evtcalib_folder + evtdata_name, sep=';')
            subsetdataset = pd.read_csv(complete_subset, sep=';')
            same = same_values_2array(basedataset.EVID.values, subsetdataset.EVID.values)
            
            if same and complete_subset!= evtcalib_folder + evtdata_name:
                print(subset + ' has the same events as the base database')
                continue
            print(subset)
            for ponderation in ponderation_list:
                print(ponderation)
                #completeobsdataname = data_folder + obsdata_name
                if option_gamma:
                    pass
                else:
                    Kovbeta_onedataset(complete_subset, obsdata_name,
                                       outputfolder=outputfolder,
                                       liste_beta_ini=liste_beta_ini,
                                       ponderation=ponderation,
                                       binning_type=binning_type,
                                       regiondata_name=regiondata_name,
                                       NminIter=3, NmaxIter=50)
else:
    complete_subset = evtcalib_folder + evtdata_name
    for ponderation in ponderation_list:
        print(ponderation)
        print(complete_subset)
        #completeobsdataname = data_folder + obsdata_name
        if option_gamma:
            Kovbetagamma_onedataset(complete_subset, obsdata_name, outputfolder,
                                    liste_beta_ini, liste_gamma_ini, ponderation,
                                    binning_type, regiondata_name,
                                    NminIter=3, NmaxIter=50)
        else:
            Kovbeta_onedataset(complete_subset, obsdata_name,
                               outputfolder=outputfolder,
                               liste_beta_ini=liste_beta_ini,
                               ponderation=ponderation,
                               binning_type=binning_type,
                               regiondata_name=regiondata_name,
                               NminIter=3, NmaxIter=50)




"""
obsbin_plus = prepare_input4calibration(obsdata_name, evtdata_name, ponderation,
                                        regiondata_name, binning_type)

print(len(obsbin_plus))
#obsbin_plus = obsbin_plus[obsbin_plus.EVID.isin(liste_avt2006)]
#obsbin_plus = obsbin_plus[obsbin_plus.RegID.isin([102, 104])]
#obsbin_plus = obsbin_plus[obsbin_plus.RegID.isin([103, 102, 104])]
#obsbin_plus = obsbin_plus[obsbin_plus.RegID.isin([103])]
#obsbin_plus = obsbin_plus[obsbin_plus.RegID.isin([101])]
print(len(obsbin_plus))
#obsbin_plus = obsbin_plus[~obsbin_plus.EVID.isin([310037])]
#obsbin_plus = obsbin_plus[obsbin_plus.EVID.isin([250038, 880077, 640001, 310037,
#                                                 650501, 650505, 3534,
#                                                 2019111102, 740150, 740153,
#                                                 3527])]
#print(len(obsbin_plus))

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
if option_gamma == 'Zero':
    liste_evt = np.unique(obsbin_plus.EVID.values)
    for beta_ini in liste_beta_ini:
        print(beta_ini)
        repere_betaini = repere + str(int(beta_ini*-10))
        obsBin_plus_end, beta, cov_beta, suivi_beta = calib_attBeta_Kov(liste_evt, obsbin_plus, beta_ini, 
                                                              NminIter=3, NmaxIter=50, suivi_inversion=False,
                                                              dossier_suivi='../Outputs/suivi_inv_par_evt_ALPSPYREST_lesplusbeaux')
        obsBin_plus_end.to_csv('../Outputs/obsbin_end'+ repere_betaini+'.csv', index=False, sep=';')
        output_file = open('../Outputs/beta_final'+repere_betaini+'.txt', 'w')
        output_file.write('Beta:' +str(beta)+'\n')
        output_file.write('Beta ini:' +str(beta_ini)+'\n')
        output_file.write('Nbre iteration:' +str(len(suivi_beta)) + ' \n')
        output_file.write('Nbre iteration max:' +str(50) + ' \n')
        output_file.write('Nbre iteration min:' +str(3) + ' \n')
        output_file.write('Nbre evt:' +str(len(liste_evt))+'\n')
        output_file.write('Nbre data:' +str(len(obsBin_plus_end))+'\n')
        output_file.write('Fichier Evt:' +evtdata_name+'\n')
        output_file.write('Fichier Obs:' +obsdata_name+'\n')
    
        output_file.write('Nom fichier execute:' +sys.argv[0]+'\n')
        output_file.close()
        ax.plot(range(len(suivi_beta)), suivi_beta, label=str(round(beta_ini, 2)))
    ax.set_xlabel("# iteration")
    ax.set_ylabel("Beta value")
    ax.grid(which='both')
    ax.legend()
    fig.savefig('../Outputs/suivi_beta_pondevtuni_plusieursbetaini.png', dpi=150)
"""
