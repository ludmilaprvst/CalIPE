# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:43:48 2021

@author: PROVOST-LUD
"""
import sys
sys.path.append('../calib_fc')
import pandas as pd
from create_subsets import create_liste_subset, filter_by_nevt, check_duplicate, create_basicdb_criteria
from create_subsets import create_subsets



"""
Create a table with the metadata used to create the subsets. Excel file can be created/modified manually:
"""
# basic_db_name = '../../Data/FR_extended_01/FR_extended_01.txt'
# obsdata_name = '../../Data/ObsData/ObsCalibration_Frextended.txt'
# subset_folder = '../../Data'
# criteria = create_basicdb_criteria(basic_db_name, obsdata_name,
#                             binning_type='RAVG',
#                             outputfolder='../../Data',
#                             regiondata_name='',
#                             ponderation='Ponderation evt-uniforme',
#                             )

#%% FRextended
basic_db_name = '../../Data/FR_extended_01/FR_extended_01.txt'
obsdata_name = '../../Data/ObsData/ObsCalibration_Frextended_filtered.txt'
subset_folder = '../../Data/FR_extended_01/Subset_03'

criteria = pd.read_excel('../../Data/subset_criteria_FR_extended_01.xlsx')
global_liste, criteres = create_liste_subset(criteria,
                                             year_inf=[1850, 1921, 1980],
                                             year_sup=[2020, 2006],
                                             QH=[['A'], ['A', 'B'], ['A', 'B', 'C'], ['A', 'B', 'C', 'E']],
                                             NClass=[2, 3, 4, 5, 6],
                                             Nobs=[10, 50, 100, 200],
                                             Dc=[10, 25, 50, 100])

print(len(global_liste))

new_liste_nmin, new_critere_nmin = filter_by_nevt(global_liste, criteres, nmin=10)
print(len(new_liste_nmin))
new_liste, new_critere = check_duplicate(new_liste_nmin, new_critere_nmin)
print(len(new_liste))

create_subsets(new_liste, new_critere, basic_db_name, folder=subset_folder)

"""
#%% ITinstru
basic_db_name = '../../Data/IT_instru01/IT_instru_01c.txt'
obsdata_name = '../../Data/ObsData/obs_IT.txt'
subset_folder = '../../Data/IT_instru01/Subsets_01'

criteria = pd.read_excel('../../Data/IT_instru01/Subsets_01/subset_criteria_IT_instru_01c.xlsx')
global_liste, criteres = create_liste_subset(criteria,
                                             year_inf=[1970],
                                             year_sup=[2012],
                                             QH=[['A', 'B', 'C'], ['A','B']],
                                             NClass=[4, 5, 6],
                                             Nobs=[10, 50, 100, 200],
                                             Dc=[30, 60, 100, 500])
print(len(global_liste))

new_liste_nmin, new_critere_nmin = filter_by_nevt(global_liste, criteres, nmin=10)
print(len(new_liste_nmin))
new_liste, new_critere = check_duplicate(new_liste_nmin, new_critere_nmin)
print(len(new_liste))

create_subsets(new_liste, new_critere, basic_db_name, folder=subset_folder)
"""
"""
#%% FRinstru
basic_db_name = '../../Data/FR_instru_01/input_evt_calib_FRinstru01.txt'
obsdata_name = '../../Data/ObsData/ObsCalibration_Frextended_filtered.txt'
subset_folder = '../../Data/FR_instru_01/Subsets_02b'

criteria = pd.read_excel('../../Data/FR_instru_01/subset_criteria_input_evt_calib_FRinstru01.xlsx')
global_liste, criteres = create_liste_subset(criteria,
                                             year_inf=[1980],
                                             year_sup=[2020, 2006],
                                             QH=[['A','B'], ['A']],
                                             NClass=[2, 3, 4, 5, 6],
                                             Nobs=[10, 50, 100, 200],
                                             Dc=[10, 25, 50, 100])
print(len(global_liste))

new_liste_nmin, new_critere_nmin = filter_by_nevt(global_liste, criteres, nmin=10)
print(len(new_liste_nmin))
new_liste, new_critere = check_duplicate(new_liste_nmin, new_critere_nmin)
print(len(new_liste))

create_subsets(new_liste, new_critere, basic_db_name, folder=subset_folder)
"""