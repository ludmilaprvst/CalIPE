# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:43:48 2021

@author: PROVOST-LUD
"""
import sys
sys.path.append('../../calib_fc')
import pandas as pd
from create_subsets import create_liste_subset, filter_by_nevt, check_duplicate, create_basicdb_criteria
from create_subsets import create_subsets

basic_db_name = '../../Data/FR_extended_01/FR_extended_01.txt'
obsdata_name = '../../Data/ObsData/ObsCalibration_Frextended.txt'
subset_folder = '../../Data'
criteria = create_basicdb_criteria(basic_db_name, obsdata_name,
                            binning_type='RAVG',
                            outputfolder='../../Data',
                            regiondata_name='',
                            ponderation='Ponderation evt-uniforme',
                            )

#%%
criteria = pd.read_excel('../../Data/subset_criteria_FR_extended_01.xlsx')
global_liste, criteres = create_liste_subset(criteria,
                                             year_inf=[1850, 1950],
                                             year_sup=[2020, 2006],
                                             QH=[['A'], ['A', 'B'], ['A', 'B', 'C'], ['A', 'B', 'C', 'E']],
                                             NClass=[2, 3, 4],
                                             Nobs=[10, 25, 50, 100, 150],
                                             Dc=[10, 25, 50])
print(len(global_liste))

new_liste_nmin, new_critere_nmin = filter_by_nevt(global_liste, criteres, nmin=10)
print(len(new_liste_nmin))
new_liste, new_critere = check_duplicate(new_liste_nmin, new_critere_nmin)
print(len(new_liste))

create_subsets(new_liste, new_critere, basic_db_name, folder=subset_folder)
