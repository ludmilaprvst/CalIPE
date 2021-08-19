# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:43:48 2021

@author: PROVOST-LUD
"""

import pandas as pd
from create_subsets import create_bootstrap_notrandom
from create_subsets import create_subsets

basic_db_name = '../../Data/FR_extended_01/FR_extended_01.txt'
obsdata_name = '../../Data/DataObs/ObsCalibration_Frextended.txt'
subset_folder = '../../Data/FR_extended_01'
#criteria = create_basicdb_criteria(basic_db_name, obsdata_name,
#                            binning_type='RAVG',
#                            outputfolder='',
#                            regiondata_name='',
#                            ponderation='Ponderation evt-uniforme',
#                            )
basic_db = pd.read_csv(basic_db_name, sep=';')

global_liste, criteres = create_bootstrap_notrandom(basic_db.EVID.values)
print(len(global_liste))


create_subsets(global_liste, criteres, basic_db_name, column_criteria=['Deleted event id'],
               folder=subset_folder, basename='bootstrap_1evt')
