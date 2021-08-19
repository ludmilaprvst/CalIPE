# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:28:51 2021

@author: PROVOST-LUD
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('../calib_fc')
from prepa_data import prepare_input4calibration


obsdata_name = '../../Data/ObsData/ObsCalibration_Fr_Instru02_filtoutsiders.txt'
evtdata_name = '../../Data/FR_instru_01/input_evt_calib_FRinstru_sansAlsace2018.txt'
regiondata_name = '../../Data/Regions/region_FRinstru.txt'
output_folder = '../../Data/FR_instru_01/Regions_01'
ponderation = 'Ponderation evt-reg'

evtdata = pd.read_csv(evtdata_name, sep=';')
obsbin_plus = prepare_input4calibration(obsdata_name, evtdata_name, ponderation,
                                            regiondata_name, binning_type='RAVG')

obsbin_plus_gp = obsbin_plus.groupby('EVID')
obsbin_plus_mean = obsbin_plus_gp[['Depth', 'Hmin', 'Hmax', 'Mag', 'StdM', 'RegID']].mean()
obsbin_plus_mean.reset_index(inplace=True)
regid_liste = np.unique(obsbin_plus_mean.RegID.values)
#%%
for regid in regid_liste:
    head, basename = os.path.split(evtdata_name)
    basename = basename[:-4]
    output_name = output_folder + '/' + basename + '_RegID'  + str(int(regid)) + '.csv'
    list_evt_region = obsbin_plus_mean[obsbin_plus_mean.RegID==regid].EVID.values
    evt_region = evtdata[evtdata.EVID.isin(list_evt_region)]
    evt_region.to_csv(output_name, sep=';', index=False)