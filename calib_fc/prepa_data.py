# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:52:50 2021

@author: PROVOST-LUD
"""

import pandas as pd
import EvtIntensityObject as eio
from ponderations import evt_weights, attribute_region

class fichier_input:
    def __init__(self, obsdata, evtdata):
        self.EvtFile = evtdata
        self.ObsFile = obsdata

def prepare_input4calibration(obsdata_name, evtdata_name, ponderation,
                              regiondata_name='',
                              binning_type = 'RAVG'):
    obsdata = pd.read_csv(obsdata_name, sep=';')
    evtdata = pd.read_csv(evtdata_name, sep=';')
    
    fichiers = fichier_input(obsdata, evtdata)
    data = eio.Evt(fichiers)
    
    columns_obsbinplus = ['EVID', 'I', 'StdI', 'Io', 'Io_std', 'Io_ini', 'Depi','Ndata', 'Mag',
                                        'StdM', 'Depth', 'Hmin', 'Hmax']
    obsbin_plus = pd.DataFrame(columns=columns_obsbinplus)
    for evid in evtdata.EVID.values:
        # Attribuer un Depi aux Iobs
        data.build(int(evid))
        data.I0 = data.Io_ini
        #print(evid)
        #print(data.I0)
        #print(data.Obsevid.head())
        # Bin des Iobs
        data.Binning_Obs(0, data.Ic, method_bin=binning_type)
        #print(data.ObsBinn)
        evt_obsbin = data.ObsBinn
        evt_obsbin.loc[:, 'Depth'] = data.depth
        evt_obsbin.loc[:, 'Hmin'] = data.Hmin
        evt_obsbin.loc[:, 'Hmax'] = data.Hmax
        evt_obsbin.loc[:, 'Mag'] = data.Mag
        evt_obsbin.loc[:, 'StdM'] = data.StdM
        evt_obsbin.loc[:, 'Io_ini'] = data.I0
        hypo_tmp = np.sqrt(evt_obsbin.Depi.values**2 + depth**2)
        X_tmp = evt_obsbin.I.values - Beta*np.log10(hypo_tmp) - Gamma*hypo_tmp
        evt_obsbin.loc[:, 'X'] = np.average(X_tmp, weights=1/evt_obsbin.StdI.values**2)
#        evt_obsbin.loc[:, 'Io'] = data.I0
        evt_obsbin = evt_obsbin[columns_obsbinplus]
        obsbin_plus = obsbin_plus.append(evt_obsbin)
    if regiondata_name != '':
        obsbin_plus = attribute_region(evtdata, obsbin_plus, regiondata_name)
    else:
        obsbin_plus.loc[:, 'RegID'] = -99
    obsbin_plus = evt_weights(obsbin_plus, ponderation)
    return obsbin_plus

def prepare_data_C1C2calib(obsbin_plus):
    for evid in evtdata.EVID.values:

#obsdata_name = '../Data/ObsCalibration_Fr_Instru_filtoutsiders.txt'
#evtdata_name = '../Data/input_evt_calib_FRinstru_sansAls2018Blay2018.txt'
#regiondata_name = '../Data/region_FRinstru.txt'
#binning_type = 'RAVG'
#ponderation = 'Ponderation dI'