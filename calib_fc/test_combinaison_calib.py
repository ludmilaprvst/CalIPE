# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:52:57 2021

@author: PROVOST-LUD
"""
from combinaison_calib import calib_attBeta_Kov_unit, calib_attBeta_Kov
from ponderations import evt_weights
import pytest
import pandas as pd
import numpy as np

def test_calib_attBeta_Kov_unit():
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_coeff.txt')
    
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth_ok'] = depth
#        obs_data.loc[obs_data.EVID==evid, 'Hmin'] = evt_data[evt_data.EVID==evid].Hinf.values[0]
#        obs_data.loc[obs_data.EVID==evid, 'Hmax'] = evt_data[evt_data.EVID==evid].Hsup.values[0]
    
    beta_ini = -3.5
    obs_data.loc[:, 'eqStd'] = obs_data.StdI.values
    obs_data.loc[:, 'Depth'] = 10
    obs_data.loc[:, 'Hmin'] = 1
    obs_data.loc[:, 'Hmax'] = 25
    obs_data.loc[:, 'Io_std'] = 0.5
    obs_data.loc[:, 'Io_ini'] = obs_data.Io.values
    
    obs_data = evt_weights(obs_data, 'Ponderation dI')
    print(obs_data.columns)
    obs_data_inv, beta_inv = calib_attBeta_Kov_unit(liste_evt, obs_data, beta_ini)
    assert beta_inv == pytest.approx(Beta, 0.01)
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_inv = obs_data_inv[obs_data_inv.EVID==evid].Depth.values[0]
        I0 = obs_data_inv[obs_data_inv.EVID==evid].Io_ini.values[0]
        I0_inv = obs_data_inv[obs_data_inv.EVID==evid].Io.values[0]
        assert np.round(depth, 1) == np.round(depth_inv, 1)
        assert I0 == pytest.approx(I0_inv, 0.001)
        

        
def test_calib_attBeta_Kov():
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_coeff.txt')
    
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth_ok'] = depth
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
        obs_data.loc[obs_data.EVID==evid, 'Hmin'] = evt_data[evt_data.EVID==evid].Hinf.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Hmax'] = evt_data[evt_data.EVID==evid].Hsup.values[0]
    obs_data.loc[:, 'Io_std'] = 0.5
    obs_data.loc[:, 'Io_ini'] = obs_data.Io.values
    obs_data.loc[:, 'eqStd'] = obs_data.StdI.values
    # test 01 : bon beta
    beta_ini = -3.5
    obs_data_inv, beta_inv, suivi_beta = calib_attBeta_Kov(liste_evt, obs_data, beta_ini)
    assert beta_inv == pytest.approx(Beta, 0.01)
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_inv = obs_data_inv[obs_data_inv.EVID==evid].Depth.values[0]
        I0 = obs_data_inv[obs_data_inv.EVID==evid].Io_ini.values[0]
        I0_inv = obs_data_inv[obs_data_inv.EVID==evid].Io.values[0]
        assert np.round(depth, 1) == np.round(depth_inv, 1)
        assert I0 == pytest.approx(I0_inv, 0.001)
        
    # test 02 : bon beta
    beta_ini = -3.5
    obs_data.loc[:, 'Hmin'] = 1
    obs_data.loc[:, 'Hmax'] = 25
    obs_data.loc[:, 'Io'] = obs_data.loc[:, 'Io_ini']
    obs_data_inv, beta_inv, suivi_beta = calib_attBeta_Kov(liste_evt, obs_data, beta_ini)
    assert beta_inv == pytest.approx(Beta, 0.001)
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_inv = obs_data_inv[obs_data_inv.EVID==evid].Depth.values[0]
        I0 = obs_data_inv[obs_data_inv.EVID==evid].Io_ini.values[0]
        I0_inv = obs_data_inv[obs_data_inv.EVID==evid].Io.values[0]
        assert np.round(depth, 1) == np.round(depth_inv, 1)
        assert I0 == pytest.approx(I0_inv, 0.001)
        
    # test 03 : pas bon beta
    beta_ini = -3.0
    # reinitialisation
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
        obs_data.loc[obs_data.EVID==evid, 'Hmin'] = evt_data[evt_data.EVID==evid].Hinf.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Hmax'] = evt_data[evt_data.EVID==evid].Hsup.values[0]
    obs_data.loc[:, 'Io'] = obs_data.loc[:, 'Io_ini']
    
    obs_data_inv, beta_inv, suivi_beta = calib_attBeta_Kov(liste_evt, obs_data, beta_ini)
    assert beta_inv == pytest.approx(Beta, 0.01)
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_inv = obs_data_inv[obs_data_inv.EVID==evid].Depth.values[0]
        I0 = obs_data_inv[obs_data_inv.EVID==evid].Io_ini.values[0]
        I0_inv = obs_data_inv[obs_data_inv.EVID==evid].Io.values[0]
        assert np.round(depth, 1) == pytest.approx(depth_inv, 0.5)
        assert I0 == pytest.approx(I0_inv, 0.01)
    
    beta_ini = -4.0
    # reinitialisation
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
        obs_data.loc[obs_data.EVID==evid, 'Hmin'] = evt_data[evt_data.EVID==evid].Hinf.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Hmax'] = evt_data[evt_data.EVID==evid].Hsup.values[0]
    obs_data.loc[:, 'Io'] = obs_data.loc[:, 'Io_ini']
    
    obs_data_inv, beta_inv, suivi_beta = calib_attBeta_Kov(liste_evt, obs_data, beta_ini)
    assert beta_inv == pytest.approx(Beta, 0.01)
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_inv = obs_data_inv[obs_data_inv.EVID==evid].Depth.values[0]
        I0 = obs_data_inv[obs_data_inv.EVID==evid].Io_ini.values[0]
        I0_inv = obs_data_inv[obs_data_inv.EVID==evid].Io.values[0]
        assert np.round(depth, 1) == pytest.approx(depth_inv, 0.5)
        assert I0 == pytest.approx(I0_inv, 0.01)
        
    beta_ini = -2.0
    # reinitialisation
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
        obs_data.loc[obs_data.EVID==evid, 'Hmin'] = evt_data[evt_data.EVID==evid].Hinf.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Hmax'] = evt_data[evt_data.EVID==evid].Hsup.values[0]
    obs_data.loc[:, 'Io'] = obs_data.loc[:, 'Io_ini']
    
    obs_data_inv, beta_inv, suivi_beta = calib_attBeta_Kov(liste_evt, obs_data, beta_ini)
    assert beta_inv == pytest.approx(Beta, 0.1)
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_inv = obs_data_inv[obs_data_inv.EVID==evid].Depth.values[0]
        I0 = obs_data_inv[obs_data_inv.EVID==evid].Io_ini.values[0]
        I0_inv = obs_data_inv[obs_data_inv.EVID==evid].Io.values[0]
        assert np.round(depth, 1) == pytest.approx(depth_inv, 0.5)
        assert I0 == pytest.approx(I0_inv, 0.1)
    
    beta_ini = -5.0
    # reinitialisation
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
        obs_data.loc[obs_data.EVID==evid, 'Hmin'] = evt_data[evt_data.EVID==evid].Hinf.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Hmax'] = evt_data[evt_data.EVID==evid].Hsup.values[0]
    obs_data.loc[:, 'Io'] = obs_data.loc[:, 'Io_ini']
    
    obs_data_inv, beta_inv, suivi_beta = calib_attBeta_Kov(liste_evt, obs_data, beta_ini)
    assert beta_inv == pytest.approx(Beta, 0.1)
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_inv = obs_data_inv[obs_data_inv.EVID==evid].Depth.values[0]
        I0 = obs_data_inv[obs_data_inv.EVID==evid].Io_ini.values[0]
        I0_inv = obs_data_inv[obs_data_inv.EVID==evid].Io.values[0]
        assert np.round(depth, 1) == pytest.approx(depth_inv, 0.5)
        assert I0 == pytest.approx(I0_inv, 0.1)
#    # test 03 : pas bon beta
#    beta_ini = -2.0
#    for evid in liste_evt:
#        depth = evt_data[evt_data.EVID==evid].H.values[0]
#        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
#        obs_data.loc[obs_data.EVID==evid, 'Hmin'] = evt_data[evt_data.EVID==evid].Hinf.values[0]
#        obs_data.loc[obs_data.EVID==evid, 'Hmax'] = evt_data[evt_data.EVID==evid].Hsup.values[0]
#    obs_data_inv, beta_inv = calib_attBeta_Kov(liste_evt, obs_data, beta_ini)
#    assert beta_inv == pytest.approx(Beta, 0.01)
#    for evid in liste_evt:
#        depth = evt_data[evt_data.EVID==evid].H.values[0]
#        depth_inv = obs_data_inv[obs_data_inv.EVID==evid].Depth.values[0]
#        I0 = obs_data_inv[obs_data_inv.EVID==evid].Io.values[0]
#        I0_inv = obs_data_inv[obs_data_inv.EVID==evid].Io_inv.values[0]
#        assert np.round(depth, 1) == np.round(depth_inv, 1)
#        assert I0 == pytest.approx(I0_inv, 0.001)
    
