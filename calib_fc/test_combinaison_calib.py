# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:52:57 2021

@author: PROVOST-LUD
"""
import sys
sys.path.append('../postprocessing_fc')
from combinaison_calib import calib_attBeta_Kov_unit, calib_attBeta_Kov
from combinaison_calib import calib_attBetaGamma_Kov_unit, calib_attBetaGamma_Kov
from combinaison_calib import initialize_HI0, initialize_C1C2
from combinaison_calib import calib_C1C2H, calib_C1C2

from ponderations import evt_weights
import pytest
import pandas as pd
import numpy as np


def test_calib_C1C2H():
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_coeff.txt')
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    C1 = coeff_data['C1'].values[0]
    C2 = coeff_data['C2'].values[0]
    obs_data.loc[:, 'Io_std'] = 0.5
    obs_data.loc[:, 'eqStdM'] = 0.5
    obs_data.loc[:, 'Io_ini'] = obs_data.Io.values
    obs_data.loc[:, 'RegID'] = 1
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        mag = evt_data[evt_data.EVID==evid].Mag.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = 10
        obs_data.loc[obs_data.EVID==evid, 'Hmin'] = 1
        obs_data.loc[obs_data.EVID==evid, 'Hmax'] = 25
        obs_data.loc[obs_data.EVID==evid, 'Mag'] = mag
    obs_data.loc[:, 'beta'] = Beta
    obs_data.loc[:, 'gamma'] = 0
    obs_data.loc[:, 'Hmin_ini'] = 1
    obs_data.loc[:, 'Hmax_ini'] = 25
    ObsBin_plus, resC1regC2 = calib_C1C2H(liste_evt, obs_data, NmaxIter=50)
    print('C1', 'C2')
    print(resC1regC2[0][0], resC1regC2[0][1])
    assert resC1regC2[0][0] == pytest.approx(C1, 0.001)
    assert resC1regC2[0][1] == pytest.approx(C2, 0.001)
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        print(obs_data.columns)
        print(depth, obs_data[obs_data.EVID==evid]['Depth'].values[0])
        assert obs_data[obs_data.EVID==evid]['Depth'].values[0] == pytest.approx(depth, 0.001)
    ObsBin_plus, resC1regC2 = calib_C1C2H(liste_evt, obs_data, NmaxIter=49)
    print('C1', 'C2')
    print(resC1regC2[0][0], resC1regC2[0][1])
    assert resC1regC2[0][0] == pytest.approx(C1, 0.001)
    assert resC1regC2[0][1] == pytest.approx(C2, 0.001)

def test_calib_C1C2():
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_coeff.txt')
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    C1 = coeff_data['C1'].values[0]
    C2 = coeff_data['C2'].values[0]
    obs_data.loc[:, 'Io_std'] = 0.5
    obs_data.loc[:, 'eqStdM'] = 0.5
    obs_data.loc[:, 'Io_ini'] = obs_data.Io.values
    obs_data.loc[:, 'RegID'] = 1
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        mag = evt_data[evt_data.EVID==evid].Mag.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
        obs_data.loc[obs_data.EVID==evid, 'Hmin'] = 1
        obs_data.loc[obs_data.EVID==evid, 'Hmax'] = 25
        obs_data.loc[obs_data.EVID==evid, 'Mag'] = mag
    obs_data.loc[:, 'beta'] = Beta
    obs_data.loc[:, 'gamma'] = 0
    ObsBin_plus, resC1regC2 = calib_C1C2(liste_evt, obs_data, NmaxIter=50)
    print('C1', 'C2')
    print(resC1regC2[0][0], resC1regC2[0][1])
    assert resC1regC2[0][0] == pytest.approx(C1, 0.001)
    assert resC1regC2[0][1] == pytest.approx(C2, 0.001)
 
 
def test_initialize_C1C2():
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_coeff.txt')
    
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    C1 = coeff_data['C1'].values[0]
    C2 = coeff_data['C2'].values[0]
    obs_data.loc[:, 'Io_std'] = 0.5
    obs_data.loc[:, 'eqStdM'] = 0.5
    obs_data.loc[:, 'Io_ini'] = obs_data.Io.values
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        mag = evt_data[evt_data.EVID==evid].Mag.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
 
        obs_data.loc[obs_data.EVID==evid, 'Mag'] = mag
    obs_data.loc[:, 'beta'] = Beta
    obs_data.loc[:, 'gamma'] = 0

    C1_inv, C2_inv = initialize_C1C2(obs_data)
    assert C1_inv == pytest.approx(C1, 0.001)
    assert C2_inv == pytest.approx(C2, 0.001)
    
def test_initialize_HI0():
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_coeff.txt')
    
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    obs_data.loc[:, 'Depth'] = 10
    obs_data.loc[:, 'Hmin'] = 1
    obs_data.loc[:, 'Hmax'] = 25
    obs_data.loc[:, 'Io_std'] = 0.5
    obs_data.loc[:, 'Io_ini'] = obs_data.Io.values
    
    obs_data_inv = initialize_HI0(obs_data, liste_evt, Beta, 0, NmaxIter=50)
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_inv = obs_data_inv[obs_data_inv.EVID==evid].Depth.values[0]
        I0 = obs_data_inv[obs_data_inv.EVID==evid].Io_ini.values[0]
        I0_inv = obs_data_inv[obs_data_inv.EVID==evid].Io.values[0]
        assert np.round(depth, 1) == pytest.approx(depth_inv, 0.5)
        assert I0 == pytest.approx(I0_inv, 0.1)

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
        
        
def test_calib_attBetaGamma_Kov_unit():
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset01_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset01_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset01_coeff.txt')
    
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    Gamma = coeff_data['Gamma'].values[0]
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth_ok'] = depth
#        obs_data.loc[obs_data.EVID==evid, 'Hmin'] = evt_data[evt_data.EVID==evid].Hinf.values[0]
#        obs_data.loc[obs_data.EVID==evid, 'Hmax'] = evt_data[evt_data.EVID==evid].Hsup.values[0]
    
    beta_ini = -3.5
    gamma_ini = -0.001
    obs_data.loc[:, 'eqStd'] = obs_data.StdI.values
    obs_data.loc[:, 'Depth'] = 10
    obs_data.loc[:, 'Hmin'] = 1
    obs_data.loc[:, 'Hmax'] = 25
    obs_data.loc[:, 'Io_std'] = 0.5
    obs_data.loc[:, 'Io_ini'] = obs_data.Io.values
    
    obs_data = evt_weights(obs_data, 'Ponderation dI')
    print(obs_data.columns)
    obs_data_inv, beta_inv, gamma_inv = calib_attBetaGamma_Kov_unit(liste_evt, obs_data, beta_ini, gamma_ini)
    assert beta_inv == pytest.approx(Beta, 0.01)
    assert gamma_inv == pytest.approx(Gamma, 0.0001)
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_inv = obs_data_inv[obs_data_inv.EVID==evid].Depth.values[0]
        I0 = obs_data_inv[obs_data_inv.EVID==evid].Io_ini.values[0]
        I0_inv = obs_data_inv[obs_data_inv.EVID==evid].Io.values[0]
        assert np.round(depth, 1) == np.round(depth_inv, 1)
        assert I0 == pytest.approx(I0_inv, 0.001)
        
    beta_ini = -3.5
    gamma_ini = -0.0011
    obs_data.loc[:, 'eqStd'] = obs_data.StdI.values
    obs_data.loc[:, 'Depth'] = obs_data.loc[:, 'Depth_ok']
    obs_data.loc[:, 'Hmin'] = 1
    obs_data.loc[:, 'Hmax'] = 25
    obs_data.loc[:, 'Io_std'] = 0.5
    obs_data.loc[:, 'Io_ini'] = obs_data.Io.values
    
    obs_data = evt_weights(obs_data, 'Ponderation dI')
    print(obs_data.columns)
    obs_data_inv, beta_inv, gamma_inv = calib_attBetaGamma_Kov_unit(liste_evt, obs_data, beta_ini, gamma_ini)
    assert beta_inv == pytest.approx(Beta, 0.01)
    assert gamma_inv == pytest.approx(Gamma, 0.01)
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_inv = obs_data_inv[obs_data_inv.EVID==evid].Depth.values[0]
        I0 = obs_data_inv[obs_data_inv.EVID==evid].Io_ini.values[0]
        I0_inv = obs_data_inv[obs_data_inv.EVID==evid].Io.values[0]
        assert depth == pytest.approx(depth_inv, 0.5)
        assert I0 == pytest.approx(I0_inv, 0.001)
      
def test_calib_attBetaGamma_Kov():
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset01_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset01_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset01_coeff.txt')
    
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    Gamma = coeff_data['Gamma'].values[0]
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth_ok'] = depth
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
        obs_data.loc[obs_data.EVID==evid, 'Hmin'] = evt_data[evt_data.EVID==evid].Hinf.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Hmax'] = evt_data[evt_data.EVID==evid].Hsup.values[0]
    obs_data.loc[:, 'Io_std'] = 0.5
    obs_data.loc[:, 'Io_ini'] = obs_data.Io.values
    obs_data.loc[:, 'eqStd'] = obs_data.StdI.values
    # test 01 : bon beta et bon gamma
    beta_ini = -3.5
    gamma_ini = -0.001
    obs_data_inv, beta_inv, gamma_inv, cov_betagamma, suivi_beta, suivi_gamma = calib_attBetaGamma_Kov(liste_evt,
                                                                                              obs_data,
                                                                                              beta_ini,
                                                                                              gamma_ini)
    assert beta_inv == pytest.approx(Beta, 0.01)
    assert gamma_inv == pytest.approx(Gamma, 0.01)
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_inv = obs_data_inv[obs_data_inv.EVID==evid].Depth.values[0]
        I0 = obs_data_inv[obs_data_inv.EVID==evid].Io_ini.values[0]
        I0_inv = obs_data_inv[obs_data_inv.EVID==evid].Io.values[0]
        assert depth == pytest.approx(depth_inv, 0.5)
        assert I0 == pytest.approx(I0_inv, 0.001)
        
    
    # test 02 : bon beta
    beta_ini = -3.5
    gamma_ini = -0.001
    obs_data.loc[:, 'Hmin'] = 1
    obs_data.loc[:, 'Hmax'] = 25
    obs_data.loc[:, 'Io'] = obs_data.loc[:, 'Io_ini']
    obs_data_inv, beta_inv, gamma_inv, cov_betagamma, suivi_beta, suivi_gamma = calib_attBetaGamma_Kov(liste_evt,
                                                                                              obs_data,
                                                                                              beta_ini,
                                                                                              gamma_ini)
    assert beta_inv == pytest.approx(Beta, 0.001)
    assert gamma_inv == pytest.approx(Gamma, 0.01)
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_inv = obs_data_inv[obs_data_inv.EVID==evid].Depth.values[0]
        I0 = obs_data_inv[obs_data_inv.EVID==evid].Io_ini.values[0]
        I0_inv = obs_data_inv[obs_data_inv.EVID==evid].Io.values[0]
        assert depth == pytest.approx(depth_inv, 0.5)
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
    obs_data_inv, beta_inv, cov_beta, suivi_beta = calib_attBeta_Kov(liste_evt, obs_data, beta_ini)
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
    obs_data_inv, beta_inv, cov_beta, suivi_beta = calib_attBeta_Kov(liste_evt, obs_data, beta_ini)
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
    
    obs_data_inv, beta_inv, cov_beta, suivi_beta = calib_attBeta_Kov(liste_evt, obs_data, beta_ini)
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
    
    obs_data_inv, beta_inv, cov_beta, suivi_beta = calib_attBeta_Kov(liste_evt, obs_data, beta_ini)
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
    
    obs_data_inv, beta_inv, cov_beta, suivi_beta = calib_attBeta_Kov(liste_evt, obs_data, beta_ini)
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
    
    obs_data_inv, beta_inv, cov_beta, suivi_beta = calib_attBeta_Kov(liste_evt, obs_data, beta_ini)
    assert beta_inv == pytest.approx(Beta, 0.1)
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_inv = obs_data_inv[obs_data_inv.EVID==evid].Depth.values[0]
        I0 = obs_data_inv[obs_data_inv.EVID==evid].Io_ini.values[0]
        I0_inv = obs_data_inv[obs_data_inv.EVID==evid].Io.values[0]
        assert np.round(depth, 1) == pytest.approx(depth_inv, 0.5)
        assert I0 == pytest.approx(I0_inv, 0.1)



    

    