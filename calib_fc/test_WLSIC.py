# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:54:31 2021

@author: PROVOST-LUD
"""
import pytest
import WLSIC
import pandas as pd
import numpy as np
import time
import random

def test_Kov_do_wlsic_I0():
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_coeff.txt')
    
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    for evid in liste_evt:
        obsbin = obs_data[obs_data.EVID==evid]
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        I0 = obsbin.Io.values[0]
        resI0 = WLSIC.WLSIC_Kov_oneEvt(obsbin, depth, Beta, 0, I0).do_wlsic_I0(I0-1, I0+1)
        assert np.round(resI0[0][0], 3) == np.round(I0, 3)
    
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset04_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset04_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset04_coeff.txt')
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    Gamma = coeff_data['Gamma'].values[0]
    for evid in liste_evt:
        obsbin = obs_data[obs_data.EVID==evid]
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        I0 = obsbin.Io.values[0]
        I0_ini = I0-0.5
        resI0 = WLSIC.WLSIC_Kov_oneEvt(obsbin, depth, Beta, Gamma, I0_ini).do_wlsic_I0(I0-1, I0+1)
        assert resI0[0][0] == pytest.approx(I0, 0.001)
    
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset01_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset01_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset01_coeff.txt')
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    Gamma = coeff_data['Gamma'].values[0]
    for evid in liste_evt:
        obsbin = obs_data[obs_data.EVID==evid]
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        I0 = obsbin.Io.values[0]
        I0_ini = I0-0.5
        resI0 = WLSIC.WLSIC_Kov_oneEvt(obsbin, depth, Beta, Gamma, I0_ini).do_wlsic_I0(I0-1, I0+1)
        assert resI0[0][0] == pytest.approx(I0, 0.001)
        
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset02_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset02_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset02_coeff.txt')
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    Gamma = coeff_data['Gamma'].values[0]
    for evid in liste_evt:
        obsbin = obs_data[obs_data.EVID==evid]
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        I0 = obsbin.Io.values[0]
        I0_ini = I0-0.5
        resI0 = WLSIC.WLSIC_Kov_oneEvt(obsbin, depth, Beta, Gamma, I0_ini).do_wlsic_I0(I0-1, I0+1)
        assert resI0[0][0] == pytest.approx(I0, 0.001)
        


def test_Kov_do_wlsic_depth():
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_coeff.txt')
    
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    for evid in liste_evt:
        obsbin = obs_data[obs_data.EVID==evid]
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_ini = 10
        obsbin.loc[:, 'Hypo'] = np.sqrt(obsbin['Depi'].values**2 + depth_ini**2)
        Hmin = 1
        Hmax = 20
        I0 = obsbin.Io.values[0]
        resH = WLSIC.WLSIC_Kov_oneEvt(obsbin, depth_ini, Beta, 0, I0).do_wlsic_depth(Hmin, Hmax)
        assert np.round(resH[0][0], 3) == np.round(depth, 3)
        
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset04_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset04_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset04_coeff.txt')
    
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    Gamma = coeff_data['Gamma'].values[0]
    for evid in liste_evt:
        obsbin = obs_data[obs_data.EVID==evid]
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        depth_ini = 10
        obsbin.loc[:, 'Hypo'] = np.sqrt(obsbin['Depi'].values**2 + depth_ini**2)
        I0 = obsbin.Io.values[0]
        Hmin = 1
        Hmax = 20
        resH = WLSIC.WLSIC_Kov_oneEvt(obsbin, depth_ini, Beta, Gamma, I0).do_wlsic_depth(Hmin, Hmax)
        assert np.round(resH[0][0], 1) == np.round(depth, 1)
        

def test_Kov_do_wls_beta():        
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset03_coeff.txt')
    
    obs_data.loc[:, 'eqStd'] = obs_data.loc[:, 'StdI']
    
    liste_evt = evt_data.EVID.values
    Beta = coeff_data['Beta'].values[0]
    beta_ini = -3
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
            
    resBeta = WLSIC.WLS_Kov(obs_data, beta_ini, 0).do_wls_beta()
    assert resBeta[0][0] == pytest.approx(Beta, 0.01)
    
    
   
def test_Kov_do_wls_beta_gamma():
    liste_test_id = [ '01', '02', '03', '04']
    for test_id in liste_test_id:
        obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_obs.txt')
        evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_evt.txt')
        coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_coeff.txt')
        obs_data.loc[:, 'eqStd'] = obs_data.loc[:, 'StdI']
        
        liste_evt = evt_data.EVID.values
        Beta = coeff_data['Beta'].values[0]
        Gamma = coeff_data['Gamma'].values[0]
        beta_ini = -3
        gamma_ini = 0
        for evid in liste_evt:
            depth = evt_data[evt_data.EVID==evid].H.values[0]
            obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
                
        resBetaGamma = WLSIC.WLS_Kov(obs_data, beta_ini, gamma_ini).do_wls_betagamma()
        assert resBetaGamma[0][0] == pytest.approx(Beta, 0.01)
        assert np.round(resBetaGamma[0][1], 5) == np.round(Gamma, 5)
         

def test_do_wlsic_depth():
    liste_test_id = [ '01', '02', '03', '04']
    for test_id in liste_test_id:
        obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_obs.txt')
        evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_evt.txt')
        coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_coeff.txt')
            
        liste_evt = evt_data.EVID.values
        Beta = coeff_data['Beta'].values[0]
        Gamma = coeff_data['Gamma'].values[0]
        C1 = coeff_data['C1'].values[0]
        C2 = coeff_data['C2'].values[0]
        for evid in liste_evt:
            obsbin = obs_data[obs_data.EVID==evid]
            depth = evt_data[evt_data.EVID==evid].H.values[0]
            mag = evt_data[evt_data.EVID==evid].Mag.values[0]
            depth_ini = 10
            Hmin = 1
            Hmax = 20
            resH = WLSIC.WLSIC_oneEvt(obsbin, depth_ini, mag, Beta, Gamma, C1, C2).do_wlsic_depth(Hmin, Hmax)
            assert resH[0][0] == pytest.approx(depth, 0.01)
                          

   
def test_do_linregressC1regC2():
    test_id = '05'
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_coeff.txt')
    print(coeff_data.columns)
    
    liste_evt = evt_data.EVID.values
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        mag = evt_data[evt_data.EVID==evid].Mag.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
        obs_data.loc[obs_data.EVID==evid, 'Mag'] = mag
        temp = obs_data[obs_data.EVID==evid]
        obs_data.loc[obs_data.EVID==evid, 'eqStd'] = temp.StdI.values
        
    Beta = coeff_data['Beta'].values[0]
    C1a = coeff_data['C1a'].values[0]
    C1b = coeff_data[' C1b'].values[0]
    C2 = coeff_data[' C2'].values[0]
    obs_data.loc[:, 'beta'] = Beta
    obs_data.loc[:, 'gamma'] = 0
    
    resC1regC2 = WLSIC.WLS(obs_data, 1, 1, Beta, 0).do_linregressC1regC2(ftol=2e-3, max_nfev=1000, sigma=obs_data.StdI)
    print(resC1regC2)
    assert resC1regC2[0][0] == pytest.approx(C1a, abs=0.001)
    assert resC1regC2[0][1] == pytest.approx(C1b, abs=0.001)
    assert resC1regC2[0][2] == pytest.approx(C2, abs=0.001)
    
    test_id = '06'
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_coeff.txt')
    print(coeff_data.columns)
    
    liste_evt = evt_data.EVID.values
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        mag = evt_data[evt_data.EVID==evid].Mag.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
        obs_data.loc[obs_data.EVID==evid, 'Mag'] = mag
        temp = obs_data[obs_data.EVID==evid]
        obs_data.loc[obs_data.EVID==evid, 'eqStd'] = temp.StdI.values
        
    Beta = coeff_data['Beta'].values[0]
    C1a = coeff_data['C1a'].values[0]
    C1b = coeff_data[' C1b'].values[0]
    C1c = coeff_data[' C1c'].values[0]
    C1d = coeff_data[' C1d'].values[0]
    C2 = coeff_data[' C2'].values[0]
    obs_data.loc[:, 'beta'] = Beta
    obs_data.loc[:, 'gamma'] = 0
    
    resC1regC2 = WLSIC.WLS(obs_data, 1, 1, Beta, 0).do_linregressC1regC2(ftol=2e-3, max_nfev=1000, sigma=obs_data.StdI)

    assert resC1regC2[0][0] == pytest.approx(C1a, abs=0.001)
    assert resC1regC2[0][1] == pytest.approx(C1b, abs=0.001)
    assert resC1regC2[0][2] == pytest.approx(C1c, abs=0.001)
    assert resC1regC2[0][3] == pytest.approx(C1d, abs=0.001)
    assert resC1regC2[0][4] == pytest.approx(C2, abs=0.001)
    
    test_id = '07'
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_coeff.txt')

    liste_evt = evt_data.EVID.values
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        mag = evt_data[evt_data.EVID==evid].Mag.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
        obs_data.loc[obs_data.EVID==evid, 'Mag'] = mag
        temp = obs_data[obs_data.EVID==evid]
        obs_data.loc[obs_data.EVID==evid, 'eqStd'] = temp.StdI.values
        
    Beta = coeff_data['Beta'].values[0]
    C1a = coeff_data['C1a'].values[0]
    C2 = coeff_data[' C2'].values[0]
    obs_data.loc[:, 'beta'] = Beta
    obs_data.loc[:, 'gamma'] = 0
    
    
    resC1regC2 = WLSIC.WLS(obs_data, 1, 1, Beta, 0).do_linregressC1regC2(ftol=2e-3, max_nfev=1000, sigma=obs_data.StdI.values)

    assert resC1regC2[0][0] == pytest.approx(C1a, abs=0.001)
    assert resC1regC2[0][1] == pytest.approx(C2, abs=0.001)
    
    
def test_do_wls_C1C2BetaH():
    # Does not work with regional C1/Beta?
    liste_test_id = [ '03']
    for test_id in liste_test_id:
        obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_obs.txt')
        evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_evt.txt')
        coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_coeff.txt')
            
        liste_evt = evt_data.EVID.values
        Beta = coeff_data['Beta'].values[0]
        C1 = coeff_data['C1'].values[0]
        C2 = coeff_data['C2'].values[0]
        c1_ini = 1
        c2_ini = 1
        beta_ini = -3
        for evid in liste_evt:
            depth = evt_data[evt_data.EVID==evid].H.values[0]
            mag = evt_data[evt_data.EVID==evid].Mag.values[0]
            hinf = evt_data[evt_data.EVID==evid].Hinf.values[0]
            hsup = evt_data[evt_data.EVID==evid].Hsup.values[0]
            temp = obs_data[obs_data.EVID==evid]
            obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
            obs_data.loc[obs_data.EVID==evid, 'Hmin'] = hinf
            obs_data.loc[obs_data.EVID==evid, 'Hmax'] = hsup
            obs_data.loc[obs_data.EVID==evid, 'Mag'] = mag
            obs_data.loc[obs_data.EVID==evid, 'eqStd'] = temp.StdI.values
        
        #obs_data = obs_data[obs_data.EVID==evid]
        t0 = time.time()
        resC1C2Beta = WLSIC.WLS(obs_data, c1_ini, c2_ini, beta_ini, 0).do_wls_C1C2BetaH(ftol=2e-3, max_nfev=1000)
        t1 = time.time()
        print(t1-t0)
        Ipred_data = resC1C2Beta[0][0] + resC1C2Beta[0][1]*obs_data.Mag + resC1C2Beta[0][2]*np.log10(obs_data.Hypo)
        Ipred2_data = C1 + C2*obs_data.Mag + Beta*np.log10(obs_data.Hypo)
        print(resC1C2Beta)
#        print(Ipred_data-obs_data.I.values)
#        print(Ipred2_data-obs_data.I.values)
        print(np.mean(Ipred_data-obs_data.I.values))
        print(np.mean(Ipred2_data-obs_data.I.values))
        assert resC1C2Beta[0][0] == pytest.approx(C1, abs=0.1)
        assert resC1C2Beta[0][1] == pytest.approx(C2, abs=0.05)
        assert resC1C2Beta[0][2] == pytest.approx(Beta, abs=0.05)
        compt = 0
        for evid in liste_evt:
            depth = evt_data[evt_data.EVID==evid].H.values[0]
            assert resC1C2Beta[0][3+compt] == pytest.approx(depth, abs=0.7)
            compt += 1
            
        c1_ini = C1
        c2_ini = C2
        beta_ini = -3.5
        for evid in liste_evt:
            depth = evt_data[evt_data.EVID==evid].H.values[0]
            mag = evt_data[evt_data.EVID==evid].Mag.values[0]
            hinf = evt_data[evt_data.EVID==evid].Hinf.values[0]
            hsup = evt_data[evt_data.EVID==evid].Hsup.values[0]
            temp = obs_data[obs_data.EVID==evid]
            obs_data.loc[obs_data.EVID==evid, 'Depth'] = np.max([1, depth-random.choice([-2, 0, 2])*(depth-hinf)])
            obs_data.loc[obs_data.EVID==evid, 'Hmin'] = 1
            obs_data.loc[obs_data.EVID==evid, 'Hmax'] = 25
            obs_data.loc[obs_data.EVID==evid, 'Mag'] = mag
            obs_data.loc[obs_data.EVID==evid, 'eqStd'] = temp.StdI.values
        
        #obs_data = obs_data[obs_data.EVID==evid]
        t0 = time.time()
        resC1C2Beta2 = WLSIC.WLS(obs_data, c1_ini, c2_ini, beta_ini, 0).do_wls_C1C2BetaH(ftol=2e-3, max_nfev=1000)
        t1 = time.time()
        print(t1-t0)
        Ipred_data = resC1C2Beta2[0][0] + resC1C2Beta2[0][1]*obs_data.Mag + resC1C2Beta2[0][2]*np.log10(obs_data.Hypo)
        Ipred2_data = C1 + C2*obs_data.Mag + Beta*np.log10(obs_data.Hypo)
        print(resC1C2Beta)
#        print(Ipred_data-obs_data.I.values)
#        print(Ipred2_data-obs_data.I.values)
        print(np.mean(Ipred_data-obs_data.I.values))
        print(np.mean(Ipred2_data-obs_data.I.values))
        assert resC1C2Beta2[0][0] == pytest.approx(C1, abs=0.1)
        assert resC1C2Beta2[0][1] == pytest.approx(C2, abs=0.05)
        assert resC1C2Beta2[0][2] == pytest.approx(Beta, abs=0.05)
        compt = 0
        for evid in liste_evt:
            depth = evt_data[evt_data.EVID==evid].H.values[0]
            assert resC1C2Beta[0][3+compt] == pytest.approx(depth, abs=0.7)
            compt += 1
    
"""    
def test_do_linregressC1C2BetaHregC1Beta():
    test_id = '05'
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_coeff.txt')
    print(coeff_data.columns)
    
    liste_evt = evt_data.EVID.values
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        mag = evt_data[evt_data.EVID==evid].Mag.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
        obs_data.loc[obs_data.EVID==evid, 'Hmin'] = 1
        obs_data.loc[obs_data.EVID==evid, 'Hmax'] = 25
        obs_data.loc[obs_data.EVID==evid, 'Mag'] = mag
        temp = obs_data[obs_data.EVID==evid]
        obs_data.loc[obs_data.EVID==evid, 'eqStd'] = temp.StdI.values
        
    Beta = coeff_data['Beta'].values[0]
    C1a = coeff_data['C1a'].values[0]
    C1b = coeff_data[' C1b'].values[0]
    C2 = coeff_data[' C2'].values[0]
    
    result = WLSIC.WLS(obs_data, 1, 1, Beta, 0).do_wls_C1C2BetaH_2regC1beta(ftol=2e-3, max_nfev=1000)
    print(result)
    assert result[0][0] == pytest.approx(C1a, abs=0.001)
    assert result[0][1] == pytest.approx(C1b, abs=0.001)
    assert result[0][2] == pytest.approx(C2, abs=0.001)
    assert result[0][3] == pytest.approx(Beta, abs=0.001)
    assert result[0][4] == pytest.approx(Beta, abs=0.001)
    
    test_id = '08'
    obs_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_obs.txt')
    evt_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_evt.txt')
    coeff_data = pd.read_csv('../Testpy_dataset/pytest_dataset' + test_id + '_coeff.txt')
    print(coeff_data.columns)
    
    liste_evt = evt_data.EVID.values
    for evid in liste_evt:
        depth = evt_data[evt_data.EVID==evid].H.values[0]
        mag = evt_data[evt_data.EVID==evid].Mag.values[0]
        obs_data.loc[obs_data.EVID==evid, 'Depth'] = depth
        obs_data.loc[obs_data.EVID==evid, 'Hmin'] = 1
        obs_data.loc[obs_data.EVID==evid, 'Hmax'] = 25
        obs_data.loc[obs_data.EVID==evid, 'Mag'] = mag
        temp = obs_data[obs_data.EVID==evid]
        obs_data.loc[obs_data.EVID==evid, 'eqStd'] = temp.StdI.values

        
    C1a = coeff_data['C1a'].values[0]
    C1b = coeff_data[' C1b'].values[0]
    C2 = coeff_data[' C2'].values[0]
    betaa = coeff_data[' Betaa'].values[0]
    betab = coeff_data[' Betab'].values[0]
    
    result = WLSIC.WLS(obs_data, 1, 1.5, Beta, 0).do_wls_C1C2BetaH_2regC1beta(ftol=5e-5, max_nfev=5000,
                                                                              betaa=-4.15, C1a=2, C1b=4)

    assert result[0][0] == pytest.approx(C1a, abs=0.001)
    assert result[0][1] == pytest.approx(C1b, abs=0.001)
    assert result[0][2] == pytest.approx(C2, abs=0.001)
    assert result[0][3] == pytest.approx(betaa, abs=0.001)
    assert result[0][4] == pytest.approx(betab, abs=0.001)
    
"""