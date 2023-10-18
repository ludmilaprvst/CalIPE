# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:14:05 2021

@author: PROVOST-LUD
"""

import pytest
import pandas as pd
import numpy as np
from ponderations import weight_IStdI, normaliser_poids_par_evt
from ponderations import weight_evtUni, weight_evtStdM, weight_evtStdH
from ponderations import weight_IStdI_evtStdM
from ponderations import normaliser_par_MClass
from ponderations import weight_IStdI_evtStdM_gMclass, weight_IStdI_evtUni
from ponderations import weight_IStdI_evtStdM, weight_IStdI_evtStdH
from ponderations import weight_IStdI_evtStdM_gRegion
from ponderations import attribute_region, normaliser_par_region

def test_weight_IStdI():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    obsdata = weight_IStdI(obsdata)
    assert obsdata.loc[:, 'StdI'].equals(obsdata.loc[:, 'eqStd'])


def test_normaliser_poids_par_evt():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    obsdata = weight_IStdI(obsdata)
    liste_evt = np.unique(obsdata.EVID.values)
    for evid in liste_evt:
        obsdata = normaliser_poids_par_evt(evid, obsdata)
        somme_poids_evid = np.sum(obsdata[obsdata.EVID==evid]['Poids_int_norm'].values)
        assert somme_poids_evid == pytest.approx(1, 0.001)
 

def test_weight_evtUni():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    liste_evt = np.unique(obsdata.EVID.values)
    obsdata.loc[:, 'poids_test'] = 1/len(liste_evt)
    obsdata = weight_evtUni(obsdata)
    assert obsdata.loc[:, 'Poids_evt'].equals(obsdata.loc[:, 'poids_test'])


def test_weight_IStdI_evtUni():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    obsdata = weight_IStdI_evtUni(obsdata)
    assert 'eqStd' in obsdata.columns

        

def test_weight_evtStdM():
    evtdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    # Attribuer les StdM par EVID dans obsdata
    for evid in evtdata.EVID.values:
        StdM = evtdata[evtdata.EVID==evid].StdM.values[0]
        obsdata.loc[obsdata.EVID==evid, 'StdM'] = StdM
    # Appliquer weight_evtStdM
    obsdata = weight_evtStdM(obsdata)
    for evid in evtdata.EVID.values:
        poids_evt = obsdata.loc[obsdata.EVID==evid, 'Poids_evt'].mean()
        StdM = evtdata[evtdata.EVID==evid].StdM.values[0]
        StdM = np.max([StdM, 0.1])
        assert poids_evt == pytest.approx(1/StdM**2)


def test_weight_IStdI_evtStdM():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    evtdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    for evid in evtdata.EVID.values:
        Mag = evtdata[evtdata.EVID==evid].Mag.values[0]
        obsdata.loc[obsdata.EVID==evid, 'Mag'] = Mag
        StdM = evtdata[evtdata.EVID==evid].StdM.values[0]
        obsdata.loc[obsdata.EVID==evid, 'StdM'] = np.max([StdM, 0.1])
    obsdata = weight_IStdI_evtStdM(obsdata)
    assert 'eqStd' in obsdata.columns


def test_weight_evtStdH():
    evtdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    # Attribuer les StdH par EVID dans obsdata
    for evid in evtdata.EVID.values:
        Hmin = evtdata[evtdata.EVID==evid].Hinf.values[0]
        Hmax = evtdata[evtdata.EVID==evid].Hsup.values[0]
        obsdata.loc[obsdata.EVID==evid, 'Hmin'] = Hmin
        obsdata.loc[obsdata.EVID==evid, 'Hmax'] = Hmax
    # Appliquer weight_evtStdH
    obsdata = weight_evtStdH(obsdata)
    for evid in evtdata.EVID.values:
        poids_evt = obsdata.loc[obsdata.EVID==evid, 'Poids_evt'].mean()
        StdH = (evtdata[evtdata.EVID==evid].Hsup.values[0]-evtdata[evtdata.EVID==evid].Hinf.values[0])/2
        assert poids_evt == pytest.approx(1/StdH**2)

def test_weight_IStdI_evtStdH():
    evtdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    # Attribuer les StdH par EVID dans obsdata
    for evid in evtdata.EVID.values:
        Hmin = evtdata[evtdata.EVID==evid].Hinf.values[0]
        Hmax = evtdata[evtdata.EVID==evid].Hsup.values[0]
        obsdata.loc[obsdata.EVID==evid, 'Hmin'] = Hmin
        obsdata.loc[obsdata.EVID==evid, 'Hmax'] = Hmax
    obsdata = weight_IStdI_evtStdH(obsdata)
    assert 'eqStd' in obsdata.columns


def test_normaliser_par_MClass():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    evtdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    minMag = evtdata.Mag.min()
    maxMag = evtdata.Mag.max()
    mag_bins = np.arange(minMag, maxMag+0.5, 0.5)
    for evid in evtdata.EVID.values:
        Mag = evtdata[evtdata.EVID==evid].Mag.values[0]
        obsdata.loc[obsdata.EVID==evid, 'Mag'] = Mag
        StdM = evtdata[evtdata.EVID==evid].StdM.values[0]
        obsdata.loc[obsdata.EVID==evid, 'StdM'] = np.max([StdM, 0.1])
    obsdata = weight_IStdI_evtStdM(obsdata)
    obsdata, nbre_bin = normaliser_par_MClass(obsdata)
    obsdata.loc[:, 'range1'] = pd.cut(obsdata.Mag, mag_bins, include_lowest=True)
    mag_bins_df = np.unique(obsdata.range1.values)
    for bins in mag_bins_df:
        ind = obsdata.range1==bins
        assert obsdata.loc[ind, 'Poids'].sum() == pytest.approx(1, 0.001)

def test_weight_IStdI_evtStdM_gMclass():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    evtdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    for evid in evtdata.EVID.values:
        Mag = evtdata[evtdata.EVID==evid].Mag.values[0]
        obsdata.loc[obsdata.EVID==evid, 'Mag'] = Mag
        StdM = evtdata[evtdata.EVID==evid].StdM.values[0]
        obsdata.loc[obsdata.EVID==evid, 'StdM'] = np.max([StdM, 0.1])
    obsdata = weight_IStdI_evtStdM_gMclass(obsdata)
    assert 'eqStd' in obsdata.columns


def test_attribute_region():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    fichier_regions = '../Testpy_dataset/regions_test.txt'
    data_evt = pd.read_csv('../Testpy_dataset/loc_evt_pond_reg.txt', sep='\t')
    liste_evt = data_evt.EVID.values
    obsdata = obsdata[obsdata.EVID.isin(liste_evt)]
    obsdata = attribute_region(data_evt, obsdata, fichier_regions)
    for evid in liste_evt:
        tmpobs = obsdata[obsdata.EVID==evid]
        regid = tmpobs['RegID'].values[0]
        tmpevt = data_evt[data_evt.EVID==evid]
        id_loc = tmpevt['id_loc'].values[0]
        
        assert regid == pytest.approx(id_loc, 10)


def test_normaliser_par_region():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    obsdata = weight_IStdI(obsdata)
    fichier_regions = '../Testpy_dataset/regions_test.txt'
    data_evt = pd.read_csv('../Testpy_dataset/loc_evt_pond_reg.txt', sep='\t')
    liste_evt = data_evt.EVID.values
    obsdata = obsdata[obsdata.EVID.isin(liste_evt)]
    obsdata = attribute_region(data_evt, obsdata, fichier_regions)
    regions = np.unique(obsdata.RegID.values)
    for evid in liste_evt:
        obsdata = normaliser_poids_par_evt(evid, obsdata)
    for regid in regions:
        obsdata = normaliser_par_region(regid, obsdata)
        somme_poids_reg = np.sum(obsdata[obsdata.RegID==regid]['Poids_inreg_norm'].values)
        assert somme_poids_reg == pytest.approx(1, 0.001)
       
def test_weight_IStdI_evtStdM_gRegion():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    evtdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_evt.txt')
    fichier_regions = '../Testpy_dataset/regions_test.txt'
    data_evt = pd.read_csv('../Testpy_dataset/loc_evt_pond_reg.txt', sep='\t')
    liste_evt = data_evt.EVID.values
    obsdata = obsdata[obsdata.EVID.isin(liste_evt)]
    obsdata = attribute_region(data_evt, obsdata, fichier_regions)
    for evid in evtdata.EVID.values:
        Mag = evtdata[evtdata.EVID==evid].Mag.values[0]
        obsdata.loc[obsdata.EVID==evid, 'Mag'] = Mag
        StdM = evtdata[evtdata.EVID==evid].StdM.values[0]
        obsdata.loc[obsdata.EVID==evid, 'StdM'] = np.max([StdM, 0.1])
    obsdata = weight_IStdI_evtStdM_gRegion(obsdata)
    assert 'eqStd' in obsdata.columns


# # def test_Kovatt_ponderation_evt_reg():
# #     obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
# #     fichier_regions = '../Testpy_dataset/regions_test.txt'
# #     data_evt = pd.read_csv('../Testpy_dataset/loc_evt_pond_reg.txt', sep='\t')
# #     liste_evt = data_evt.EVID.values
# #     obsdata = obsdata[obsdata.EVID.isin(liste_evt)]
# #     obsdata = attribute_region(data_evt, obsdata, fichier_regions)
# #     obsdata = Kovatt_ponderation_evt_reg(obsdata)
# #     regions = np.unique(obsdata.RegID.values)

# #     poids = 1/obsdata.loc[:, 'eqStd']**2
# #     assert np.sum(poids) == pytest.approx(1, 0.001)
    
# #     regions = np.unique(obsdata.RegID.values)
# #     poids_reg = 1/len(regions)
# #     for regid in regions:
# #         sum_regid = np.sum(1/(obsdata[obsdata.RegID==regid]['eqStd']**2))
# #         assert sum_regid == pytest.approx(poids_reg, 0.001)


        
# # def test_Kovatt_ponderation_dI():
# #     obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
# #     obsdata = Kovatt_ponderation_dI(obsdata)
# #     assert obsdata.loc[:, 'StdI'].equals(obsdata.loc[:, 'eqStd'])