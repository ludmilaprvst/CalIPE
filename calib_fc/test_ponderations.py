# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:14:05 2021

@author: PROVOST-LUD
"""

import pytest
import pandas as pd
import numpy as np
from ponderations import normaliser_poids_par_evt, Kovatt_ponderation_evt_uniforme
from ponderations import Kovatt_ponderation_dI, Kovatt_ponderation_evt_reg
from ponderations import attribute_region, normaliser_par_region

def test_normaliser_poids_par_evt():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    liste_evt = np.unique(obsdata.EVID.values)
    for evid in liste_evt:
        obsdata = normaliser_poids_par_evt(evid, obsdata)
        somme_poids_evid = np.sum(obsdata[obsdata.EVID==evid]['Poids_inevt_norm'].values)
        assert somme_poids_evid == pytest.approx(1, 0.001)

def test_normaliser_par_region():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
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
       
def test_Kovatt_ponderation_evt_uniforme():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    obsdata = Kovatt_ponderation_evt_uniforme(obsdata)
    poids = 1/obsdata.loc[:, 'eqStd']**2
    assert np.sum(poids) == pytest.approx(1, 0.001)
    
    liste_evt = np.unique(obsdata.EVID.values)
    poids_evt = 1/len(liste_evt)
    for evid in liste_evt:
        sum_evid = np.sum(1/(obsdata[obsdata.EVID==evid]['eqStd']**2))
        assert sum_evid == pytest.approx(poids_evt, 0.001)


def test_Kovatt_ponderation_evt_reg():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    fichier_regions = '../Testpy_dataset/regions_test.txt'
    data_evt = pd.read_csv('../Testpy_dataset/loc_evt_pond_reg.txt', sep='\t')
    liste_evt = data_evt.EVID.values
    obsdata = obsdata[obsdata.EVID.isin(liste_evt)]
    obsdata = attribute_region(data_evt, obsdata, fichier_regions)
    obsdata = Kovatt_ponderation_evt_reg(obsdata)
    regions = np.unique(obsdata.RegID.values)

    poids = 1/obsdata.loc[:, 'eqStd']**2
    assert np.sum(poids) == pytest.approx(1, 0.001)
    
    regions = np.unique(obsdata.RegID.values)
    poids_reg = 1/len(regions)
    for regid in regions:
        sum_regid = np.sum(1/(obsdata[obsdata.RegID==regid]['eqStd']**2))
        assert sum_regid == pytest.approx(poids_reg, 0.001)

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
        
def test_Kovatt_ponderation_dI():
    obsdata = pd.read_csv('../Testpy_dataset/pytest_dataset03_obs.txt')
    obsdata = Kovatt_ponderation_dI(obsdata)
    assert obsdata.loc[:, 'StdI'].equals(obsdata.loc[:, 'eqStd'])