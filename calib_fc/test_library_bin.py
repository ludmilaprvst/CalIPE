# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 17:47:23 2021

@author: PROVOST-LUD
"""


import pytest
import pandas as pd
import numpy as np

from library_bin import ROBS, RAVG

def test_ROBS():
    obsdata = pd.read_csv('../Testpy_dataset/obs_test.txt', sep='\t')
    expected_output = pd.read_csv('../Testpy_dataset/result_ROBS_H=0.txt', sep='\t')
    
    I0 = 8
    QI0 = 0.5
    Ic = 6
    depth = 0
    obsbin = ROBS(obsdata, depth, Ic, I0, QI0)
    
    for ind, row in obsbin.iterrows():
        I_eval = row['I']
        expected_row = expected_output[expected_output.I==I_eval]
        assert expected_row['Depi'].values[0] == pytest.approx(row['Depi'], 0.01)
        assert expected_row['StdLogR'].values[0] == pytest.approx(row['StdLogR'], 0.01)
        assert expected_row['StdI'].values[0] == pytest.approx(row['StdI'], 0.01)
        assert expected_row['Ndata'].values[0] == pytest.approx(row['Ndata'], 0.1)
    
    obsdata = pd.read_csv('../Testpy_dataset/obs_test2.txt', sep='\t')
    expected_output = pd.read_csv('../Testpy_dataset/result2_ROBS_H=0.txt', sep='\t')
    obsbin = ROBS(obsdata, depth, Ic, I0, QI0)
    for ind, row in obsbin.iterrows():
        I_eval = row['I']
        expected_row = expected_output[expected_output.I==I_eval]
        assert expected_row['Depi'].values[0] == pytest.approx(row['Depi'], 0.01)
        assert expected_row['StdLogR'].values[0] == pytest.approx(row['StdLogR'], 0.01)
        assert expected_row['StdI'].values[0] == pytest.approx(row['StdI'], 0.01)
        assert expected_row['Ndata'].values[0] == pytest.approx(row['Ndata'], 0.1)
        
        
def test_RAVG():
    obsdata = pd.read_csv('../Testpy_dataset/obs_test.txt', sep='\t')
    expected_output = pd.read_csv('../Testpy_dataset/result_RAVG_H=0.txt', sep='\t')
    
    I0 = 8
    QI0 = 0.5
    Ic = 6
    depth = 0
    obsbin = RAVG(obsdata, depth, Ic, I0, QI0)
    obsbin.sort_values(by='I', ascending=False, inplace=True)
    obsbin.reset_index(inplace=True)
    print(expected_output)
    for ind, row in obsbin.iterrows():
        #I_eval = row['I']
        print(ind)
        expected_row = expected_output.loc[ind, :]
        #print(expected_row['I'].values[0], I_eval)
        assert expected_row['I'] == pytest.approx(row['I'], 0.01)
        assert expected_row['Depi'] == pytest.approx(row['Depi'], 0.01)
        assert expected_row['StdLogR'] == pytest.approx(row['StdLogR'], 0.01)
        assert expected_row['StdI'] == pytest.approx(row['StdI'], 0.01)
        assert expected_row['Ndata'] == pytest.approx(row['Ndata'], 0.1)        
        