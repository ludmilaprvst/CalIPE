# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 17:47:23 2021

@author: PROVOST-LUD
"""


import pytest
import pandas as pd
import numpy as np

from library_bin_new import RAVG

def test_RAVG():
    obsdata = pd.read_csv('../Testpy_dataset/obs_test.txt', sep='\t')
    expected_output = pd.read_csv('../Testpy_dataset/result_RAVG_H=0.txt', sep='\t')
    I0 = 8
    QI0 = 0.5
    Ic = 6
    depth = 0
    obsbin = RAVG(obsdata, depth, Ic, I0, QI0)
    
    for ind, row in obsbin.iterrows():
        I_eval = row['I']
        expected_row = expected_output[expected_output.I==I_eval]
        assert expected_row['Depi'].values[0] == pytest.approx(row['Depi'], 0.01)
        assert expected_row['StdLogR'].values[0] == pytest.approx(row['StdLogR'], 0.01)
        assert expected_row['StdI'].values[0] == pytest.approx(row['StdI'], 0.01)
        assert expected_row['Ndata'].values[0] == pytest.approx(row['Ndata'], 0.1)