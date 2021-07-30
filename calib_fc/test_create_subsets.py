# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 17:36:13 2021

@author: PROVOST-LUD
"""

import pytest
from create_subsets import create_liste_subset, filter_by_nevt, check_duplicate
from create_subsets import same_values_2array, create_bootstrap_notrandom
import pandas as pd
import numpy as np


def test_create_liste_subset():
    criteria = pd.read_excel("../Testpy_dataset/subset_criteria_test.xlsx")
    
    year_inf = []
    year_sup = [2020, 2006]
    QH = [['A'], ['A', 'B']]
    NClass = [2, 3, 4]
    Nobs = [10, 25, 50, 100, 150]
    Dc = [10, 25, 50]
    global_liste, criteres = create_liste_subset(criteria, year_inf, year_sup,
                        QH, NClass, Nobs, Dc)
    assert len(global_liste) == 180
    
def test_filter_by_nevt():
    
    global_liste = [[101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                    [101, 102, 103, 104],
                    [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]]
    d = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c'], 'col3': ['A,B', 'A', 'B']}
    criteres = pd.DataFrame(data=d)
    new_liste, new_critere = filter_by_nevt(global_liste, criteres, nmin=8)
    assert len(new_liste) == 2
    assert len(new_critere) == 2
    assert new_critere.loc[0, 'col1'] == 1
    assert new_critere.loc[1, 'col1'] == 3
    
def test_check_duplicate():
    
    global_liste = [[101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                    [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                    [101, 102, 103, 104],
                    [101, 102, 103, 104],
                    [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]]
    d = {'col1': [1, 2, 3, 4, 5],
         'col2': ['a', 'b', 'c', 'd', 'e'],
         'col3': ['A,B', 'A', 'B', 'A,B,C', 'A,B']}
    criteres = pd.DataFrame(data=d)
    new_liste, new_critere = check_duplicate(global_liste, criteres)
    assert len(new_liste) == 3
    assert len(new_critere) == 3
    assert new_critere.loc[0, 'col1'] == 2
    assert new_critere.loc[1, 'col1'] == 4
    assert new_critere.loc[2, 'col1'] == 5
    
def test_same_values_2array():
    array1 = [1, 2, 3, 4]
    array2 = [4, 2, 1, 3]
    array3 = [1, 2, 3]
    array4 = [5, 6, 7, 8]
    assert same_values_2array(array1, array2) == True
    assert same_values_2array(array1, array3) == False
    assert same_values_2array(array1, array4) == False
    
def test_create_bootstrap_notrandom():
    liste01 = np.array([1, 2, 3, 4, 5])
    bootstrap_list, bootstrap_criteres = create_bootstrap_notrandom(liste01)
    assert len(bootstrap_criteres.columns) == 1
    assert bootstrap_criteres.columns[0] == 'Deleted event id'
    for ind in range(5):
        assert bootstrap_criteres.loc[ind, 'Deleted event id'] == ind+1
    assert np.array_equal(bootstrap_list[0], np.array([2, 3, 4, 5]))
    assert np.array_equal(bootstrap_list[1], np.array([1, 3, 4, 5]))
    assert np.array_equal(bootstrap_list[2], np.array([1, 2, 4, 5]))
    