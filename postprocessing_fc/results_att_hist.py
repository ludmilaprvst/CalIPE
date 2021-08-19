# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:41:44 2021

@author: PROVOST-LUD
"""

import numpy as np
import pandas as pd
from results_bootstrap_hist import get_hist_bootstrap

def get_beta_hist(output_folder, pas=0.05):
    liste_beta = get_hist_bootstrap(output_folder)[0]
    mean = np.mean(liste_beta)
    min_beta = np.min(liste_beta)
    max_beta = np.max(liste_beta)
    ecart_max = np.max([np.abs(mean-pas/2-min_beta), np.abs(mean+pas/2-max_beta)])
    nbre_pas = np.round(ecart_max/pas+1, 0)
    min_bin = (mean - pas/2) - pas*nbre_pas 
    max_bin = (mean + pas/2) + pas*nbre_pas + pas 
    beta_bins = np.arange(min_bin, max_bin, pas)
    hist, bin_edges = np.histogram(liste_beta, bins=beta_bins)
    hist = hist/sum(hist)
    milieu =  bin_edges[:-1]+pas/2
    milieu = np.around(milieu, decimals=2)
    output = pd.DataFrame({'beta':milieu, 'proba':hist})
    output = output[output['proba']!=0]
    return output