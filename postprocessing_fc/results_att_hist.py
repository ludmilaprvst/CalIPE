# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:41:44 2021

@author: PROVOST-LUD
"""

import numpy as np
import pandas as pd
import os
from postprocessing_Kovbeta import readBetaFile, readBetaGammaFile

def get_list_beta_folder(output_folder):
    liste_fichiers = os.listdir(output_folder)
    liste_beta = []
    fichiers_beta = []
    for fichier in liste_fichiers:
        if 'betaFinal' in fichier:
            beta = readBetaFile(output_folder+'/'+fichier)[0]
            liste_beta.append(beta)
            fichiers_beta.append(fichier)
    return liste_beta, fichiers_beta

def get_list_betagamma_folder(output_folder):
    liste_fichiers = os.listdir(output_folder)
    liste_beta = []
    liste_gamma = []
    fichiers_betagamma = []
    for fichier in liste_fichiers:
        if 'betagammaFinal' in fichier:
            beta = readBetaGammaFile(output_folder+'/'+fichier)[0]
            liste_beta.append(beta)
            gamma = readBetaGammaFile(output_folder+'/'+fichier)[3]
            liste_gamma.append(gamma)
            fichiers_betagamma.append(fichier)
    return liste_beta, liste_gamma, fichiers_betagamma

def get_beta_hist(output_folder, pas=0.05):
    liste_beta = get_list_beta_folder(output_folder)[0]
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

def get_betagamma_hist(output_folder, xedges = np.arange(-5, -1.9, 0.1), yedges = np.arange(-0.01, 0.00002, 0.00001)):
    liste_beta, liste_gamma, fichiers_betagamma = get_list_betagamma_folder(output_folder)
    hist, xedges, yedges = np.histogram2d(liste_beta, liste_gamma, bins=(xedges, yedges))
    hist = hist/sum(hist)
    return hist, xedges, yedges