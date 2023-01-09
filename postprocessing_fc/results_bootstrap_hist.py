# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:55:30 2021

@author: PROVOST-LUD
"""

import os
from postprocessing_Kovbeta import readBetaFile
import numpy as np
import pandas as pd
import ntpath


def get_hist_bootstrap(output_folder):
    liste_fichiers = os.listdir(output_folder)
    liste_beta = []
    for fichier in liste_fichiers:
        if 'betaFinal' in fichier:
            if 'FR' in fichier:
                beta_baseDB = readBetaFile(output_folder+'/'+fichier)[0]
            else:
                
                beta = readBetaFile(output_folder+'/'+fichier)[0]
                liste_beta.append(beta)
                #print(fichier, beta)
    return liste_beta, beta_baseDB

def plot_hist_bootstrap(ax, liste, beta_base,
                        xlim=[-4, -2], ylim=[0, 15], bbins=np.arange(-4, -2, 0.1)):
    ax.hist(liste, bins=bbins, density=False, alpha=1, label='subsets beta results')
    ax.axvline(x=beta_base, color='k', lw=1,ls='--', label='beta value from the base database')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Beta values')
    #ax.legend(loc=2)
    ax.grid(which='both')
    
def get_table_inoutputbeta(output_folder):
    liste_fichiers = os.listdir(output_folder)
    liste_beta = []
    liste_evtfile = []
    liste_betaini = []
    for fichier in liste_fichiers:
        if 'betaFinal' in fichier:
            if 'FR' in fichier:
                beta_baseDB = readBetaFile(output_folder+'/'+fichier)[0]
                liste_beta.append(beta_baseDB)
                evtfile = ntpath.basename(readBetaFile(output_folder+'/'+fichier)[8])
                basename = evtfile.split('.')[0]
                liste_evtfile.append(basename)
                liste_betaini.append(readBetaFile(output_folder+'/'+fichier)[2])
            else:
                
                beta = readBetaFile(output_folder+'/'+fichier)[0]
                liste_beta.append(beta)
                evtfile = ntpath.basename(readBetaFile(output_folder+'/'+fichier)[8])
                basename = evtfile.split('.')[0]
                liste_evtfile.append(basename)
                liste_betaini.append(readBetaFile(output_folder+'/'+fichier)[2])
                #print(fichier, beta)
                
    return pd.DataFrame.from_dict({'beta' : liste_beta,
                                   'beta_ini' : liste_betaini,
                                   'database' : liste_evtfile})
                
