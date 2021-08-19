# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:55:30 2021

@author: PROVOST-LUD
"""

import os
from postprocessing_Kovbeta import readBetaFile
import numpy as np
import matplotlib.pyplot as plt


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
"""    
output_folder = '../Outputs/FR_instru_01/Boostrap_FRinstru'
output_folder_PYREST = '../Outputs/FR_instru_01_PYREST/Bootstrap'
output_folder_ALPS = '../Outputs/FR_instru_01_ALPS/Bootstrap'
output_folder_ARMOR = '../Outputs/FR_instru_01_ARMOR/Bootstrap'

liste_beta, beta_baseDB = get_hist_bootstrap(output_folder)
liste_beta_PYREST, beta_baseDB_PYREST = get_hist_bootstrap(output_folder_PYREST)
liste_beta_ALPS, beta_baseDB_ALPS = get_hist_bootstrap(output_folder_ALPS)
liste_beta_ARMOR, beta_baseDB_ARMOR = get_hist_bootstrap(output_folder_ARMOR)

liste_beta = np.array(liste_beta)
liste_beta_PYREST = np.array(liste_beta_PYREST)
liste_beta_ALPS = np.array(liste_beta_ALPS)
liste_beta_ARMOR = np.array(liste_beta_ARMOR)

fig = plt.figure(figsize=(8, 8))
ax_FR = fig.add_subplot(221)
ax_PYREST = fig.add_subplot(222)
ax_ALPS = fig.add_subplot(224)
ax_ARMOR = fig.add_subplot(223)

plot_hist_bootstrap(ax_FR, liste_beta, beta_baseDB, xlim=[-4, -2])
ax_FR.set_title("FR")
plot_hist_bootstrap(ax_PYREST, liste_beta_PYREST, beta_baseDB_PYREST, xlim=[-4, -2])
ax_PYREST.set_title("PYREST")
plot_hist_bootstrap(ax_ALPS, liste_beta_ALPS, beta_baseDB_ALPS, xlim=[-4, -2])
ax_ALPS.set_title("ALPS")
plot_hist_bootstrap(ax_ARMOR, liste_beta_ARMOR, beta_baseDB_ARMOR, xlim=[-4, -2])
ax_ARMOR.set_title("ARMOR")
plt.tight_layout()
fig.savefig('../bootstrap_result.png', dpi=150)
"""

"""
output_folder = '../../Outputs/FR_extended_01/Subsets_01'
liste_beta, beta_baseDB = get_hist_bootstrap(output_folder)
output_folder2 = '../../Outputs/FR_instru_01/Subsets_01'
liste_beta2, beta_baseDB2 = get_hist_bootstrap(output_folder2)
liste_beta = np.array(liste_beta)
fig = plt.figure(figsize=(6, 6))
ax_FR = fig.add_subplot(111)
plot_hist_bootstrap(ax_FR, liste_beta, beta_baseDB, xlim=[-4, -2],
                    ylim=[0, 1200], bbins=np.arange(-3.7, -2.5, 0.05))

plot_hist_bootstrap(ax_FR, liste_beta2, beta_baseDB2, xlim=[-4, -2],
                    ylim=[0, 1200], bbins=np.arange(-3.7, -2.5, 0.05))
ax_FR.set_title("FR extended")
ax_FR.grid(which='both')
fig.savefig('../../FR_extendedetinstru_01_beta_distribution.png', dpi=150)
"""
"""
output_folder = '../../Outputs/FR_extended_01/Subsets_01'
liste_beta, beta_baseDB = get_hist_bootstrap(output_folder)
liste_beta = np.array(liste_beta)
fig = plt.figure(figsize=(6, 6))
ax_FR = fig.add_subplot(111)
plot_hist_bootstrap(ax_FR, liste_beta, beta_baseDB, xlim=[-4, -2],
                    ylim=[0, 1200], bbins=np.arange(-3.7, -2.5, 0.05))
ax_FR.set_title("FR extended")
fig.savefig('../../FR_extended_01_beta_distribution.png', dpi=150)
"""

"""
output_folder = '../../Outputs/FR_instru_01/bootstrap_1evt_FRinstru/Beta'
liste_beta, beta_baseDB = get_hist_bootstrap(output_folder)
liste_beta = np.array(liste_beta)
fig = plt.figure(figsize=(6, 6))
ax_FR = fig.add_subplot(111)
plot_hist_bootstrap(ax_FR, liste_beta, beta_baseDB, xlim=[-4, -2],
                    ylim=[0, 20], bbins=np.arange(-3.7, -2.5, 0.02))
ax_FR.set_title("FR instru")
ax_FR.legend()
fig.savefig('../../FR_instru_01_boostrap_beta_distribution.png', dpi=150)
"""