# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 16:10:57 2021

@author: PROVOST-LUD
"""

import sys
sys.path.append('../postprocessing_fc')
from results_att_hist import get_beta_hist
from results_bootstrap_hist import get_hist_bootstrap, plot_hist_bootstrap
import matplotlib.pyplot as plt
import numpy as np
outputfolder = '../../Outputs/FR_extended_01/Beta/Subsets_01'

histo = get_beta_hist(outputfolder)
liste_beta, beta_baseDB = get_hist_bootstrap(outputfolder)
#histo.to_csv('../../Data/Att_distributions/FR_extended_01_beta.csv', sep=';', index=False)
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

output_folder = '../../Outputs/FR_extended_01/Beta/Subsets_01'
liste_beta, beta_baseDB = get_hist_bootstrap(output_folder)
output_folder2 = '../../Outputs/FR_instru_01/Subsets_01/Beta'
liste_beta2, beta_baseDB2 = get_hist_bootstrap(output_folder2)
liste_beta = np.array(liste_beta)
fig = plt.figure(figsize=(6, 6))
ax_FR = fig.add_subplot(111)
#plot_hist_bootstrap(ax_FR, liste_beta, beta_baseDB, xlim=[-4, -2],
#                    ylim=[0, 1200], bbins=np.arange(-3.7, -2.5, 0.05))

plot_hist_bootstrap(ax_FR, liste_beta2, beta_baseDB2, xlim=[-4, -2],
                    ylim=[0, 1200], bbins=np.arange(-3.7, -2.5, 0.05))
ax_FR.set_title("FR instru")
#ax_FR.grid(which='both')
fig.savefig('../../FR_instru_01_beta_distribution.png', dpi=150)