# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:40:28 2021

@author: PROVOST-LUD
"""

import numpy as np
import pandas as pd

C1a = 2.2
C1b = 4.1
C2 = 1.6
Betaa = -4.2
Betab = -3.0
Gamma = 0

depi_range = np.logspace(0, 2.7, 100)

depth_list = [2, 5, 10, 15, 20]
mag_list = [3, 3.5, 4, 4.5, 5, 5.5, 6]

dataset_obs = pd.DataFrame(columns=['EVID', 'Depi', 'Hypo', 'I', 'Io', 'QIo',
                                    'StdI', 'StdlogR', 'Ndata', 'RegID'])
dataset_evt = pd.DataFrame(columns=['EVID', 'Mag', 'StdM', 'H', 'Hinf', 'Hsup'])


count_evt = 1
for mag in mag_list:
    for depth in depth_list:
        depi_range = np.logspace(0, 2.7, 100)
        hypo = np.sqrt(depi_range**2 + depth**2)
        Ipred = C1a + C2*mag + Betaa*np.log10(hypo) + Gamma*hypo
#        if count_evt == 27:
#            print(Ipred)
#            print(Ipred[Ipred>=1])
        if np.any(Ipred>4):
            depi_range = depi_range[Ipred>=1]
            Hypo  = hypo[Ipred>=1]
            Ipred = Ipred[Ipred>=1]
            Io = C1a + C2*mag + Betaa*np.log10(depth) + Gamma*depth
            Io = Io * np.ones(len(Ipred))
            QIo = 0.5 * np.ones(len(Ipred))
            StdI = 0.25 * np.ones(len(Ipred))
            StdlogR = 1.2 * np.ones(len(Ipred))
            Ndata = 10 * np.ones(len(Ipred))
            RegID = 101 * np.ones(len(Ipred))
            EVID = count_evt * np.ones(len(Ipred))
            temp = pd.DataFrame({'EVID' : EVID, 'Depi': depi_range, 'Hypo': Hypo,
                                 'I': Ipred, 'Io': Io, 'QIo': QIo, 'StdI' : StdI,
                                 'StdlogR': StdlogR, 'Ndata': Ndata, 'RegID': RegID})
            dataset_obs = pd.concat([dataset_obs, temp])
            dataset_evt.loc[count_evt, :] = [count_evt, mag, 0.1, depth, depth-0.5, depth+0.5]
            
            count_evt += 1
#            if count_evt == 28:
#                print(depi_range)
#                print(Ipred)

for mag in mag_list:
    for depth in depth_list:
        depi_range = np.logspace(0, 2.7, 100)
        hypo = np.sqrt(depi_range**2 + depth**2)
        Ipred = C1b + C2*mag + Betab*np.log10(hypo) + Gamma*hypo
#        if count_evt == 27:
#            print(Ipred)
#            print(Ipred[Ipred>=1])
        if np.any(Ipred>4):
            depi_range = depi_range[Ipred>=1]
            Hypo  = hypo[Ipred>=1]
            Ipred = Ipred[Ipred>=1]
            Io = C1b + C2*mag + Betab*np.log10(depth) + Gamma*depth
            Io = Io * np.ones(len(Ipred))
            QIo = 0.5 * np.ones(len(Ipred))
            StdI = 0.25 * np.ones(len(Ipred))
            StdlogR = 1.2 * np.ones(len(Ipred))
            Ndata = 10 * np.ones(len(Ipred))
            RegID = 201 * np.ones(len(Ipred))
            EVID = count_evt * np.ones(len(Ipred))
            temp = pd.DataFrame({'EVID' : EVID, 'Depi': depi_range, 'Hypo': Hypo,
                                 'I': Ipred, 'Io': Io, 'QIo': QIo, 'StdI' : StdI,
                                 'StdlogR': StdlogR, 'Ndata': Ndata, 'RegID': RegID})
            dataset_obs = pd.concat([dataset_obs, temp])
            dataset_evt.loc[count_evt, :] = [count_evt, mag, 0.1, depth, depth-0.5, depth+0.5]
            
            count_evt += 1
#            if count_evt == 28:
#                print(depi_range)
#                print(Ipred)





dataset_obs.to_csv('pytest_dataset08_obs.txt')
dataset_evt.to_csv('pytest_dataset08_evt.txt')

coeff = open('pytest_dataset08_coeff.txt', 'w')
coeff.write("C1a, C1b, C1c, C1d, C2, Betaa, Betab, Gamma\n")
coeff.write("{0},{1},{2},{3},{4},{5},{6},{7}".format(C1a, C1b, 0, 0, C2, Betaa, Betab, Gamma))
coeff.close()


