# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 16:23:37 2021

@author: PROVOST-LUD
"""

import pandas as pd
import numpy as np
import library_bin as libr


Stdobs = {'A':0.5,'B':0.577,'C':0.710,'D':1.0,'I':1.5,'K':2.0}
 


def RAVG(obsdata, depth, Ic, I0, QI0):
    """
    dans obsdata, que un evt.
    """
    #print(obsdata.EVID.values[0])
    #print(obsdata.EVID.values[0])
    #print(Ic)
    obsdata = obsdata[obsdata.Iobs>=Ic]
    #print(obsdata.EVID.values[0])
    obsdata.loc[:, 'poids'] = obsdata.apply(lambda row: 1/Stdobs[row['QIobs']]**2, axis=1)
    obsdata.loc[:, 'LogDepi'] = obsdata.apply(lambda row: np.log10(row['Depi']), axis=1)
    
    grouped_byIbin = obsdata.groupby('Iobs')
    wm = lambda x: np.average(x, weights=obsdata.loc[x.index, "poids"])
    varwm = lambda x: np.sum(obsdata.loc[x.index, "poids"]*(np.average(x, weights=obsdata.loc[x.index, "poids"]) - x)**2)/np.sum(obsdata.loc[x.index, "poids"])
    obsbin = grouped_byIbin.agg({'LogDepi':[("RAVG", wm), ("VarLogR", varwm)],
                                'EVID': [('Ndata', 'count')]})
    obsbin.columns = obsbin.columns.get_level_values(1)
    obsbin.reset_index(inplace=True)
    obsbin.loc[:, 'EVID'] = obsdata.EVID.values[0]
    obsbin.loc[:, 'I0'] = I0
    obsbin.loc[:, 'QI0'] = QI0
    obsbin.loc[:, 'Depi'] = obsbin.apply(lambda row: 10**row['RAVG'], axis=1)
    obsbin.loc[:, 'Hypo'] = obsbin.apply(lambda row: np.sqrt(row['Depi']**2+depth**2), axis=1)
    obsbin.loc[:, 'StdLogR'] = obsbin.apply(lambda row: np.sqrt(row['VarLogR']), axis=1)
    obsbin.loc[:, 'StdI_data'] = obsbin.apply(lambda row: np.abs(-3.5)*row['StdLogR'], axis=1)
    obsbin.loc[:, 'StdI_min'] = obsbin.apply(lambda row: Stdobs['C']/np.sqrt(row['Ndata']), axis=1)
    obsbin.loc[:, 'StdI'] = obsbin.apply(lambda row: np.max([row['StdI_data'], row['StdI_min']]), axis=1)
    
    obsbin = obsbin[['EVID', 'Iobs', 'Depi', 'Hypo', 'StdLogR', 'StdI', 'I0', 'QI0', 'Ndata']]
    obsbin.columns = ['EVID', 'I', 'Depi', 'Hypo', 'StdLogR', 'StdI', 'Io', 'Io_std', 'Ndata']
    return obsbin


def Binning_Obs(obsdata, depth, Ic, I0, QI0, evid):
    # Old version for binning RAVG
        Stdobs = {'A':0.5,'B':0.577,'C':0.710,'D':1.0,'I':1.5,'K':2.0}
        colonnes_binn = ['EVID','Hypo','I','Io','QIo','StdI','StdLogR','Ndata']
        Depi = []
        for epi in obsdata['Depi'].values:
            Depi.append(epi)
        Depi = np.array(Depi)
        
        Hypo = libr.Distance_c(depth, Depi)
        IOBS = []
        for iob in obsdata['Iobs'].values:
            IOBS.append(iob)
        Iobs = np.array(IOBS)

        QIOBS = []
        for qiob in obsdata['QIobs'].values:
            QIOBS.append(Stdobs[qiob])
        QIobs=np.array(QIOBS)

        
        

        depth = float(depth)
        evid = int(evid)
        Ic = float(Ic)
        SortieBinn = libr.RAVG_c(Iobs,Hypo,QIobs,I0,QI0,depth,evid, Ic,30)
        ObsBinn = pd.DataFrame(data=SortieBinn, columns=colonnes_binn)
        ObsBinn = ObsBinn[ObsBinn['EVID'] != 0]
        return ObsBinn

obsdata = pd.read_csv('../Testpy_dataset/obs_test.txt', sep='\t')
I0 = 8
QI0 = 0.5
Ic = 6
depth = 0

# test vitesse RAVG : 0.026 s Binning_Obs : 0.012 s