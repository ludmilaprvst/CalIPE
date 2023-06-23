# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:45:36 2023

@author: PROVOST-LUD
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def apply_IPE(C1, C2, beta, gamma, depth, mag, depi):
    """
    Apply an IPE to epicentral distance data, for given depth and magnitude corresponding to one earthquake

    Parameters
    ----------
    C1 : float or array
        C1 coefficient value.
    C2 : float or array
        C2 coefficient value.
    beta : float or array
        beta coefficient value.
    gamma : float or array
        gamma coefficient value.
    depth : float
        Hypocentral depth of the earthquake.
    mag : float
        Magnitude of the earthquake.
    depi : float or array
        Epicentral distance for which the predicted intensity is computed.

    Returns
    -------
    float or array
        intensity predicted by the input IPE for the given depth and magnitude at epicentral distance depi.

    """
    hypo = np.sqrt(depth**2+depi**2)
    return C1 + C2*mag + beta*np.log10(hypo)+gamma*hypo

def compute_WEresiduals(outputname, path):
    """
    
    Compute wihtin residuals for a given IPE and a list of given observations
    
    Parameters
    ----------
    outputname : str
        Name of the .csv file which contains the observed intensities, associated epicentral
        distance, hypocentral depth and magnitude and the IPE coefficient.
        Mandatory columns are:
            C1 : C1 coefficient value
            C2 : C2 coefficient value
            beta : beta coefficient value
            gamma : gamma coefficient value
            I: observed intensity
            Depi : epicentral distance associated to I
            EVID : ID of the earthquake
            Depth : hypocentral depth of the earthquake identified by EVID
            Mag : magnitude of the earthquake identified by EVID
            
    path : str
        path to the folder where outputname is saved.

    Returns
    -------
    pandas.DataFrame
        dataframe with a mean intensity residual per earthquake columns (dImeanevt) and a within_residual column.

    """
    obsbin = pd.read_csv(path + '/' + outputname)
    obsbin.loc[:, 'Ipred'] = apply_IPE(obsbin.C1.values, obsbin.C2.values,
                                       obsbin.beta.values, obsbin.gamma.values,
                                       obsbin.Depth.values, obsbin.Mag.values,
                                       obsbin.Depi.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    obsbin_gp = obsbin.groupby('EVID').mean()
    dict_meandIevt = obsbin_gp.to_dict(orient='index')
    obsbin.loc[:, 'dImeanevt'] = obsbin.apply(lambda row: dict_meandIevt[row['EVID']]['dI'], axis=1)
    obsbin.loc[:, 'within_residuals'] = obsbin.I.values - (obsbin.Ipred.values - obsbin.dImeanevt.values)
    return obsbin

def plot_WEresiduals_Depi(outputname, path, ax, evthighlight='None', plot_I0data=True):
    """
    Plot the intensity within-residual of a given IPE in respect with epicentral distance

    Parameters
    ----------
    outputname : str
        Name of the .csv file which contains the observed intensities, associated epicentral
        distance, hypocentral depth and magnitude and the IPE coefficient.
        Mandatory columns are:
            C1 : C1 coefficient value
            C2 : C2 coefficient value
            beta : beta coefficient value
            gamma : gamma coefficient value
            I: observed intensity
            Depi : epicentral distance (km) associated to I 
            EVID : ID of the earthquake
            Depth : hypocentral depth of the earthquake identified by EVID
            Mag : magnitude of the earthquake identified by EVID.
    path : str
        path to the folder where outputname is saved..
    ax : TYPE
        Matplotlib axe holding the plot.
    evthighlight : str or float, optional
        ID of the wished highlighted earthquake. One earthquake can have a different color on the plot. The default is 'None'.

    Returns
    -------
    None.

    """
    
    obsbin = compute_WEresiduals(outputname, path)
    if not plot_I0data:
        obsbin.dropna(subset='Io_ini', inplace=True)
    ax.scatter(obsbin.Depi.values, obsbin.within_residuals.values)
    if not evthighlight == 'None':
        liste_evt = np.unique(obsbin.EVID.values)
        if evthighlight in liste_evt:
            tmp = obsbin[obsbin.EVID==evthighlight]
            ax.scatter(tmp.Depi.values, tmp.within_residuals.values, label=evthighlight)
    ax.axhline(y=0, lw=2, color='k')
    ax.set_xlabel('Epicentral distance [km]')
    ax.set_ylabel('Within intensity residual')
    ax.grid(which='both')
    return obsbin
    
def plot_WEresiduals_I(outputname, path, ax, evthighlight='None', plot_I0data=True):
    """
    Plot the intensity within-residual of a given IPE in respect with observed intensity

    Parameters
    ----------
    outputname : str
        Name of the .csv file which contains the observed intensities, associated epicentral
        distance, hypocentral depth and magnitude and the IPE coefficient.
        Mandatory columns are:
            C1 : C1 coefficient value
            C2 : C2 coefficient value
            beta : beta coefficient value
            gamma : gamma coefficient value
            I: observed intensity
            Depi : epicentral distance (km) associated to I 
            EVID : ID of the earthquake
            Depth : hypocentral depth of the earthquake identified by EVID
            Mag : magnitude of the earthquake identified by EVID.
    path : str
        path to the folder where outputname is saved..
    ax : TYPE
        Matplotlib axe holding the plot.
    evthighlight : str or float, optional
        ID of the wished highlighted earthquake. One earthquake can have a different color on the plot. The default is 'None'.

    Returns
    -------
    None.

    """
    
    obsbin = compute_WEresiduals(outputname, path)
    if not plot_I0data:
        obsbin.dropna(subset='Io_ini', inplace=True)
    ax.scatter(obsbin.I.values, obsbin.within_residuals.values)
    if not evthighlight == 'None':
        liste_evt = np.unique(obsbin.EVID.values)
        if evthighlight in liste_evt:
            tmp = obsbin[obsbin.EVID==evthighlight]
            ax.scatter(tmp.I.values, tmp.within_residuals.values, label=evthighlight)
    ax.axhline(y=0, lw=2, color='k')
    ax.set_xlabel('Observed intensity')
    ax.set_ylabel('Within intensity residual')
    ax.grid(which='both')
    return obsbin
    