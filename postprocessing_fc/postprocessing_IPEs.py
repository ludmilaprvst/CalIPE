# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:45:36 2023

@author: PROVOST-LUD
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
            - C1: C1 coefficient value
            - C2: C2 coefficient value
            - beta: beta coefficient value
            - gamma: gamma coefficient value
            - I: observed (binned) intensity
            - Depi: epicentral distance associated to I
            - EVID: ID of the earthquake
            - Depth: hypocentral depth of the earthquake identified by EVID
            - Mag: magnitude of the earthquake identified by EVID
            
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
    obsbin.loc[:, 'within_residuals'] = obsbin.I.values - (obsbin.Ipred.values + obsbin.dImeanevt.values)
    #obsbin.loc[:, 'within_residuals'] = obsbin.I.values - obsbin.Ipred.values
    return obsbin

def plot_WEresiduals_Depi(outputname, path, ax, fig, evthighlight='None', plot_I0data=True, color_on='None'):
    """
    Plot the intensity within-residual of a given IPE in respect with epicentral distance

    Parameters
    ----------
    outputname : str
        Name of the .csv file which contains the observed intensities, associated epicentral
        distance, hypocentral depth and magnitude and the IPE coefficient.
        Mandatory columns are:
            - C1: C1 coefficient value
            - C2: C2 coefficient value
            - beta: beta coefficient value
            - gamma: gamma coefficient value
            - I: observed (binned) intensity
            - Depi: epicentral distance (km) associated to I 
            - EVID: ID of the earthquake
            - Depth: hypocentral depth of the earthquake identified by EVID
            - Mag: magnitude of the earthquake identified by EVID.
    path : str
        path to the folder where outputname is saved..
    ax : TYPE
        Matplotlib axe holding the plot.
    evthighlight : str or float, optional
        ID of the wished highlighted earthquake. One earthquake can have a different color on the plot. The default is 'None'.
    color_on: str
        Name of the obsbin column used to color the plot scatter points. The default is 'None', which means no special color is used.

    Returns
    -------
    None.

    """
    
    
    obsbin = compute_WEresiduals(outputname, path)
    
    if not plot_I0data:
        obsbin.dropna(subset='Io_ini', inplace=True)
    if color_on == 'None':
        color = None
    else:
        color = obsbin[color_on].values
    sc = ax.scatter(obsbin.Depi.values, obsbin.within_residuals.values, c=color)
    if not evthighlight == 'None':
        liste_evt = np.unique(obsbin.EVID.values)
        if evthighlight in liste_evt:
            tmp = obsbin[obsbin.EVID==evthighlight]
            ax.scatter(tmp.Depi.values, tmp.within_residuals.values, label=evthighlight)
    depi_mean_val = []
    depi_std_val = []
    depi_plot = []
    depi_bins = np.arange(0, obsbin.Depi.max()+10, 10)
    for i in range(len(depi_bins)-1):
        tmp = obsbin[np.logical_and(obsbin.Depi>=depi_bins[i], obsbin.Depi<depi_bins[i+1])]
        if len(tmp)>5:
            depi_mean_val.append(tmp.within_residuals.mean())
            depi_std_val.append(tmp.within_residuals.std())
            depi_plot.append(depi_bins[i])
    ax.errorbar(np.array(depi_plot)+5, depi_mean_val, yerr=depi_std_val,
                color='r', fmt='o', label='Mean within-event residual\nfor a 10 km epicentral interval')
    ax.axhline(y=0, lw=2, color='k')
    ax.set_xlabel('Epicentral distance [km]')
    ax.set_ylabel('Within-event intensity residual')
    ax.grid(which='both')
    
    if color_on != 'None':
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(sc, cax=cax, orientation='vertical')
    return obsbin

def plot_dI_Depi(outputname, path, ax, fig, evthighlight='None', plot_I0data=True, color_on='None'):
    """
    Plot the intensityresidual of a given IPE in respect with epicentral distance

    Parameters
    ----------
    outputname : str
        Name of the .csv file which contains the observed intensities, associated epicentral
        distance, hypocentral depth and magnitude and the IPE coefficient.
        Mandatory columns are:
            - C1: C1 coefficient value
            - C2: C2 coefficient value
            - beta: beta coefficient value
            - gamma: gamma coefficient value
            - I: observed (binned) intensity
            - Depi: epicentral distance (km) associated to I 
            - EVID: ID of the earthquake
            - Depth: hypocentral depth of the earthquake identified by EVID
            - Mag: magnitude of the earthquake identified by EVID.
    path : str
        path to the folder where outputname is saved..
    ax : TYPE
        Matplotlib axe holding the plot.
    evthighlight : str or float, optional
        ID of the wished highlighted earthquake. One earthquake can have a different color on the plot. The default is 'None'.
    color_on: str
        Name of the obsbin column used to color the plot scatter points. The default is 'None', which means no special color is used.

    Returns
    -------
    None.

    """
    
    
    obsbin = compute_WEresiduals(outputname, path)
    
    if not plot_I0data:
        obsbin.dropna(subset='Io_ini', inplace=True)
    if color_on == 'None':
        color = None
    else:
        color = obsbin[color_on].values
    sc = ax.scatter(obsbin.Depi.values, obsbin.dI.values, c=color)
    if not evthighlight == 'None':
        liste_evt = np.unique(obsbin.EVID.values)
        if evthighlight in liste_evt:
            tmp = obsbin[obsbin.EVID==evthighlight]
            ax.scatter(tmp.Depi.values, tmp.dI.values, label=evthighlight)
    depi_mean_val = []
    depi_std_val = []
    depi_plot = []
    depi_bins = np.append(np.arange(0, 40+10, 10), np.arange(50, obsbin.Depi.max()+20, 20))
    for i in range(len(depi_bins)-1):
        tmp = obsbin[np.logical_and(obsbin.Depi>=depi_bins[i], obsbin.Depi<depi_bins[i+1])]
        if len(tmp)>5:
            depi_mean_val.append(tmp.dI.mean())
            depi_std_val.append(tmp.dI.std())
            depi_plot.append(depi_bins[i]+(depi_bins[i+1]-depi_bins[i])/2)
    ax.errorbar(np.array(depi_plot), depi_mean_val, yerr=depi_std_val,
                color='r', fmt='o', label='Mean intensity residual\nfor a 10 km epicentral interval for depi<50 km\n and 20 km interval for depi>50 km')
    ax.axhline(y=obsbin.dI.mean(), lw=2, color='k')
    ax.set_xlabel('Epicentral distance [km]')
    ax.set_ylabel('Iobs-Ipred')
    ax.grid(which='both')
    
    if color_on != 'None':
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(sc, cax=cax, orientation='vertical')
    return obsbin
 
def plot_dI_I(outputname, path, ax, fig, evthighlight='None', plot_I0data=True, color_on='None'):
    """
    Plot the intensity residual of a given IPE in respect with observed intensity

    Parameters
    ----------
    outputname : str
        Name of the .csv file which contains the observed intensities, associated epicentral
        distance, hypocentral depth and magnitude and the IPE coefficient.
        Mandatory columns are:
            - C1: C1 coefficient value
            - C2: C2 coefficient value
            - beta: beta coefficient value
            - gamma: gamma coefficient value
            - I: observed (binned) intensity
            - Depi: epicentral distance (km) associated to I 
            - EVID: ID of the earthquake
            - Depth: hypocentral depth of the earthquake identified by EVID
            - Mag: magnitude of the earthquake identified by EVID.
    path : str
        path to the folder where outputname is saved..
    ax : TYPE
        Matplotlib axe holding the plot.
    evthighlight : str or float, optional
        ID of the wished highlighted earthquake. One earthquake can have a different color on the plot. The default is 'None'.
    color_on: str
        Name of the obsbin column used to color the plot scatter points. The default is 'None', which means no special color is used.

    Returns
    -------
    None.

    """
    
    obsbin = compute_WEresiduals(outputname, path)
    
    if not plot_I0data:
        obsbin.dropna(subset='Io_ini', inplace=True)
    if color_on == 'None':
        color = None
    else:
        color = obsbin[color_on].values    
    sc = ax.scatter(obsbin.I.values, obsbin.dI.values, c=color)
    ival_mean_val = []
    ival_std_val = []
    ival_plot = []
    for ival in np.unique(obsbin.I.values):
        tmp = obsbin[obsbin.I==ival]
        if len(tmp)>5:
            ival_mean_val.append(tmp.within_residuals.mean())
            ival_std_val.append(tmp.within_residuals.std())
            ival_plot.append(ival)
    ax.errorbar(ival_plot, ival_mean_val, yerr=ival_std_val,
                color='r', fmt='o', label='Mean intensity residual\nfor the observed intensity value')
    if not evthighlight == 'None':
        liste_evt = np.unique(obsbin.EVID.values)
        if evthighlight in liste_evt:
            tmp = obsbin[obsbin.EVID==evthighlight]
            ax.scatter(tmp.I.values, tmp.dI.values, label=evthighlight)
    ax.axhline(y=obsbin.dI.mean(), lw=2, color='k')
    ax.set_xlabel('Observed intensity')
    ax.set_ylabel('Iobs-Ipred')
    ax.grid(which='both')
    
    if color_on != 'None':
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(sc, cax=cax, orientation='vertical')
    return obsbin
   
def plot_WEresiduals_I(outputname, path, ax, fig, evthighlight='None', plot_I0data=True, color_on='None'):
    """
    Plot the intensity within-residual of a given IPE in respect with observed intensity

    Parameters
    ----------
    outputname : str
        Name of the .csv file which contains the observed intensities, associated epicentral
        distance, hypocentral depth and magnitude and the IPE coefficient.
        Mandatory columns are:
            - C1: C1 coefficient value
            - C2: C2 coefficient value
            - beta: beta coefficient value
            - gamma: gamma coefficient value
            - I: observed (binned) intensity
            - Depi: epicentral distance (km) associated to I 
            - EVID: ID of the earthquake
            - Depth: hypocentral depth of the earthquake identified by EVID
            - Mag: magnitude of the earthquake identified by EVID.
    path : str
        path to the folder where outputname is saved..
    ax : TYPE
        Matplotlib axe holding the plot.
    evthighlight : str or float, optional
        ID of the wished highlighted earthquake. One earthquake can have a different color on the plot. The default is 'None'.
    color_on: str
        Name of the obsbin column used to color the plot scatter points. The default is 'None', which means no special color is used.

    Returns
    -------
    None.

    """
    
    obsbin = compute_WEresiduals(outputname, path)
    
    if not plot_I0data:
        obsbin.dropna(subset='Io_ini', inplace=True)
    if color_on == 'None':
        color = None
    else:
        color = obsbin[color_on].values    
    sc = ax.scatter(obsbin.I.values, obsbin.within_residuals.values, c=color)
    if not evthighlight == 'None':
        liste_evt = np.unique(obsbin.EVID.values)
        if evthighlight in liste_evt:
            tmp = obsbin[obsbin.EVID==evthighlight]
            ax.scatter(tmp.I.values, tmp.within_residuals.values, label=evthighlight)
    ax.axhline(y=0, lw=2, color='k')
    ax.set_xlabel('Observed intensity')
    ax.set_ylabel('Within-event intensity residual')
    ax.grid(which='both')
    
    if color_on != 'None':
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(sc, cax=cax, orientation='vertical')
    return obsbin

def plot_BEresiduals_Mag(outputname, path, ax, fig, evthighlight='None', plot_I0data=True, color_on='None'):
    """
    Plot the intensity within-residual of a given IPE in respect with epicentral distance

    Parameters
    ----------
    outputname : str
        Name of the .csv file which contains the observed intensities, associated epicentral
        distance, hypocentral depth and magnitude and the IPE coefficient.
        Mandatory columns are:
            - C1: C1 coefficient value
            - C2: C2 coefficient value
            - beta: beta coefficient value
            - gamma: gamma coefficient value
            - I: observed (binned) intensity
            - Depi: epicentral distance (km) associated to I 
            - EVID: ID of the earthquake
            - Depth: hypocentral depth of the earthquake identified by EVID
            - Mag: magnitude of the earthquake identified by EVID.
    path : str
        path to the folder where outputname is saved..
    ax : TYPE
        Matplotlib axe holding the plot.
    evthighlight : str or float, optional
        ID of the wished highlighted earthquake. One earthquake can have a different color on the plot. The default is 'None'.
    color_on: str
        Name of the obsbin column used to color the plot scatter points. The default is 'None', which means no special color is used.

    Returns
    -------
    None.

    """
    
    
    obsbin = compute_WEresiduals(outputname, path)
    
    if not plot_I0data:
        obsbin.dropna(subset='Io_ini', inplace=True)
    if color_on == 'None':
        color = None
    else:
        color = obsbin[color_on].values
    sc = ax.scatter(obsbin.Mag.values, obsbin.dImeanevt.values, c=color)
    mag_bins = np.arange(obsbin.Mag.min(), obsbin.Mag.max()+0.25, 0.25)
    mag_mean_val = []
    mag_std_val = []
    mag_plot = []
    for i in range(len(mag_bins)-1):
        tmp = obsbin[np.logical_and(obsbin.Mag>=mag_bins[i], obsbin.Mag<mag_bins[i+1])]
        if len(tmp)>5:
            mag_mean_val.append(tmp.dImeanevt.mean())
            mag_std_val.append(tmp.dImeanevt.std())
            mag_plot.append(mag_bins[i])
    ax.errorbar(np.array(mag_plot)+0.125, mag_mean_val, yerr=mag_std_val,
                color='r', fmt='o', label='Mean between-event residual\nfor a 0.25 magnitude interval')
    if not evthighlight == 'None':
        liste_evt = np.unique(obsbin.EVID.values)
        if evthighlight in liste_evt:
            tmp = obsbin[obsbin.EVID==evthighlight]
            ax.scatter(tmp.Mag.values, tmp.dImeanevt.values, label=evthighlight)
    ax.axhline(y=obsbin.dImeanevt.mean(), lw=2, color='k')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Between-event intensity residual')
    ax.grid(which='both')
    
    if color_on != 'None':
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(sc, cax=cax, orientation='vertical')
    return obsbin