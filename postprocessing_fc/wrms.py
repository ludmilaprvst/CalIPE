# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:48:29 2021

@author: PROVOST-LUD
"""
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable

def calcul_wrms_evt(obsbin, Beta, I0, depth, Gamma=0):
    """
    Compute the WRMS for the Koveslighety equation for ine earthquake.

    Parameters
    ----------
    obsbin : pandas.DataFrame
        Dataframe with the observed intensity and their associated epicentral 
        depth for the given earthquake. Mandatory columns are:
            - I: observed intensity
            - Depi: associated epicentral intensity
            - StdI: standard deviation/uncertainty associated to the intensity value
    Beta : float
        Geometric attenuation coefficient.
    I0 : float
        Epicentral intensity.
    depth : float
        Hypocentral depth.
    Gamma : float, optional
        Intresic attenuation coefficient. The default is 0.

    Returns
    -------
    WRMS : float
        WRMS computed by comparing observed intensity and predicted intensity.

    """
    # pas testee
    Hypo = np.sqrt(obsbin.Depi.values**2 + depth**2)
    Ipred = I0 + Beta * np.log10(Hypo/depth) + Gamma*Hypo/depth
    Cd = obsbin['StdI'].values**2
    Wd = 1./Cd
    Wd = Wd/sum(Wd)
    dI = obsbin.I.values - Ipred
    WRMS = np.sum((dI**2)*Wd)/np.sum(Wd)
    return WRMS

def calcul_wrms_beta(obsdata, Beta, Gamma=0):
    """
    For one earthquake, compute the WRMS for the Koveslighety equation for a given coefficient beta values.

    Parameters
    ----------
    obsdata : pandas.DataFrame
        Dataframe with the observed intensity and their associated epicentral 
        depth for the given earthquake. Mandatory columns are:
            - I: observed intensity
            - Depi: associated epicentral intensity
            - StdI: standard deviation/uncertainty associated to the intensity value
            - Depth: hypocentral depth
            - Io: epicentral intensity
    Beta : float
        Value of the geometric attenuation coefficient beta.
     Gamma : float, optional
         Intresic attenuation coefficient. The default is 0.

    Returns
    -------
    WRMS : float
        WRMS computed by comparing observed intensity and predicted intensity.
    obsdata : pandas.DataFrame
        Dataframe with the observed intensities and their associated epicentral and
        the predicted intensities.

    """
    
    obsdata.loc[:, 'Hypo'] = np.sqrt(obsdata.loc[:, 'Depi']**2 + obsdata.loc[:, 'Depth']**2)
    obsdata.loc[:, 'logterm'] = np.log10(obsdata.loc[:, 'Hypo']/obsdata.loc[:, 'Depth'])
    obsdata.loc[:, 'Ipred'] = obsdata.loc[:, 'Io'] + Beta*obsdata.loc[:, 'logterm'] + Gamma*obsdata.loc[:, 'Hypo']/obsdata.loc[:, 'Depth']
    obsdata.loc[:, 'dI'] = np.abs(obsdata.loc[:, 'I'] - obsdata.loc[:, 'Ipred'])
    obsdata.loc[:, 'Wd'] = 1/obsdata['StdI'].values**2
    obsdata.loc[:, 'Wd'] = obsdata.loc[:, 'Wd']/np.sum(obsdata.loc[:, 'Wd'])
    WRMS = np.sum((obsdata.loc[:, 'dI']**2)*obsdata.loc[:, 'Wd'])/np.sum(obsdata.loc[:, 'Wd'])
    return WRMS, obsdata

def apply_IPE(Hypo, Mag, C1, C2, beta, gamma):
    """
    Function that apply the following equation:
        I = C1 + C2*Mag + Beta*log10(Hypo) + Gamma*Hypo

    Parameters
    ----------
    Hypo : numpy.array
        Array with the hypocentral distances in km for which the intensities will be computed.
    Mag : float
        magnitude of the earthquake for which the intensities will be computed.
    C1 : float
        C1 coefficient.
    C2 : float
        C2 coefficient.
    beta : float
        Geometric attenuation coefficient.
    gamma : float
        Intresic attenuation.

    Returns
    -------
    numpy.array
        Predicted intensities associated to the hypocentral distances Hypo.

    """
    
    return C1 + C2*Mag + beta*np.log10(Hypo)+gamma*Hypo

def calcul_wrms_C1C2betagamma(obsdata, C1, C2, Beta, Gamma):
    """
    Compute the WRMS for the following equation:
        I = C1 + C2*Mag + Beta*log10(Hypo) + Gamma*Hypo

    Parameters
    ----------
    obsdata : pandas.DataFrame
        DataFrame which contain for one earthquake the observed intensities, the associated epicentral distances,
        magnitude and hypocentral depth. Mandatory columns are:
            - I: observed intensity
            - Depi: associated epicentral intensity
            - StdI: standard deviation/uncertainty associated to the intensity value
            - Depth: hypocentral depth
            - Mag: magnitude
    C1 : float
        C1 coefficient.
    C2 : float
        C2 coefficient.
    Beta : float
        Geometric attenuation coefficient.
    Gamma : float
        Intresic attenuation.

    Returns
    -------
    WRMS : float
        WRMS computed.

    """
    
    obsdata.loc[:, 'Hypo'] = np.sqrt(obsdata.loc[:, 'Depi']**2 + obsdata.loc[:, 'Depth']**2)
    obsdata.loc[:, 'Ipred'] = obsdata.apply(lambda row: apply_IPE(row['Hypo'], row['Mag'], C1, C2, Beta, Gamma), axis=1)
    obsdata.loc[:, 'dI'] = np.abs(obsdata.loc[:, 'I'] - obsdata.loc[:, 'Ipred'])
    obsdata.loc[:, 'Wd'] = 1/obsdata['StdI'].values**2
    obsdata.loc[:, 'Wd'] = obsdata.loc[:, 'Wd']/np.sum(obsdata.loc[:, 'Wd'])
    WRMS = np.sum((obsdata.loc[:, 'dI']**2)*obsdata.loc[:, 'Wd'])/np.sum(obsdata.loc[:, 'Wd'])
    return WRMS

def get_HI0_wrms_space(obsdata, evid, Beta, minH=1, maxH=25, pasH=0.25,
                       minI0=2, maxI0=10,  pasI0=0.1):
    """
    For a given earthquake, compute the WRMS for different depths and epicentral intensity and a given attenuation
    coefficient Beta for the Koveslighety equation wih gamma equal to 0.

    Parameters
    ----------
    obsdata : pandas.DataFrame
        Dataframe with the observed intensities and associated epicentral distances.
    evid : str, float, int
        Id of the earthquake.
    Beta : float
        geometric attenuation coefficient beta.
    minH : float, optional
        Minimal depth (km) for which the WRMS will be computed. The default is 1.
    maxH : float, optional
        Maximal depth (km) for which the WRMS will be computed. The default is 25.
    pasH : float, optional
        Step in km between two depths for which the WRMS will be computed. The default is 0.25.
    minI0 : float, optional
        Minimal epicentral intensity for which the WRMS will be computed. The default is 2.
    maxI0 : float, optional
        Maximal epicentral intensity for which the WRMS will be computed. The default is 10.
    pasI0 : float, optional
        Step  between two values of epicentral intensity for which the WRMS will be computed. The default is 0.1.

    Returns
    -------
    hh_plot : numpy.array
        depths for which the WRMS has been computed
    io_plot : numpy.array
        epicentral intensities for which the WRMS has been computed
    wrms : numpy.array
        Computed WRMS. The order of the WRMS correspond to the hh_plot and io_plot orders.

    """
    
    obsbin = obsdata[obsdata.EVID==evid]
    wrms = np.array([])
    io_plot = np.array([])
    hh_plot = np.array([])
    for hh in np.arange(minH, maxH+pasH, pasH):
        for io in np.arange(minI0, maxI0+pasI0, pasI0):
            wrms_i = calcul_wrms_evt(obsbin, Beta, io, hh)
            wrms = np.append(wrms, wrms_i)
            io_plot = np.append(io_plot, io)
            hh_plot = np.append(hh_plot, hh)      
    return hh_plot, io_plot, wrms

def getHline_in_HI0wrms_space(obsdata, evid, Beta, minH=1, maxH=25, pasH=0.25,
                              minI0=2, maxI0=10, pasI0=0.1):
    """
    Get a section in the 2D depth/epicentral intensity WRMS space for one earthquake.
    The section is collected for the depth equal to the inverted depth during the attenuation calibration. 
    The WRMS space is computed with the Koveslighety equation, with gamma equal to 0.

    Parameters
    ----------
    obsdata : pandas.DataFrame
        dataframe with the observed intensities and the inverted epicentral
        intensity. The mandatory columns are:
            - EVID: ID of the earthquake
            - I: observed intensity
            - Depi: associated epicentral distance
            - StdI: standard deviation/uncertainty associated to I value
            - Io: value of the inverted epicentral intensity
    evid : float, int, str
        ID of the earthquake.
    Beta : float
        geometric attenuation coefficient.
    minH : float, optional
        Minimal depth (km) for which the WRMS will be computed. The default is 1.
    maxH : float, optional
        Maximal depth (km) for which the WRMS will be computed. The default is 25.
    pasH : float, optional
        Step in km between two depths for which the WRMS will be computed. The default is 0.25.
    minI0 : float, optional
        Minimal epicentral intensity for which the WRMS will be computed. The default is 2.
    maxI0 : float, optional
        Maximal epicentral intensity for which the WRMS will be computed. The default is 10.
    pasI0 : float, optional
        Step  between two values of epicentral intensity for which the WRMS will be computed. The default is 0.1.

    Returns
    -------
    line_wrms_hinv : TYPE
        DESCRIPTION.

    """
    
    obsbin = obsdata[obsdata.EVID==evid]
    depth = obsbin.Depth.values[0]
    # H line
    line_wrms_hinv = np.array([])
    for io in np.arange(minI0, maxI0+pasI0, pasI0):
        wrms_i = calcul_wrms_evt(obsbin, Beta, io, depth)
        line_wrms_hinv = np.append(line_wrms_hinv, wrms_i)
    return line_wrms_hinv

def getI0line_in_HI0wrms_space(obsdata, evid, Beta, minH=1, maxH=25, pasH=0.25,
                              minI0=2, maxI0=10, pasI0=0.1):
    """
    Get a section in the 2D depth/epicentral intensity WRMS space for one earthquake.
    The section is collected for the epicentral intensity equal to the inverted epicentral
    intensity during the attenuation calibration. 
    The WRMS space is computed with the Koveslighety equation, with gamma equal to 0.

    Parameters
    ----------
    obsdata : pandas.DataFrame
        dataframe with the observed intensities and the inverted epicentral
        intensity. The mandatory columns are:
            - EVID: ID of the earthquake
            - I: observed intensity
            - Depi: associated epicentral distance
            - StdI: standard deviation/uncertainty associated to I value
            - Io: value of the inverted epicentral intensity
    evid : float, int, str
        ID of the earthquake.
    Beta : float
        geometric attenuation coefficient.
    minH : float, optional
        Minimal depth (km) for which the WRMS will be computed. The default is 1.
    maxH : float, optional
        Maximal depth (km) for which the WRMS will be computed. The default is 25.
    pasH : float, optional
        Step in km between two depths for which the WRMS will be computed. The default is 0.25.
    minI0 : float, optional
        Minimal epicentral intensity for which the WRMS will be computed. The default is 2.
    maxI0 : float, optional
        Maximal epicentral intensity for which the WRMS will be computed. The default is 10.
    pasI0 : float, optional
        Step  between two values of epicentral intensity for which the WRMS will be computed. The default is 0.1.

    Returns
    -------
    line_wrms_ioinv : numpy.array
        WRMS value for epicentral intensity equal to the inverted epicentral intensity.

    """
    
    obsbin = obsdata[obsdata.EVID==evid]
    I0 = obsbin.Io.values[0]
     # Io line
    line_wrms_ioinv = np.array([])
    for hh in np.arange(minH, maxH+pasH, pasH):
        wrms_i = calcul_wrms_evt(obsbin, Beta, I0, hh)
        line_wrms_ioinv = np.append(line_wrms_ioinv, wrms_i)
    return line_wrms_ioinv

def plot_HI0_wrms_space(ax, ax_cb, fig_wrms, hh_plot, io_plot, wrms, minH, maxH, minI0, maxI0,
                        vmin=0, vmax=-99):
    """
    Plot the 2D WRMS space in function of depth and epicentral intensity, with fixed attenuation coefficients.
    xaxis is the depth, yaxis is the epicentral intensity and the color is the WRMS value.
    

    Parameters
    ----------
    ax : matplotlib.axes
        axe where the 2D WRMS space is plotted.
    ax_cb : matplotlib.axes
        axe where the corresponding colorbar is plotted.
    fig_wrms : matplotlib.figures
        Figure containing both ax and ax_cb.
    hh_plot : numpy.array
        depths for which the WMRS is computed.
    io_plot : numpy.array
        epicentral intensities for which the WMRS is computed.
    wrms : numpy.array
        Computed WRMS. The order of the WRMS correspond to the hh_plot and io_plot orders..
    minH : float, optional
        Minimal depth (km) for which the WRMS will be computed. The default is 1.
    maxH : float, optional
        Maximal depth (km) for which the WRMS will be computed. The default is 25.
    minI0 : float, optional
        Minimal epicentral intensity for which the WRMS will be computed. The default is 2.
    maxI0 : float, optional
        Maximal epicentral intensity for which the WRMS will be computed. The default is 10.
    vmin : float, optional
        Minimal value of WRMS used to define the colormap. The default is 0.
    vmax : float, optional
        Maximal value of WRMS used to define the colormap. The default is -99. In this case,
        the maximal value of all computed WRMS is used.

    Returns
    -------
    None.

    """
    
    if vmax==-99:
        vmax = np.max(wrms)
    xi, yi = np.mgrid[minH:maxH:100j, minI0:maxI0:100j]
    points_hhio = []
    for xx in range(len(hh_plot)):
        points_hhio.append([hh_plot[xx], io_plot[xx]])
    points_hhio = np.array(points_hhio)
    zi = griddata(points_hhio, wrms, (xi, yi), method='linear')
    
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes('left', size='5%', pad=1)
    
    im = ax.imshow(zi.T, origin='lower', vmin=0, vmax=vmax,
              extent=[minH, maxH, minI0, maxI0], aspect='auto',
              interpolation='nearest', cmap=plt.cm.get_cmap('terrain'))
    fig_wrms.colorbar(im, cax=ax_cb, orientation='horizontal', label='WRMS')


def classic_config_wmrsevt(fig_wrms, minI0, maxI0, minH, maxH):
    """
    Build the classical matplotlib figure configuration to plot the 2D WRMS space
    in function of depth and epicentral intenisty, with fixed attenuation coefficient.
    The WRMS space is computed with the Koveslighety equation.
    
    Parameters
    ----------
    fig_wrms : matplotlib.figures
        figure on which the configuration is built.
    minI0 : float
        Minimal epicentral intensity for which the WRMS is computed.
    maxI0 : float
        Maximal epicentral intensity for which the WRMS is computed.
    minH : float
        Minimal depth (km) for which the WRMS is computed.
    maxH : float
        Minimal depth (km) for which the WRMS is computed.

    Returns
    -------
    ax_iohh : matplotlib.axes
        axe on which the 2D WRMS space in function of depth and epicentral intensity will be plotted.
    ax_hh : matplotlib.axes
        axe on which a section of the 2D WRMS space can be plotted.
        The section is associated to a given epicentral intensity. The section is then the WRMS in function of depth for a given epicentral intensity.
    ax_io : matplotlib.axes
        axe on which a section of the 2D WRMS space can be plotted.
        The section is associated to a given depth. The section is then the WRMS in function of epicentral intensity for a depth.
    ax_cb : matplotlib.axes
        axe which will support the colorbar associated to the 2D WRMS space.

    """
    
    gs0 = gridspec.GridSpec(3, 2, width_ratios=(7, 2), height_ratios=(2, 7, 0.25),
                            hspace=0.4)
    
    ax_iohh = fig_wrms.add_subplot(gs0[1, 0])
    ax_iohh.set_ylim([minI0, maxI0])
    ax_iohh.set_xlim([minH, maxH])
    ax_io = fig_wrms.add_subplot(gs0[1, 1], sharey=ax_iohh)
    ax_hh = fig_wrms.add_subplot(gs0[0, 0], sharex=ax_iohh)
    ax_cb = fig_wrms.add_subplot(gs0[2, :])
    return ax_iohh, ax_hh, ax_io, ax_cb

def plot_wrms_withHI0lines(fig_wrms,
                           obsdata, evid, Beta,
                           minH=1, maxH=25, pasH=0.25,
                           minI0=2, maxI0=10, pasI0=0.1,
                           vmax=-99):
    """
    Plot the 2D WRMS space in depth and epicentral intensity with two sections.
    The two section of the 2D space are sections where the epicentral intensity is equal
    to the inverted epicentral intensity and where the depth is equal to the inverted depth.
    
    Parameters
    ----------
    fig_wrms : matplotlib.figures
        figure on which the plot of the 2D WRMS space and its sections are built..
    obsdata : pandas.DataFrame
        dataframe with the observed intensities and the inverted epicentral
        intensity. The mandatory columns are:
            - EVID: ID of the earthquake
            - I: observed intensity
            - Depi: associated epicentral distance
            - StdI: standard deviation/uncertainty associated to I value
            - Io: value of the inverted epicentral intensity.
    evid : float, int, str
        ID of the earthquake.
    Beta : float
        geometric attenuation coefficient.
    minH : float, optional
        Minimal depth (km) for which the WRMS will be computed. The default is 1.
    maxH : float, optional
        Maximal depth (km) for which the WRMS will be computed. The default is 25.
    pasH : float, optional
        Step in km between two depths for which the WRMS will be computed. The default is 0.25.
    minI0 : float, optional
        Minimal epicentral intensity for which the WRMS will be computed. The default is 2.
    maxI0 : float, optional
        Maximal epicentral intensity for which the WRMS will be computed. The default is 10.
    pasI0 : float, optional
        Step between two values of epicentral intensity for which the WRMS will be computed. The default is 0.1.

    Returns
    -------
    ax_iohh : matplotlib.axes
        Axe where the 2D WRMS space is plotted.
    ax_hh : matplotlib.axes
        Axe where the section where the depth is equal to the inverted depth is plotted.
    ax_io : matplotlib.axes
        Axe where the section where the epicentral intensity is equal to the inverted epicentral intensity is plotted.

    """
    
    hh_plot, io_plot, wrms = get_HI0_wrms_space(obsdata, evid, Beta,
                                                minH, maxH, pasH,
                                                minI0, maxI0, pasI0)
    line_wrms_hinv = getHline_in_HI0wrms_space(obsdata, evid, Beta,
                                               minH, maxH,pasH,
                                               minI0, maxI0, pasI0)
    line_wrms_ioinv = getI0line_in_HI0wrms_space(obsdata, evid, Beta,
                                                 minH, maxH, pasH,
                                                 minI0, maxI0, pasI0)
    obsbin = obsdata[obsdata.EVID==evid]
    I0 = obsbin.Io.values[0]
    depth = obsbin.Depth.values[0]
    
    #cax = fig_wrms.add_axes([0.27, 0.2, 0.5, 0.05])
    ax_iohh, ax_hh, ax_io, ax_cb = classic_config_wmrsevt(fig_wrms, minI0, maxI0, minH, maxH)
    
    plot_HI0_wrms_space(ax_iohh, ax_cb, fig_wrms,
                        hh_plot, io_plot, wrms, minH, maxH, minI0, maxI0,
                        vmin=0, vmax=vmax)
    ax_iohh.axvline(x=depth, color='w')
    ax_iohh.axhline(y=I0, color='w')
    
    ax_io.set_facecolor("Gray")
    ax_io.plot(line_wrms_hinv,
               np.arange(minI0, maxI0+pasI0, pasI0), color='w')

    ax_hh.set_facecolor("Gray")
    ax_hh.plot(np.arange(minH, maxH+pasH, pasH),
               line_wrms_ioinv, color='w')
    
    ax_iohh.set_xlabel('Depth [km]')
    ax_iohh.set_ylabel('Io')
    ax_hh.set_ylabel('WRMS')
    ax_io.set_xlabel('WRMS')
    
    return ax_iohh, ax_hh, ax_io

def plot_wrms_beta_1evt(ax, obsdata, evid,
                 minbeta, maxbeta, pasbeta, color, ls, gamma=0):
    """
    Plot the WRMS for one earthquake and different coefficient beta values using
    the Koveslighety equation. Epicentral intensity, depth and the gamma attenuation
    coefficient are fixed.

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    obsdata : pandas.DataFrame
        Dataframe with the observed intensity and their associated epicentral 
        depth for the given earthquake. Mandatory columns are:
            - I: observed intensity
            - Depi: associated epicentral intensity
            - StdI: standard deviation/uncertainty associated to the intensity value
            - Depth: hypocentral depth
            - Io: epicentral intensity
    evid : float, int, str
        ID of the earthquake.
    minbeta : float
        Minimal value of beta coefficient used to compute and plot the WRMS.
    maxbeta : float
        Maximal value of beta coefficient used to compute and plot the WRMS.
    pasbeta : float
        Step value between two beta values.
    color : str
        Color used in the plot (see matplotlib documentation).
    ls : str
        Linestyle used in the plot (see matplotlib documentation).

    gamma : float, optional
        Intresic attenuation coefficient. The default is 0.

    Returns
    -------
    None.

    """
    
    obsdata = obsdata[obsdata.EVID==evid]
    wrms = np.array([])
    beta_range = np.arange(minbeta, maxbeta+pasbeta, pasbeta)
    for beta in beta_range:
        wrms_i, obsdata = calcul_wrms_beta(obsdata, beta, gamma)
        wrms = np.append(wrms, wrms_i)

    ax.plot(beta_range, wrms, ls=ls, color=color,
            label=str(evid))
    ax.set_xlabel('BETA')
    ax.set_ylabel('WRMS')
