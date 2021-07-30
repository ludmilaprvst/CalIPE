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


def calcul_wrms_evt(obsbin, Beta, I0, depth):
    # pas testee
    Hypo = np.sqrt(obsbin.Depi.values**2 + depth**2)
    Ipred = I0 + Beta * np.log10(Hypo/depth)
    Cd = obsbin['StdI'].values**2
    Wd = 1./Cd
    Wd = Wd/sum(Wd)
    dI = obsbin.I.values - Ipred
    WRMS = np.sum((dI**2)*Wd)/np.sum(Wd)
    return WRMS

def calcul_wrms_beta(obsdata, Beta):
    obsdata.loc[:, 'Hypo'] = np.sqrt(obsdata.loc[:, 'Depi']**2 + obsdata.loc[:, 'Depth']**2)
    obsdata.loc[:, 'logterm'] = np.log10(obsdata.loc[:, 'Hypo']/obsdata.loc[:, 'Depth'])
    obsdata.loc[:, 'Ipred'] = obsdata.loc[:, 'Io'] + Beta* obsdata.loc[:, 'logterm']
    obsdata.loc[:, 'dI'] = np.abs(obsdata.loc[:, 'I'] - obsdata.loc[:, 'Ipred'])
    obsdata.loc[:, 'Wd'] = 1/obsdata['StdI'].values**2
    obsdata.loc[:, 'Wd'] = obsdata.loc[:, 'Wd']/np.sum(obsdata.loc[:, 'Wd'])
    WRMS = np.sum((obsdata.loc[:, 'dI']**2)*obsdata.loc[:, 'Wd'])/np.sum(obsdata.loc[:, 'Wd'])
    return WRMS, obsdata

def get_HI0_wrms_space(obsdata, evid, Beta, minH=1, maxH=25, pasH=0.25,
                       minI0=2, maxI0=10,  pasI0=0.1):
    obsbin = obsdata[obsdata.EVID==evid]
    wrms = np.array([])
    io_plot = np.array([])
    hh_plot = np.array([])
#    print(np.arange(minH, maxH+pasH, pasH))
#    print(minH, maxH+pasH, pasH)
#    print(np.arange(minI0, maxI0+pasI0, pasI0))
    for hh in np.arange(minH, maxH+pasH, pasH):
        for io in np.arange(minI0, maxI0+pasI0, pasI0):
            wrms_i = calcul_wrms_evt(obsbin, Beta, io, hh)
            wrms = np.append(wrms, wrms_i)
            io_plot = np.append(io_plot, io)
            hh_plot = np.append(hh_plot, hh)
        
    return hh_plot, io_plot, wrms

def getHline_in_HI0wrms_space(obsdata, evid, Beta, minH=1, maxH=25, pasH=0.25,
                              minI0=2, maxI0=10, pasI0=0.1):
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
    obsbin = obsdata[obsdata.EVID==evid]
    I0 = obsbin.Io.values[0]
     # Io line
    line_wrms_ioinv = np.array([])
    for hh in np.arange(minH, maxH+pasH, pasH):
        wrms_i = calcul_wrms_evt(obsbin, Beta, I0, hh)
        line_wrms_ioinv = np.append(line_wrms_ioinv, wrms_i)
    return line_wrms_ioinv

def plot_HI0_wrms_space(ax, hh_plot, io_plot, wrms, minH, maxH, minI0, maxI0,
                        vmin=0, vmax=-99):
    if vmax==-99:
        vmax = np.max(wrms)
    xi, yi = np.mgrid[minH:maxH:100j, minI0:maxI0:100j]
    points_hhio = []
    for xx in range(len(hh_plot)):
        points_hhio.append([hh_plot[xx], io_plot[xx]])
    points_hhio = np.array(points_hhio)
    zi = griddata(points_hhio, wrms, (xi, yi), method='linear')
    
    ax.imshow(zi.T, origin='lower', vmin=0, vmax=vmax,
              extent=[minH, maxH, minI0, maxI0], aspect='auto',
              interpolation='nearest', cmap=plt.cm.get_cmap('terrain'))


def classic_config_wmrsevt(fig_wrms, minI0, maxI0, minH, maxH):
    
    gs0 = gridspec.GridSpec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7))
    
    ax_iohh = fig_wrms.add_subplot(gs0[1, 0])
    ax_iohh.set_ylim([minI0, maxI0])
    ax_iohh.set_xlim([minH, maxH])
    ax_io = fig_wrms.add_subplot(gs0[1, 1], sharey=ax_iohh)
    ax_hh = fig_wrms.add_subplot(gs0[0, 0], sharex=ax_iohh)
    return ax_iohh, ax_hh, ax_io

def plot_wrms_withHI0lines(fig_wrms,
                           obsdata, evid, Beta,
                           minH=1, maxH=25, pasH=0.25,
                           minI0=2, maxI0=10, pasI0=0.1,
                           vmax=-99):
    
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
    
    
    ax_iohh, ax_hh, ax_io = classic_config_wmrsevt(fig_wrms, minI0, maxI0, minH, maxH)
    
    plot_HI0_wrms_space(ax_iohh, hh_plot, io_plot, wrms, minH, maxH, minI0, maxI0,
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
                 minbeta, maxbeta, pasbeta, color, ls):
    obsdata = obsdata[obsdata.EVID==evid]
    wrms = np.array([])
    beta_range = np.arange(minbeta, maxbeta+pasbeta, pasbeta)
    for beta in beta_range:
        wrms_i, obsdata = calcul_wrms_beta(obsdata, beta)
        wrms = np.append(wrms, wrms_i)

    ax.plot(np.arange(-2, -5.1, -0.1), wrms, ls=ls, color=color,
            label=str(evid))
    ax.set_xlabel('BETA')
    ax.set_ylabel('WRMS')

def plot_wrms_beta(ax, obsdata, liste_evt,
                 minbeta, maxbeta, pasbeta, color):
    wrms = np.array([])
    beta_range = np.arange(minbeta, maxbeta+pasbeta, pasbeta)
    for beta in beta_range:
        wrms_i, obsdata = calcul_wrms_beta(obsdata, beta)
        wrms = np.append(wrms, wrms_i)

    ax.plot(np.arange(-2, -5.1, -0.1), wrms, lw=2, color=color)
    ax.set_xlabel('BETA')
    ax.set_ylabel('WRMS')
    
    
    