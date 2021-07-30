# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:28:50 2021

@author: PROVOST-LUD
"""

import pandas as pd
import numpy as np
import WLSIC
from wrms import plot_wrms_withHI0lines, plot_wrms_beta, plot_wrms_beta_1evt
from wrms import getHline_in_HI0wrms_space, getI0line_in_HI0wrms_space
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

# Mettre une option "enregistrer le chemin d'inversion"



def critere_stop(suivi_beta, suivi_depth, suivi_I0, NminIter,
                 stop_beta=0.001, stop_depth=0.05, stop_I0=0.01):
    """
    Function that check if the stop criteria of the iterative inversion are reached.
    The inversion stop if the three last inverted parameters is stable: the mean
    difference of the three last values should be under a given value.
    
    :param suivi_beta: Contains the inverted geometric attenuation coefficient beta from the
                       first iteration to the current iteration
                       
    :param suivi_depth: Contains for each earthquake the inverted depth values from the
                       first iteration to the current iteration
    :param suivi_I0: Contains for each earthquake the inverted epicentral intensity
                     values from the first iteration to the current iteration
    :param NminIter: minimal number of iterations before stop
    :param stop_beta: stop value for the geometric attenuation coefficient beta
    :param stop_depth: stop value for the depth. The mean difference of the three
                       last values is meaned over the earthquakes
    :param stop_I0: stop value for the epicentral intensity. The mean difference
                    of the three last values is meaned over the earthquakes
    :type suivi_beta: list
    :type suivi_depth: dict
    :type suivi_I0: dict
    :type NminIter: int
    :type stop_beta: float
    :type stop_depth: float
    :type stop_I0: float
    """
    # pas testee
    ConvrateBeta = sum(abs(np.diff(suivi_beta[-3:])))/NminIter
    ConvrateDepth = 0
    ConvrateI0 = 0
    for evid in suivi_depth.keys():
        last3_depth = suivi_depth[evid][-3:]
        last3_I0 = suivi_I0[evid][-3:]
        ConvrateDepth = ConvrateDepth + sum(abs(np.diff(last3_depth)))/NminIter
        ConvrateI0 = ConvrateI0 + sum(abs(np.diff(last3_I0)))/NminIter
    ConvrateDepth = ConvrateDepth/len(suivi_depth.keys())
    ConvrateI0 = ConvrateI0/len(suivi_depth.keys())
    condition = (ConvrateBeta<=stop_beta) and (ConvrateDepth<=stop_depth)and(ConvrateI0<=stop_I0)
    return condition

def define_ls_color_byevt(count, cmap='tab20b', len_cmap=20):
    """
    Function that attribute a different color and line style to a number.
    The number should be lower than 99.
    
    :param count: the number for which a color and a line style is needed.
    :param cmap: colormap choosed to attribute the colors (see matplotlib colormaps)
    :param len_cmap: number of color considered
    :type count: int
    :type cmap: str
    :type len_cmap: int
    
    :return: a line style and a color
    """
    cmap = cm.get_cmap(cmap, len_cmap)
    ls_list = ['-', ':', '-+', '-s', '-o']
    if count < 20:
        ls = ls_list[0]
        color = cmap(count)
    elif count < 40:
        ls = ls_list[1]
        color = cmap(count-20)
    elif count < 60:
        ls = ls_list[2]
        color = cmap(count-40)
    elif count < 80:
        ls = ls_list[3]
        color = cmap(count-60)
    elif count < 100:
        ls = ls_list[4]
        color = cmap(count-80)
    else:
        print("Too much event to follow each inversion, stop to the 99th event")
        return '-', 'Gray'
    return ls, color

def define_ls_color_bydepth(depth, cmap='viridis_r', vmin=2, vmax=20):
    """
    Function that associates a color to a number.
    
    :param depth: the number for which a color is needed.
    :param cmap: colormap choosed to attribute the colors (see matplotlib colormaps)
    :param vmin: minimum value to define the color map
    :param vmax: maximal value to define the color map
    :type depth: int
    :type cmap: str
    :type vmin: float
    :type vmax: float
    
    :return: a line style, the color associated to the input number, the chosen
             colormap object, the norm object associated to the colormap and
             defined by vmin and vmax.
    """
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap)
    color = cmap(norm(depth))
    ls='-'
    return ls, color, cmap, norm
    

def calib_attBeta_Kov(liste_evt, ObsBin_plus, beta_ini, 
                      NminIter=3, NmaxIter=50, suivi_inversion=False,
                      dossier_suivi=''):
    """
    Function that inverse iteratively and sequentially depth, epicentral intensity
    and the geometric attenuation coefficient beta for a given earthquake list and 
    binned intensity.
    
    :param liste_evt: Contains the ID of the earthquakes used for the inversion
    :param ObsBin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 
                            'I' the binned intensity value,
                            'StdI' the binned intensity standard deviation,
                            'Depi' the associated epicentral distance,
                            'Depth' the initial depth before inversion,
                            'Hmin' the lower depth limit for inversion,
                            'Hmax' the upper depth limit for inversion,
                            'Io' the initial epicentral intensity before inversion,
                            'Io_ini' the initial epicentral intensity before inversion,
                            'Io_std' the epicentral intensity standard deviation,
                            'eqStd' the inverse of the root square of the weights
                                    used in the geometric attenuation coefficient,
                            
    :param NminIter: minimum number of iteration
    :param NmaxIter: maximal number of iteration
    :param suivi_inversion: option to visualize the WRMS of depth, epicentral
                            intensity and geometric attenuation coefficient beta
                            at different iterations. Time consuming option.
    :param dossier_suivi: folder in which the figures with the WRMS will be stored.
    :type liste_evt: list
    :type ObsBin_plus: pandas.DataFrame
    :type NminIter: int
    :type NmaxIter: int
    :type suivi_inversion: bool
    :type dossier_suivi: str
        
    :return: the ObsBin_plus dataframe with depth and epicentral intensity after inversion
             the output eometric attenuation coefficient beta, the covariance
             of the geometric attenuation coefficient, a list with the beta values
             for each iteration.
    """
    
    # Yellow: stade inverse
    # White: stade ini
    
        
    iteration = 0
    beta = beta_ini
    suivi_depth = {}
    suivi_I0 = {}
    suivi_beta = np.array([beta])
    for evid in liste_evt:
        obsbin = ObsBin_plus[ObsBin_plus.EVID==evid]
        depth = obsbin.Depth.values[0]
        I0 = obsbin.Io.values[0]
        suivi_depth[evid] = np.array([depth])
        suivi_I0[evid] = np.array([I0])
    ObsBin_plus.loc[:, 'Hmin_ini'] =  ObsBin_plus.loc[:, 'Hmin']
    ObsBin_plus.loc[:, 'Hmax_ini'] =  ObsBin_plus.loc[:, 'Hmax']
    
    while iteration < NmaxIter:
        nomfig_base = 'Iteration' + "{:02d}".format(iteration) + '_'
        if suivi_inversion and iteration in [1, 3, 10]:
            fig_beta0 = plt.figure(figsize=(6, 6))
            
            ax_beta0 = fig_beta0.add_subplot(111)
            ax_beta0.set_facecolor("Gray")
            plot_wrms_beta(ax_beta0, ObsBin_plus, liste_evt,
                           -2, -5, -0.1, color='w')
            ax_beta0.axvline(x=beta, color='w', ls='--',
                             label='Initial state')
            
        ObsBin_plus, beta = calib_attBeta_Kov_unit(liste_evt, ObsBin_plus, beta)
        #print(beta)
        suivi_beta = np.append(suivi_beta, beta)
        if suivi_inversion and iteration in [1, 3, 10]:
            nomfig_base = 'Iteration' + "{:02d}".format(iteration) + '_'
            plot_wrms_beta(ax_beta0, ObsBin_plus, liste_evt,
                           -2, -5, -0.1, color='Yellow')
            ax_beta0.axvline(x=beta, color='Yellow', ls='--',
                             label='Post inversion state')
            ax_beta0.legend()
            
            fig_beta_byevt = plt.figure(figsize=(6, 6))
            ax_betabyevt = fig_beta_byevt.add_subplot(111)
            #cax = fig_beta_byevt.add_axes([0.27, 0.8, 0.5, 0.05])
            ax_betabyevt.set_facecolor("Gray")
            plot_wrms_beta(ax_betabyevt, ObsBin_plus, liste_evt,
                           -2, -5, -0.1, color='r')
            for count, evid in enumerate(liste_evt):
                ls, color = define_ls_color_byevt(count)
                depth = ObsBin_plus[ObsBin_plus.EVID==evid]['Depth'].values[0]
                #ls, color, cmap, norm = define_ls_color_bydepth(depth)
                
                plot_wrms_beta_1evt(ax_betabyevt, ObsBin_plus, evid,
                           -2, -5, -0.1, color=color, ls=ls)
#            cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
#                                norm=norm,
#                                orientation='horizontal')
#            cb1.set_label('Depth [km]')
            ax_betabyevt.axvline(x=beta, color='r', ls='--',
                             label='Post inversion state')
            ax_betabyevt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                                borderaxespad=0., facecolor='Gray')
            fig_beta_byevt.savefig(dossier_suivi + '/' + nomfig_base + 'beta_byevt.png',
                               bbox_inches='tight')
            
            fig_beta0.savefig(dossier_suivi + '/' + nomfig_base + 'beta.png')
            plt.close(fig_beta_byevt)
            plt.close(fig_beta0)
        
        minH = 1
        maxH = 25
        pasH = 0.25
        minI0 = 2
        maxI0 = 10
        pasI0 = 0.1
        for evid in liste_evt:
            obsbin = ObsBin_plus[ObsBin_plus.EVID==evid]
            depth = obsbin.Depth.values[0]
            I0 = obsbin.Io.values[0]
            suivi_depth[evid] = np.append(suivi_depth[evid], depth)
            suivi_I0[evid] = np.append(suivi_I0[evid], I0)
            if (evid in [6, 13, 17, 30]) and suivi_inversion and iteration in [1, 3, 10]:
                obsdata_tmp = ObsBin_plus[ObsBin_plus.EVID==evid]
                obsdata_tmpavt = copy.deepcopy(obsdata_tmp)
                obsdata_tmpavt.loc[:, 'Depth'] = suivi_depth[evid][-2]
                obsdata_tmpavt.loc[:, 'Io'] = suivi_I0[evid][-2]
                line_wrms_hinv = getHline_in_HI0wrms_space(ObsBin_plus, evid, suivi_beta[-2],
                                                           minH, maxH, pasH,
                                                           minI0, maxI0, pasI0)
                line_wrms_ioinv = getI0line_in_HI0wrms_space(ObsBin_plus, evid, suivi_beta[-2],
                                                             minH, maxH, pasH,
                                                             minI0, maxI0, pasI0)
                fig_wrms = plt.figure(figsize=(6, 6))
                ax_iohh, ax_hh, ax_io = plot_wrms_withHI0lines(fig_wrms,
                                                               obsdata_tmpavt, evid,
                                                               beta, vmax=2)
                ax_iohh.axvline(x=suivi_depth[evid][-1], color='Yellow')
                ax_iohh.axhline(y=suivi_I0[evid][-1], color='Yellow')
                ax_io.plot(line_wrms_hinv,
                           np.arange(minI0, maxI0+pasI0, pasI0), color='Yellow')
                ax_hh.plot(np.arange(minH, maxH+pasH, pasH),
                           line_wrms_ioinv, color='Yellow')
                nomfig_base = 'Iteration' + "{:02d}".format(iteration) + '_evt' + "{:02d}".format(evid)
                fig_wrms.savefig(dossier_suivi + '/' + nomfig_base + 'HI0_sat2wrms.png')
                #fig_wrms.clf()
                plt.close(fig_wrms)
            
        iteration += 1
        if iteration > NminIter:
            if critere_stop(suivi_beta, suivi_depth, suivi_I0, NminIter,
                            stop_beta=0.001, stop_depth=0.05, stop_I0=0.01):
                resBetaStd = WLSIC.WLS_Kov(ObsBin_plus, beta, 0).do_wls_beta_std()
                cov_beta = resBetaStd[1]
                break
    if iteration>=NmaxIter:
        resBetaStd = WLSIC.WLS_Kov(ObsBin_plus, beta, 0).do_wls_beta_std()
        cov_beta = resBetaStd[1]
    return ObsBin_plus, beta, cov_beta, suivi_beta
        

def calib_attBeta_Kov_unit(liste_evt, ObsBin_plus, beta):
    """
    Function that inverse sequentially depth, epicentral intensity and the
    geometric attenuation coefficient beta for a given earthquake list and 
    binned intensity and one iteration.
    
    :param liste_evt: Contains the ID of the earthquakes used for the inversion
    :param ObsBin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 
                            'I' the binned intensity value,
                            'StdI' the binned intensity standard deviation,
                            'Depi' the associated epicentral distance,
                            'Depth' the initial depth before inversion,
                            'Hmin' the lower depth limit for inversion,
                            'Hmax' the upper depth limit for inversion,
                            'Io' the initial epicentral intensity before inversion,
                            'Io_ini' the initial epicentral intensity before inversion,
                            'Io_std' the epicentral intensity standard deviation,
                            'eqStd' the inverse of the root square of the weights
                                    used in the geometric attenuation coefficient,
                            
    :param beta: initial value of the geometric attenuation coefficient beta
    :type liste_evt: list
    :type ObsBin_plus: pandas.DataFrame
    :type beta: float
        
    :return: the ObsBin_plus dataframe with depth and epicentral intensity after inversion
             and the inverted geometric attenuation coefficient beta
    """
    for evid in liste_evt:
        obsbin = ObsBin_plus[ObsBin_plus.EVID==evid]
        depth = obsbin.Depth.values[0]
        Hmin = obsbin.Hmin.values[0]
        Hmax = obsbin.Hmax.values[0]
        obsbin.loc[:, 'Hypo'] = np.sqrt(obsbin['Depi'].values**2 + depth**2)
        I0 = obsbin.Io.values[0]
        I0_min = obsbin.Io_ini.values[0] - 2*obsbin.Io_std.values[0]
        I0_max = obsbin.Io_ini.values[0] + 2*obsbin.Io_std.values[0]
        mini_iteration = 1
        while mini_iteration<2:
            # Inversion de la profondeur
            resH = WLSIC.WLSIC_Kov_oneEvt(obsbin, depth, beta, 0, I0).do_wlsic_depth(Hmin, Hmax)
            new_depth = resH[0][0]
            ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Depth'] = new_depth
            # Inversion de l'intensite epicentrale
            resI0 = WLSIC.WLSIC_Kov_oneEvt(obsbin, new_depth, beta, 0, I0).do_wlsic_I0(I0_min, I0_max)
            I0_inv = resI0[0][0]
            I0 = I0_inv
            mini_iteration += 1
        ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Io'] = I0_inv
    resBeta = WLSIC.WLS_Kov(ObsBin_plus, beta, 0).do_wls_beta()
    beta = resBeta[0][0]
    
    return ObsBin_plus, beta

def calib_attBetaGamma_Kov():
    pass

def calib_attBetaGammaReg_Kov():
    pass