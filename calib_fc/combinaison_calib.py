# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:28:50 2021

@author: PROVOST-LUD
"""

import pandas as pd
import numpy as np
import WLSIC
import WLSIC_2
import statsmodels.formula.api as sm
#import sys
#sys.path.append('../postprocessing_fc')
#from wrms import plot_wrms_withHI0lines, plot_wrms_beta, plot_wrms_beta_1evt
#from wrms import getHline_in_HI0wrms_space, getI0line_in_HI0wrms_space
#from wrms import calcul_wrms_C1C2betagamma
from prepa_data import update_XCaCb, add_I0as_datapoint
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

# Mettre une option "enregistrer le chemin d'inversion"
pd.options.mode.chained_assignment = None 


def critere_stop(suivi_beta, suivi_depth, suivi_I0,
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
    :param stop_beta: stop value for the geometric attenuation coefficient beta
    :param stop_depth: stop value for the depth. The mean difference of the three
                       last values is meaned over the earthquakes
    :param stop_I0: stop value for the epicentral intensity. The mean difference
                    of the three last values is meaned over the earthquakes
    :type suivi_beta: list
    :type suivi_depth: dict
    :type suivi_I0: dict
    :type stop_beta: float
    :type stop_depth: float
    :type stop_I0: float
    """
    # pas testee
    NminIter = 3
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

def critere_stop2(suivi_beta, suivi_gamma, suivi_depth, suivi_I0,
                 stop_beta=0.001, stop_gamma=0.00001, stop_depth=0.05, stop_I0=0.01):
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
    #NminIter = 3
    ConvrateBeta = sum(abs(np.diff(suivi_beta[-3:])))/2
    ConvrateGamma = sum(abs(np.diff(suivi_gamma[-3:])))/2
    ConvrateDepth = 0
    ConvrateI0 = 0
    for evid in suivi_depth.keys():
        last3_depth = suivi_depth[evid][-3:]
        last3_I0 = suivi_I0[evid][-3:]
        ConvrateDepth = ConvrateDepth + sum(abs(np.diff(last3_depth)))/2
        ConvrateI0 = ConvrateI0 + sum(abs(np.diff(last3_I0)))/2
    ConvrateDepth = ConvrateDepth/len(suivi_depth.keys())
    ConvrateI0 = ConvrateI0/len(suivi_depth.keys())
    condition = (ConvrateBeta<=stop_beta) and (ConvrateGamma<=stop_gamma) and (ConvrateDepth<=stop_depth)and(ConvrateI0<=stop_I0)
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
                      NminIter=3, NmaxIter=50, stop_beta=0.001, stop_depth=0.05,
                      stop_I0=0.01,
                      suivi_inversion=False,
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
    :param beta_ini: initial value of the geometric attenuation coefficient beta                        
    :param NminIter: minimum number of iteration
    :param NmaxIter: maximal number of iteration
    :param suivi_inversion: option to visualize the WRMS of depth, epicentral
                            intensity and geometric attenuation coefficient beta
                            at different iterations. Time consuming option.
    :param dossier_suivi: folder in which the figures with the WRMS will be stored.
    :type liste_evt: list
    :type ObsBin_plus: pandas.DataFrame
    :type beta_ini: float
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
        ObsBin_plus, beta = calib_attBeta_Kov_unit(liste_evt, ObsBin_plus, beta)
        suivi_beta = np.append(suivi_beta, beta)

        for evid in liste_evt:
            obsbin = ObsBin_plus[ObsBin_plus.EVID==evid]
            depth = obsbin.Depth.values[0]
            I0 = obsbin.Io.values[0]
            suivi_depth[evid] = np.append(suivi_depth[evid], depth)
            suivi_I0[evid] = np.append(suivi_I0[evid], I0)
       

        iteration += 1
        if iteration > NminIter:
            if critere_stop(suivi_beta, suivi_depth, suivi_I0, 
                            stop_beta, stop_depth, stop_I0):
                print('iteration:')
                print(iteration)
                resBetaStd = WLSIC.WLS_Kov(ObsBin_plus, beta, 0).do_wls_beta_std()
                cov_beta = resBetaStd[1]
                break
    if iteration>=NmaxIter:
        resBetaStd = WLSIC.WLS_Kov(ObsBin_plus, beta, 0).do_wls_beta_std()
        cov_beta = resBetaStd[1]
    return ObsBin_plus, beta, cov_beta, suivi_beta
        

def calib_attBeta_Kov_unit(liste_evt, ObsBin_plus, beta,
                           ):
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
        # obsbin = obsbin.astype({'EVID':'str', 'I':'float64', 'StdI': 'float64',
        #                         'Io':'float64', 'Io_std': 'float64', 'Io_ini': 'float64',
        #                         'Ndata':'int', 'Mag':'float64', 'StdM':'float64',
        #                         'Depth':'float64', 'Hmin': 'float64', 'Hmax':'float64',
        #                         'RegID':'str', 'eqStd':'float64', 'Hmin_ini':'float64',
        #                         'Hmax_ini':'float64'})
        depth = copy.deepcopy(obsbin.Depth.values[0])
        Hmin = obsbin.Hmin.values[0]
        Hmax = obsbin.Hmax.values[0]
        epi = copy.deepcopy(obsbin['Depi'].values.astype(float))
        #obsbin.loc[:, 'Hypo'] = np.sqrt(obsbin['Depi'].values.astype(float)**2 + depth**2)
        obsbin.loc[:, 'Hypo'] = np.sqrt(epi**2 + depth**2)
        #obsbin['Hypo'] = np.sqrt(obsbin['Depi'].astype(float)**2 + depth**2)
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

def calib_attBetaGamma_Kov(liste_evt, ObsBin_plus, beta_ini, gamma_ini,
                           NminIter=3, NmaxIter=50,
                           stop_beta=0.001, stop_gamma=0.00001, stop_depth=0.05,
                           stop_I0=0.01):
    """
    Function that inverse iteratively and sequentially depth, epicentral intensity
    and the attenuation coefficients beta and gamma for a given earthquake list and 
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
                            
    :param beta_ini: initial value of the geometric attenuation coefficient beta
    :param gamma_ini: initial value of the intrinsic attenuation coefficient gamma
    :param NminIter: minimum number of iteration
    :param NmaxIter: maximal number of iteration
    :param dossier_suivi: folder in which the figures with the WRMS will be stored.
    :type liste_evt: list
    :type ObsBin_plus: pandas.DataFrame
    :type beta_ini: float
    :type gamma_ini: float
    :type NminIter: int
    :type NmaxIter: int
    :type suivi_inversion: bool
    :type dossier_suivi: str
        
    :return: the ObsBin_plus dataframe with depth and epicentral intensity after inversion
             the output eometric attenuation coefficient beta, the covariance
             of the geometric attenuation coefficient, a list with the beta values
             for each iteration.
    """
    iteration = 0
    beta = beta_ini
    gamma = gamma_ini
    suivi_depth = {}
    suivi_I0 = {}
    suivi_beta = np.array([beta])
    suivi_gamma = np.array([gamma])
    for evid in liste_evt:
        obsbin = ObsBin_plus[ObsBin_plus.EVID==evid]
        depth = obsbin.Depth.values[0]
        I0 = obsbin.Io.values[0]
        suivi_depth[evid] = np.array([depth])
        suivi_I0[evid] = np.array([I0])
    ObsBin_plus.loc[:, 'Hmin_ini'] =  ObsBin_plus.loc[:, 'Hmin']
    ObsBin_plus.loc[:, 'Hmax_ini'] =  ObsBin_plus.loc[:, 'Hmax']
    
    while iteration < NmaxIter:
         
        ObsBin_plus, beta, gamma = calib_attBetaGamma_Kov_unit(liste_evt, ObsBin_plus, beta, gamma)
        #print(beta)
        suivi_beta = np.append(suivi_beta, beta)
        suivi_gamma = np.append(suivi_gamma, gamma)
        for evid in liste_evt:
            obsbin = ObsBin_plus[ObsBin_plus.EVID==evid]
            depth = obsbin.Depth.values[0]
            I0 = obsbin.Io.values[0]
            suivi_depth[evid] = np.append(suivi_depth[evid], depth)
            suivi_I0[evid] = np.append(suivi_I0[evid], I0)
        iteration += 1
        if iteration > NminIter:
            if critere_stop2(suivi_beta, suivi_gamma, suivi_depth, suivi_I0,
                            stop_beta=stop_beta, stop_gamma=stop_gamma,
                            stop_depth=stop_depth, stop_I0=stop_I0):

                break

    resBetaGammaStd = WLSIC.WLS_Kov(ObsBin_plus, beta, gamma).do_wls_betagamma_std()
    cov_betagamma = resBetaGammaStd[1]
    return ObsBin_plus, beta, gamma, cov_betagamma, suivi_beta, suivi_gamma

def calib_attBetaGamma_Kov_unit(liste_evt, ObsBin_plus, beta, gamma):
    """
    Function that inverse sequentially depth, epicentral intensity and the
    attenuation coefficients beta and gamma for a given earthquake list and 
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
    :param gamma: initial value of the intrinsic attenuation coefficient gamma
    :type liste_evt: list
    :type ObsBin_plus: pandas.DataFrame
    :type beta: float
    :type gamma: float
        
    :return: the ObsBin_plus dataframe with depth and epicentral intensity after inversion
             and the inverted attenuation coefficients beta and gamma
    """
    for evid in liste_evt:
        obsbin = ObsBin_plus[ObsBin_plus.EVID==evid]
        depth = obsbin.Depth.values[0]
        Hmin = obsbin.Hmin.values[0]
        Hmax = obsbin.Hmax.values[0]
        obsbin.loc[:, 'Hypo'] = np.sqrt(obsbin['Depi'].values.astype(float)**2 + depth**2)
        I0 = obsbin.Io.values[0]
        I0_min = obsbin.Io_ini.values[0] - 2*obsbin.Io_std.values[0]
        I0_max = obsbin.Io_ini.values[0] + 2*obsbin.Io_std.values[0]
        mini_iteration = 1
        while mini_iteration<2:
            # Inversion de la profondeur
            resH = WLSIC.WLSIC_Kov_oneEvt(obsbin, depth, beta, gamma, I0).do_wlsic_depth(Hmin, Hmax)
            new_depth = resH[0][0]
            ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Depth'] = new_depth
            # Inversion de l'intensite epicentrale
            resI0 = WLSIC.WLSIC_Kov_oneEvt(obsbin, new_depth, beta, gamma, I0).do_wlsic_I0(I0_min, I0_max)
            I0_inv = resI0[0][0]
            I0 = I0_inv
            mini_iteration += 1
            ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Io'] = I0_inv
            
    resBetaGamma = WLSIC.WLS_Kov(ObsBin_plus, beta, gamma).do_wls_betagamma()
    beta = resBetaGamma[0][0]
    gamma = resBetaGamma[0][1]
    return ObsBin_plus, beta, gamma

def calib_attBetaGammaReg_Kov():
    pass

def critere_arret_HI0(suivi_depth, suivi_I0,
                      stop_depth=0.05, stop_I0=0.01):
    ConvrateDepth = 0
    ConvrateI0 = 0
    NminIter=3
    for evid in suivi_depth.keys():
        last3_depth = suivi_depth[evid][-3:]
        last3_I0 = suivi_I0[evid][-3:]
        ConvrateDepth = ConvrateDepth + sum(abs(np.diff(last3_depth)))/NminIter
        ConvrateI0 = ConvrateI0 + sum(abs(np.diff(last3_I0)))/NminIter
    ConvrateDepth = ConvrateDepth/len(suivi_depth.keys())
    ConvrateI0 = ConvrateI0/len(suivi_depth.keys())
    condition = (ConvrateDepth<=stop_depth)and(ConvrateI0<=stop_I0)
    return condition

def critere_arret_HI0C1C2(suivi_depth, suivi_I0, suivi_C1, suivi_C2,
                         stop_depth=0.05, stop_I0=0.01, stop_C1=0.001, stop_C2=0.001):
    ConvrateDepth = 0
    ConvrateI0 = 0
    NminIter=3
    ConvrateC1 = sum(abs(np.diff(suivi_C1[-3:])))/NminIter
    ConvrateC2 = sum(abs(np.diff(suivi_C2[-3:])))/NminIter
    for evid in suivi_depth.keys():
        last3_depth = suivi_depth[evid][-3:]
        last3_I0 = suivi_I0[evid][-3:]
        ConvrateDepth = ConvrateDepth + sum(abs(np.diff(last3_depth)))/NminIter
        ConvrateI0 = ConvrateI0 + sum(abs(np.diff(last3_I0)))/NminIter
    ConvrateDepth = ConvrateDepth/len(suivi_depth.keys())
    ConvrateI0 = ConvrateI0/len(suivi_depth.keys())
    condition = (ConvrateDepth<=stop_depth)and(ConvrateI0<=stop_I0)and(ConvrateC1<=stop_C1)and(ConvrateC2<=stop_C2)
    return condition


def initialize_HI0(ObsBin_plus, liste_evt, beta, gamma, NmaxIter=50):
    # a tester
    suivi_depth = {}
    suivi_I0 = {}
    for evid in liste_evt:
        obsbin = ObsBin_plus[ObsBin_plus.EVID==evid].copy()
        depth = obsbin.Depth.values[0]
        Hmin = obsbin.Hmin.values[0]
        Hmax = obsbin.Hmax.values[0]
        obsbin.loc[:, 'Hypo'] = np.sqrt(obsbin['Depi'].values.astype(float)**2 + depth**2)
        I0 = obsbin.Io.values[0]
        I0_min = obsbin.Io_ini.values[0] - 2*obsbin.Io_std.values[0]
        I0_max = obsbin.Io_ini.values[0] + 2*obsbin.Io_std.values[0]
        suivi_depth[evid] = np.array([depth])
        suivi_I0[evid] = np.array([I0])
        mini_iteration = 0
        while mini_iteration<NmaxIter:
            # Inversion de la profondeur
            resH = WLSIC.WLSIC_Kov_oneEvt(obsbin, depth, beta, gamma, I0).do_wlsic_depth(Hmin, Hmax)
            new_depth = resH[0][0]
            ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Depth'] = new_depth
            # Inversion de l'intensite epicentrale
            resI0 = WLSIC.WLSIC_Kov_oneEvt(obsbin, new_depth, beta, gamma, I0).do_wlsic_I0(I0_min, I0_max)
            I0_inv = resI0[0][0]
            StdI0_inv = np.sqrt(np.diag(resI0[1][0]))
            
            I0 = I0_inv
            mini_iteration += 1
            ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Io'] = I0_inv
            ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'StdIo_inv'] = StdI0_inv
            suivi_depth[evid] = np.append(suivi_depth[evid], new_depth)
            suivi_I0[evid] = np.append(suivi_I0[evid], I0)
            if mini_iteration > 3:
                if critere_arret_HI0(suivi_depth, suivi_I0):
                    break
    return ObsBin_plus

def initialize_C1C2(ObsBin_plus):
    ObsBin_plus.loc[:, 'intercept'] = 1
    obsgp = ObsBin_plus.groupby('EVID').mean()
    #print(obsgp)
    resultCaCb = sm.WLS(obsgp.Mag, obsgp[['intercept', 'X']],
                        weights=1/obsgp.eqStdM.values**2) .fit()
    #print(resultCaCb)
    Ca = resultCaCb.params[1]
    Cb = resultCaCb.params[0]
    C2 = 1/Ca
    C1 = -Cb/Ca
    return C1, C2

def check_ifHlim_ok(obsbin, beta, C1, C2):
    # Verif des Hlim  - compatible avec I0+-2std
    #print(obsbin.columns)
    Ioinf = obsbin.Io_ini.values[0] - 2*obsbin.Io_std.values[0]
    Iosup = obsbin.Io_ini.values[0] + 2*obsbin.Io_std.values[0]
    Hmin_ini = obsbin.Hmin_ini.values[0]
    Hmax_ini = obsbin.Hmax_ini.values[0]
    evid = obsbin.EVID.values[0]
    mag = obsbin.Mag.values[0]
    hmintest = 10**((Iosup-C1-C2*mag)/beta)
    hmaxtest = 10**((Ioinf-C1-C2*mag)/beta)
#    print(beta)
#    print(evid, hmintest, hmaxtest, Hmin_ini, Hmax_ini)
#    print(hmintest > Hmax_ini)
    
    
    if hmaxtest < Hmax_ini:
        Hmax = hmaxtest
    else:
        Hmax = Hmax_ini
    if hmaxtest < Hmin_ini:
        Hmax = Hmin_ini + 1
        
    if hmintest > Hmax_ini:
        Hmin = Hmax_ini - 1
    elif hmintest > Hmin_ini:
        Hmin = hmintest
    else:
        Hmin = Hmin_ini
    Hmin = np.max([0.1, Hmin])
    if Hmin >= Hmax: 
        Hmin = Hmin_ini
    if Hmax <= Hmin: 
        Hmax = Hmax_ini
    return Hmin, Hmax


def calib_C1C2_unit(liste_evt, ObsBin_plus, beta, gamma, C1, C2):
    # Inversion H
    for evid in liste_evt:
        obsbin = ObsBin_plus[ObsBin_plus.EVID==evid]
        depth = obsbin.Depth.values[0]
        mag = obsbin.Mag.values[0]
        I0 = obsbin.Io.values[0]
        I0_min = obsbin.Io_ini.values[0] - 2*obsbin.Io_std.values[0]
        I0_max = obsbin.Io_ini.values[0] + 2*obsbin.Io_std.values[0]
        Hmin, Hmax = check_ifHlim_ok(obsbin, beta, C1, C2)
        ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Hmin'] = Hmin
        ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Hmax'] = Hmax
        if depth > Hmax:
            depth = Hmax
        elif depth < Hmin:
            depth = Hmin
#        print(evid, Hmin, Hmax, depth)
        resH = WLSIC.WLSIC_oneEvt(obsbin, depth, mag, beta, gamma, C1, C2).do_wlsic_depth(Hmin, Hmax)
        new_depth = resH[0][0]
        ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Depth'] = new_depth
        resI0 = WLSIC.WLSIC_Kov_oneEvt(obsbin, new_depth, beta, gamma, I0).do_wlsic_I0(I0_min, I0_max)
        I0_inv = resI0[0][0]
        StdI0_inv = np.sqrt(np.diag(resI0[1][0]))
        ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Io'] = I0_inv
        ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'StdIo_inv'] = StdI0_inv

    # Updating the I0 additional data point    
    ObsBin_plus = ObsBin_plus[ObsBin_plus.Ndata!=0] # On enleve les anciens
    ObsBin_plus = add_I0as_datapoint(ObsBin_plus, liste_evt) # On ajoute les nouveaux
    if 'level_0' in ObsBin_plus.columns:
        ObsBin_plus.drop(['level_0'], axis=1)
    ObsBin_plus = update_XCaCb(ObsBin_plus, beta, gamma)
    # Inversion C1, C2
    [C1, C2] = WLSIC.WLS(ObsBin_plus, C1, C2, beta, gamma).do_wls_C1C2()
    #print(ObsBin_plus.tail())
    return ObsBin_plus, C1, C2

def calib_C1C2(liste_evt, ObsBin_plus, beta, gamma, NmaxIter=50):
    # Attention ObsBin_plus a deja eqStdM
    ObsBin_plus = initialize_HI0(ObsBin_plus, liste_evt, beta, gamma)
    # Ajouter I0
    ObsBin_plus.loc[:, 'Hmin_ini'] =  ObsBin_plus.loc[:, 'Hmin']
    ObsBin_plus.loc[:, 'Hmax_ini'] =  ObsBin_plus.loc[:, 'Hmax']
    ObsBin_plus = add_I0as_datapoint(ObsBin_plus, liste_evt)
    ObsBin_plus = update_XCaCb(ObsBin_plus, beta, gamma)
    
    C1, C2 = initialize_C1C2(ObsBin_plus)
    #print(C1, C2)
    iteration = 0
    suivi_depth = {}
    suivi_I0 = {}
    suivi_C1 = np.array([C1])
    suivi_C2 = np.array([C2])
    for evid in liste_evt:
        obsbin = ObsBin_plus[ObsBin_plus.EVID==evid]
        depth = obsbin.Depth.values[0]
        I0 = obsbin.Io.values[0]
        suivi_depth[evid] = np.array([depth])
        suivi_I0[evid] = np.array([I0])
    while iteration < NmaxIter:
        ObsBin_plus, C1, C2 = calib_C1C2_unit(liste_evt, ObsBin_plus, beta, gamma, C1, C2)
        iteration += 1
        suivi_C1 = np.append(suivi_C1, C1)
        suivi_C2 = np.append(suivi_C2, C2)
        for evid in liste_evt:
            obsbin = ObsBin_plus[ObsBin_plus.EVID==evid]
            depth = obsbin.Depth.values[0]
            I0 = obsbin.Io.values[0]
            suivi_depth[evid] = np.append(suivi_depth[evid], depth)
            suivi_I0[evid] = np.append(suivi_I0[evid], I0)
        if iteration >=3:
            if critere_arret_HI0C1C2(suivi_depth, suivi_I0, suivi_C1, suivi_C2):
                break
    return ObsBin_plus, C1, C2

def critere_arret_HI0C1C2beta(suivi_depth, suivi_I0, suivi_C1, suivi_C2,
                              suivi_beta, stop_beta=0.001, stop_depth=0.05,
                              stop_I0=0.01, stop_C1=0.001, stop_C2=0.001):
    ConvrateDepth = 0
    ConvrateI0 = 0
    NminIter=3
    ConvrateC1 = sum(abs(np.diff(suivi_C1[-3:])))/NminIter
    ConvrateC2 = sum(abs(np.diff(suivi_C2[-3:])))/NminIter
    ConvrateBeta = sum(abs(np.diff(suivi_beta[-3:])))/NminIter
    for evid in suivi_depth.keys():
        last3_depth = suivi_depth[evid][-3:]
        last3_I0 = suivi_I0[evid][-3:]
        ConvrateDepth = ConvrateDepth + sum(abs(np.diff(last3_depth)))/NminIter
        ConvrateI0 = ConvrateI0 + sum(abs(np.diff(last3_I0)))/NminIter
    ConvrateDepth = ConvrateDepth/len(suivi_depth.keys())
    ConvrateI0 = ConvrateI0/len(suivi_depth.keys())
    condition = (ConvrateDepth<=stop_depth)and(ConvrateI0<=stop_I0)and(ConvrateC1<=stop_C1)and(ConvrateC2<=stop_C2)and(ConvrateBeta<=stop_beta)
#    print(ConvrateBeta)
#    print(suivi_beta[-3:])
    return condition

def calib_C1C2beta_unit(liste_evt, ObsBin_plus, C1, C2, beta,
                   NmaxIter=50, add_I0=True,
                   inverse_depth=False, inverse_I0=False):
    for evid in liste_evt:
        if inverse_depth:
            obsbin = ObsBin_plus[ObsBin_plus.EVID==evid]
            depth = obsbin.Depth.values[0]
            mag = obsbin.Mag.values[0]
            I0 = obsbin.Io.values[0]
            I0_min = obsbin.Io_ini.values[0] - 2*obsbin.Io_std.values[0]
            I0_max = obsbin.Io_ini.values[0] + 2*obsbin.Io_std.values[0]
            Hmin, Hmax = check_ifHlim_ok(obsbin, beta, C1, C2)
            ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Hmin'] = Hmin
            ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Hmax'] = Hmax
            if depth > Hmax:
                depth = Hmax
            elif depth < Hmin:
                depth = Hmin
            #print(evid, Hmin, Hmax, depth)
            resH = WLSIC.WLSIC_oneEvt(obsbin, depth, mag, beta, 0, C1, C2).do_wlsic_depth(Hmin, Hmax)
            new_depth = resH[0][0]
            ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Depth'] = new_depth
        if inverse_I0:
            resI0 = WLSIC.WLSIC_Kov_oneEvt(obsbin, new_depth, beta, 0, I0).do_wlsic_I0(I0_min, I0_max)
            I0_inv = resI0[0][0]
            StdI0_inv = np.sqrt(np.diag(resI0[1][0]))
            ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Io'] = I0_inv
            ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'StdIo_inv'] = StdI0_inv
    if inverse_I0 and add_I0:
        # Updating the I0 additional data point    
        ObsBin_plus = ObsBin_plus[ObsBin_plus.Ndata!=0] # On enleve les anciens
        ObsBin_plus = add_I0as_datapoint(ObsBin_plus, liste_evt)
    C1C2Beta = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).do_wls_C1C2Beta()
    C1 = C1C2Beta[0][0]
    C2 = C1C2Beta[0][1]
    beta = C1C2Beta[0][2]
    if beta>-1:
        beta=-1
    return ObsBin_plus, C1, C2, beta

def calib_C1C2beta(liste_evt, ObsBin_plus, C1, C2, beta,
                   NmaxIter=50, add_I0=True,
                   inverse_depth=False, inverse_I0=False):
    ObsBin_plus.loc[:, 'Hmin_ini'] =  ObsBin_plus.loc[:, 'Hmin']
    ObsBin_plus.loc[:, 'Hmax_ini'] =  ObsBin_plus.loc[:, 'Hmax']
    ObsBin_plus.loc[:, 'StdIo_inv'] =  0.2
    if add_I0:
        ObsBin_plus = add_I0as_datapoint(ObsBin_plus, liste_evt)
        ObsBin_plus.sort_values(by=['EVID', 'I'], inplace=True)
    eta_values_tested = np.arange(0, 1.01, 0.01)
    stock_wrms = []
    stock_C1 = []
    stock_C2 = []
    stock_beta = []
    for eta in eta_values_tested:
        sigma = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).compute_2Dsigma(eta)
        C1C2Beta = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).do_wls_C1C2Beta(sigma=sigma)
        C1_inv = C1C2Beta[0][0]
        C2_inv = C1C2Beta[0][1]
        beta_inv = C1C2Beta[0][2]
        wrms = calcul_wrms_C1C2betagamma(ObsBin_plus, C1_inv, C2_inv, beta_inv, 0)
        stock_wrms.append(wrms)
        stock_C1.append(C1_inv)
        stock_C2.append(C2_inv)
        stock_beta.append(beta_inv)
    
    return stock_wrms, stock_C1, stock_C2, stock_beta

def calib_C1C2beta2(liste_evt, ObsBin_plus, C1, C2, beta,
                   NmaxIter=50, add_I0=True,
                   inverse_depth=False, inverse_I0=False):
    ObsBin_plus = initialize_HI0(ObsBin_plus, liste_evt, beta, 0)
    ObsBin_plus.loc[:, 'Hmin_ini'] =  ObsBin_plus.loc[:, 'Hmin']
    ObsBin_plus.loc[:, 'Hmax_ini'] =  ObsBin_plus.loc[:, 'Hmax']

    if add_I0:
        ObsBin_plus = add_I0as_datapoint(ObsBin_plus, liste_evt)
        #ObsBin_plus.sort_values(by=['EVID', 'I'], inplace=True)
    #eta_values_tested = np.arange(0, 1.01, 0.01)
    eta_values_tested = [0]
    stock_wrms = []
    stock_C1 = []
    stock_C2 = []
    stock_beta = []
    for eta in eta_values_tested:
        sigma = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).compute_2Dsigma(eta)
        C1C2Beta = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).do_wls_C1C2Beta(sigma=sigma)
        C1_inv = C1C2Beta[0][0]
        C2_inv = C1C2Beta[0][1]
        beta_inv = C1C2Beta[0][2]
        wrms = calcul_wrms_C1C2betagamma(ObsBin_plus, C1_inv, C2_inv, beta_inv, 0)
        stock_wrms.append(wrms)
        stock_C1.append(C1_inv)
        stock_C2.append(C2_inv)
        stock_beta.append(beta_inv)
    C1 = C1_inv
    C2 = C2_inv
    beta = beta_inv
    
    return ObsBin_plus, stock_wrms, C1, C2, beta


def update_depth(ObsBin_plus, depths, liste_evt):
    for compt, evid in enumerate(liste_evt):
        depth = depths[compt]
        ObsBin_plus.loc[ObsBin_plus.EVID==evid, 'Depth'] = depth
    return ObsBin_plus

def calib_C1C2betaH(liste_evt, ObsBin_plus, C1, C2, beta,
                   NmaxIter=50, add_I0=True,
                   inverse_depth=False, inverse_I0=False):
    ObsBin_plus = initialize_HI0(ObsBin_plus, liste_evt, beta, 0)
    ObsBin_plus.loc[:, 'Hmin_ini'] =  ObsBin_plus.loc[:, 'Hmin']
    ObsBin_plus.loc[:, 'Hmax_ini'] =  ObsBin_plus.loc[:, 'Hmax']

    if add_I0:
        ObsBin_plus = add_I0as_datapoint(ObsBin_plus, liste_evt)
        ObsBin_plus.sort_values(by=['EVID', 'I'], inplace=True)
    #eta_values_tested = np.arange(0, 1.01, 0.01)
    eta_values_tested = [0]
#    stock_wrms = []
#    stock_C1 = []
#    stock_C2 = []
#    stock_beta = []
    for eta in eta_values_tested:
#        sigma = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).compute_2Dsigma(eta, col='StdI')
#        C1C2BetaH = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).do_wls_C1C2BetaH(sigma=sigma)
        sigma = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).compute_2Dsigma(eta)
        C1C2BetaH = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).do_wls_C1C2BetaH(sigma=sigma)
        #sigma = self.Obsbin_plus['StdI'].values
#        C1_inv = C1C2BetaH[0][0]
#        C2_inv = C1C2BetaH[0][1]
#        beta_inv = C1C2BetaH[0][2]
#        wrms = calcul_wrms_C1C2betagamma(ObsBin_plus, C1_inv, C2_inv, beta_inv, 0)
#        stock_wrms.append(wrms)
#        stock_C1.append(C1_inv)
#        stock_C2.append(C2_inv)
#        stock_beta.append(beta_inv)
#    C1 = C1_inv
#    C2 = C2_inv
#    beta = beta_inv
#    print(C1C2BetaH[0])
#    print(C1C2BetaH[0][3:])
    ObsBin_plus = update_depth(ObsBin_plus, C1C2BetaH[0][3:], liste_evt)
    return ObsBin_plus, C1C2BetaH

def calib_C1C2betagammaH(liste_evt, ObsBin_plus, C1, C2, beta, gamma,
                       NmaxIter=50, add_I0=True,
                       inverse_depth=False, inverse_I0=False):
    ObsBin_plus = initialize_HI0(ObsBin_plus, liste_evt, beta, 0)
    ObsBin_plus.loc[:, 'Hmin_ini'] =  ObsBin_plus.loc[:, 'Hmin']
    ObsBin_plus.loc[:, 'Hmax_ini'] =  ObsBin_plus.loc[:, 'Hmax']

    if add_I0:
        ObsBin_plus = add_I0as_datapoint(ObsBin_plus, liste_evt)
        ObsBin_plus.sort_values(by=['EVID', 'I'], inplace=True)
    #eta_values_tested = np.arange(0, 1.01, 0.01)
    eta_values_tested = [0]
#    stock_wrms = []
#    stock_C1 = []
#    stock_C2 = []
#    stock_beta = []
    for eta in eta_values_tested:
#        sigma = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).compute_2Dsigma(eta, col='StdI')
#        C1C2BetaH = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).do_wls_C1C2BetaH(sigma=sigma)
        sigma = WLSIC.WLS(ObsBin_plus, C1, C2, beta, gamma).compute_2Dsigma(eta)
        C1C2BetaH = WLSIC.WLS(ObsBin_plus, C1, C2, beta, gamma).do_wls_C1C2BetaGammaH(sigma=sigma)
        #sigma = self.Obsbin_plus['StdI'].values
#        C1_inv = C1C2BetaH[0][0]
#        C2_inv = C1C2BetaH[0][1]
#        beta_inv = C1C2BetaH[0][2]
#        wrms = calcul_wrms_C1C2betagamma(ObsBin_plus, C1_inv, C2_inv, beta_inv, 0)
#        stock_wrms.append(wrms)
#        stock_C1.append(C1_inv)
#        stock_C2.append(C2_inv)
#        stock_beta.append(beta_inv)
#    C1 = C1_inv
#    C2 = C2_inv
#    beta = beta_inv
#    print(C1C2BetaH[0])
#    print(C1C2BetaH[0][3:])
    ObsBin_plus = update_depth(ObsBin_plus, C1C2BetaH[0][3:], liste_evt)
    return ObsBin_plus, C1C2BetaH


def calib_C1C2betaHb(liste_evt, ObsBin_plus, C1, C2, beta,
                   NmaxIter=50, add_I0=True,
                   inverse_depth=False, inverse_I0=False):
    ObsBin_plus = initialize_HI0(ObsBin_plus, liste_evt, beta, 0)
    ObsBin_plus.loc[:, 'Hmin_ini'] =  ObsBin_plus.loc[:, 'Hmin']
    ObsBin_plus.loc[:, 'Hmax_ini'] =  ObsBin_plus.loc[:, 'Hmax']

    if add_I0:
        ObsBin_plus = add_I0as_datapoint(ObsBin_plus, liste_evt)
        ObsBin_plus.sort_values(by=['EVID', 'I'], inplace=True)
    #eta_values_tested = np.arange(0, 1.01, 0.01)
    eta_values_tested = [0.1]
#    stock_wrms = []
#    stock_C1 = []
#    stock_C2 = []
#    stock_beta = []
    for eta in eta_values_tested:
#        sigma = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).compute_2Dsigma(eta, col='StdI')
#        C1C2BetaH = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).do_wls_C1C2BetaH(sigma=sigma)
#        sigma = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).compute_2Dsigma(eta)
        C1C2BetaH = WLSIC_2.WLS(ObsBin_plus, C1, C2, beta, 0).do_wls_C1C2BetaH2()
        #sigma = self.Obsbin_plus['StdI'].values
#        C1_inv = C1C2BetaH[0][0]
#        C2_inv = C1C2BetaH[0][1]
#        beta_inv = C1C2BetaH[0][2]
#        wrms = calcul_wrms_C1C2betagamma(ObsBin_plus, C1_inv, C2_inv, beta_inv, 0)
#        stock_wrms.append(wrms)
#        stock_C1.append(C1_inv)
#        stock_C2.append(C2_inv)
#        stock_beta.append(beta_inv)
#    C1 = C1_inv
#    C2 = C2_inv
#    beta = beta_inv
    ObsBin_plus = update_depth(ObsBin_plus, C1C2BetaH.x[3:], liste_evt)
    return C1C2BetaH, ObsBin_plus


def calib_C1C2beta0(liste_evt, ObsBin_plus, C1, C2, beta,
                   NmaxIter=50, add_I0=True,
                   inverse_depth=False, inverse_I0=False):
    ObsBin_plus.loc[:, 'Hmin_ini'] =  ObsBin_plus.loc[:, 'Hmin']
    ObsBin_plus.loc[:, 'Hmax_ini'] =  ObsBin_plus.loc[:, 'Hmax']
    ObsBin_plus.loc[:, 'StdIo_inv'] =  0.2
    
    
    if add_I0:
        ObsBin_plus = add_I0as_datapoint(ObsBin_plus, liste_evt)
    
    ObsBin_plus = update_XCaCb(ObsBin_plus, beta, 0)
    C1, C2 = initialize_C1C2(ObsBin_plus)
    print(C1, C2)
    suivi_depth = {}
    suivi_I0 = {}
    suivi_C1 = np.array([C1])
    suivi_C2 = np.array([C2])
    suivi_beta = np.array([beta])
    for evid in liste_evt:
        obsbin = ObsBin_plus[ObsBin_plus.EVID==evid]
        depth = obsbin.Depth.values[0]
        I0 = obsbin.Io.values[0]
        suivi_depth[evid] = np.array([depth])
        suivi_I0[evid] = np.array([I0])
    iteration = 0
    #initiate C1, C2 and beta according to given initiate depth and I0
    C1C2Beta = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).do_wls_C1C2Beta()
    C1 = C1C2Beta[0][0]
    C2 = C1C2Beta[0][1]
    beta = C1C2Beta[0][2]
    while iteration < NmaxIter:
        iteration += 1
        #print(beta)
        ObsBin_plus, C1, C2, beta = calib_C1C2beta_unit(liste_evt, ObsBin_plus, beta, C1, C2,
                                                        add_I0=add_I0,
                                                        inverse_depth=inverse_depth, inverse_I0=inverse_I0)
        #print(beta)
        suivi_C1 = np.append(suivi_C1, C1)
        suivi_C2 = np.append(suivi_C2, C2)
        suivi_beta = np.append(suivi_beta, beta)
        for evid in liste_evt:
            obsbin = ObsBin_plus[ObsBin_plus.EVID==evid]
            depth = obsbin.Depth.values[0]
            I0 = obsbin.Io.values[0]
            suivi_depth[evid] = np.append(suivi_depth[evid], depth)
            suivi_I0[evid] = np.append(suivi_I0[evid], I0)
        if iteration >=3:
            if critere_arret_HI0C1C2beta(suivi_depth, suivi_I0,
                                         suivi_C1, suivi_C2, suivi_beta):
                break
    return ObsBin_plus, C1, C2, beta, suivi_beta, suivi_C1, suivi_C2