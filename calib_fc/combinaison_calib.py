# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:28:50 2021

@author: PROVOST-LUD
"""

import pandas as pd
import numpy as np
import WLSIC
import WLSIC_2
#import statsmodels.formula.api as sm
import statsmodels.api as sm
#import statsmodels.regression.linear_model as linm
#import sys
# sys.path.append('../postprocessing_fc')
#from wrms import plot_wrms_withHI0lines, plot_wrms_beta, plot_wrms_beta_1evt
#from wrms import getHline_in_HI0wrms_space, getI0line_in_HI0wrms_space
#from wrms import calcul_wrms_C1C2betagamma
from prepa_data import add_I0as_datapoint
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
    condition = (ConvrateBeta <= stop_beta) and (
        ConvrateDepth <= stop_depth) and (ConvrateI0 <= stop_I0)
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
    condition = (ConvrateBeta <= stop_beta) and (ConvrateGamma <= stop_gamma) and (
        ConvrateDepth <= stop_depth) and (ConvrateI0 <= stop_I0)
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
    ls = '-'
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
        obsbin = ObsBin_plus[ObsBin_plus.EVID == evid]
        depth = obsbin.Depth.values[0]
        I0 = obsbin.Io.values[0]
        suivi_depth[evid] = np.array([depth])
        suivi_I0[evid] = np.array([I0])
    ObsBin_plus.loc[:, 'Hmin_ini'] = ObsBin_plus.loc[:, 'Hmin']
    ObsBin_plus.loc[:, 'Hmax_ini'] = ObsBin_plus.loc[:, 'Hmax']

    while iteration < NmaxIter:
        ObsBin_plus, beta = calib_attBeta_Kov_unit(
            liste_evt, ObsBin_plus, beta)
        suivi_beta = np.append(suivi_beta, beta)

        for evid in liste_evt:
            obsbin = ObsBin_plus[ObsBin_plus.EVID == evid]
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
                resBetaStd = WLSIC.WLS_Kov(
                    ObsBin_plus, beta, 0).do_wls_beta_std()
                cov_beta = resBetaStd[1]
                break
    if iteration >= NmaxIter:
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
        obsbin = ObsBin_plus[ObsBin_plus.EVID == evid]
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
        while mini_iteration < 2:
            # Inversion de la profondeur
            resH = WLSIC.WLSIC_Kov_oneEvt(
                obsbin, depth, beta, 0, I0).do_wlsic_depth(Hmin, Hmax)
            new_depth = resH[0][0]
            ObsBin_plus.loc[ObsBin_plus.EVID == evid, 'Depth'] = new_depth
            # Inversion de l'intensite epicentrale
            resI0 = WLSIC.WLSIC_Kov_oneEvt(
                obsbin, new_depth, beta, 0, I0).do_wlsic_I0(I0_min, I0_max)
            I0_inv = resI0[0][0]
            I0 = I0_inv
            mini_iteration += 1
        ObsBin_plus.loc[ObsBin_plus.EVID == evid, 'Io'] = I0_inv
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
        obsbin = ObsBin_plus[ObsBin_plus.EVID == evid]
        depth = obsbin.Depth.values[0]
        I0 = obsbin.Io.values[0]
        suivi_depth[evid] = np.array([depth])
        suivi_I0[evid] = np.array([I0])
    ObsBin_plus.loc[:, 'Hmin_ini'] = ObsBin_plus.loc[:, 'Hmin']
    ObsBin_plus.loc[:, 'Hmax_ini'] = ObsBin_plus.loc[:, 'Hmax']

    while iteration < NmaxIter:

        ObsBin_plus, beta, gamma = calib_attBetaGamma_Kov_unit(
            liste_evt, ObsBin_plus, beta, gamma)
        # print(beta)
        suivi_beta = np.append(suivi_beta, beta)
        suivi_gamma = np.append(suivi_gamma, gamma)
        for evid in liste_evt:
            obsbin = ObsBin_plus[ObsBin_plus.EVID == evid]
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

    resBetaGammaStd = WLSIC.WLS_Kov(
        ObsBin_plus, beta, gamma).do_wls_betagamma_std()
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
        obsbin = ObsBin_plus[ObsBin_plus.EVID == evid]
        depth = obsbin.Depth.values[0]
        Hmin = obsbin.Hmin.values[0]
        Hmax = obsbin.Hmax.values[0]
        obsbin.loc[:, 'Hypo'] = np.sqrt(
            obsbin['Depi'].values.astype(float)**2 + depth**2)
        I0 = obsbin.Io.values[0]
        I0_min = obsbin.Io_ini.values[0] - 2*obsbin.Io_std.values[0]
        I0_max = obsbin.Io_ini.values[0] + 2*obsbin.Io_std.values[0]
        mini_iteration = 1
        while mini_iteration < 2:
            # Inversion de la profondeur
            resH = WLSIC.WLSIC_Kov_oneEvt(
                obsbin, depth, beta, gamma, I0).do_wlsic_depth(Hmin, Hmax)
            new_depth = resH[0][0]
            ObsBin_plus.loc[ObsBin_plus.EVID == evid, 'Depth'] = new_depth
            # Inversion de l'intensite epicentrale
            resI0 = WLSIC.WLSIC_Kov_oneEvt(
                obsbin, new_depth, beta, gamma, I0).do_wlsic_I0(I0_min, I0_max)
            I0_inv = resI0[0][0]
            I0 = I0_inv
            mini_iteration += 1
            ObsBin_plus.loc[ObsBin_plus.EVID == evid, 'Io'] = I0_inv

    resBetaGamma = WLSIC.WLS_Kov(ObsBin_plus, beta, gamma).do_wls_betagamma()
    beta = resBetaGamma[0][0]
    gamma = resBetaGamma[0][1]
    return ObsBin_plus, beta, gamma


def critere_arret_HI0(suivi_depth, suivi_I0,
                      stop_depth=0.05, stop_I0=0.01):
    """
    Check if the conditions are met to stop the inversion process in the initialize_HI0() function.
    
    Parameters
    ----------
    suivi_depth : dict
        Dictionary that contains by earthquake ID (key of the dictionary) by chronological
        order the depth for each iteration of the inversion scheme of the initialize_HI0() function.
    suivi_I0 : dict
        Dictionary that contains by earthquake ID (key of the dictionary) by chronological
        order the epicentral intensity for each iteration of the inversion scheme of the initialize_HI0() function.
    stop_depth : float, optional
        Criterion to achieve to end the inversion process of the initialize_HI0() function. The default is 0.05.
        To compute the value compared to this criterion, first the mean of the difference of the three last 
        depths in the iteration is calculated for each earthquake. The means of all calibration earthquakes are then added.
    stop_I0 : float, optional
        Criterion to achieve to end the inversion process of the initialize_HI0() function. The default is 0.01.
        To compute the value compared to this criterion, first the mean of the difference of the three last 
        epicentral intensities in the iteration is calculated for each earthquake. The means of all calibration earthquakes are then added.

    Returns
    -------
    condition : boolean
        Equal to True if the values compared to both stop_I0 and stop_depth are smaller to
        stop_I0 and stop_depth.

    """
    ConvrateDepth = 0
    ConvrateI0 = 0
    NminIter = 3
    for evid in suivi_depth.keys():
        last3_depth = suivi_depth[evid][-3:]
        last3_I0 = suivi_I0[evid][-3:]
        ConvrateDepth = ConvrateDepth + sum(abs(np.diff(last3_depth)))/NminIter
        ConvrateI0 = ConvrateI0 + sum(abs(np.diff(last3_I0)))/NminIter
    ConvrateDepth = ConvrateDepth/len(suivi_depth.keys())
    ConvrateI0 = ConvrateI0/len(suivi_depth.keys())
    condition = (ConvrateDepth <= stop_depth) and (ConvrateI0 <= stop_I0)
    return condition


def initialize_HI0(ObsBin_plus, liste_evt, beta, gamma, NmaxIter=50):
    """
    Compute a depth and an epicentral intensity compatible with a beta and a gamma
    value using the Koveslighety equation:
        I = I0 + beta.log10(hypo/depth) + gamma.(hypo-depth)
    with I0 the epicentral intensity and hypo the hyocentral distance
    Depth and epicentral intensity are estimated 
    within bounds representing their uncertainties. Depth and epicentral intensity are estimated
    for all earthquakes listed in liste_evt.
    For more information, see Provost (in writing), CalIPE

    Parameters
    ----------
    ObsBin_plus : pandas.DataFrame
        DataFrame with all isoseismal radii (intensity bin) used in the inversion. The dataframe
        should contain the isoseismal radii of each earthquake listed in liste_evt. The mandatory columns are:
            EVID : ID of the earthquake
            Depth : initial depth of the earthquake
            Hmin : minimal value of the depth uncertainty bounds
            Hmax : maximal value of the depth uncertainty bounds
            I : intensity value of the isoseismal radii
            StdI : uncertainty associated to I
            Depi : epcientral distance of the isoseismal radii
            I0 : initial epicentral intensity of the earthquake
            Io_ini : same as I0
            Io_std : uncertainty associated to the epicentral intensity value
    liste_evt : list
        list the earthquake whose depths and epicentral intensities will be inverted
    beta : float
        beta coefficient in the Koveslighety equation.
    gamma : float
        gamma coefficient in the Koveslighety equation.
    NmaxIter : int, optional
        Maximal number of iteration allowed. The default is 50.

    Returns
    -------
    ObsBin_plus : pandas.DataFrame
        One column is added to the input ObsBin_plus, called StdIo_inv. This
        column contains the output standard deviation of the inversion of epicentral intensity.
        Columns I0 and Depth are updated with the output values ot the invversion process. 

    """
    # a tester
    suivi_depth = {}
    suivi_I0 = {}
    for evid in liste_evt:
        obsbin = ObsBin_plus[ObsBin_plus.EVID == evid].copy()
        depth = obsbin.Depth.values[0]
        Hmin = obsbin.Hmin.values[0]
        Hmax = obsbin.Hmax.values[0]
        obsbin.loc[:, 'Hypo'] = np.sqrt(
            obsbin['Depi'].values.astype(float)**2 + depth**2)
        I0 = obsbin.Io.values[0]
        I0_min = obsbin.Io_ini.values[0] - 2*obsbin.Io_std.values[0]
        I0_max = obsbin.Io_ini.values[0] + 2*obsbin.Io_std.values[0]
        suivi_depth[evid] = np.array([depth])
        suivi_I0[evid] = np.array([I0])
        mini_iteration = 0
        while mini_iteration < NmaxIter:
            # Inversion de la profondeur
            resH = WLSIC.WLSIC_Kov_oneEvt(
                obsbin, depth, beta, gamma, I0).do_wlsic_depth(Hmin, Hmax)
            new_depth = resH[0][0]
            ObsBin_plus.loc[ObsBin_plus.EVID == evid, 'Depth'] = new_depth
            # Inversion de l'intensite epicentrale
            resI0 = WLSIC.WLSIC_Kov_oneEvt(
                obsbin, new_depth, beta, gamma, I0).do_wlsic_I0(I0_min, I0_max)
            I0_inv = resI0[0][0]
            StdI0_inv = np.sqrt(np.diag(resI0[1][0]))

            I0 = I0_inv
            mini_iteration += 1
            ObsBin_plus.loc[ObsBin_plus.EVID == evid, 'Io'] = I0_inv
            ObsBin_plus.loc[ObsBin_plus.EVID == evid, 'StdIo_inv'] = StdI0_inv
            suivi_depth[evid] = np.append(suivi_depth[evid], new_depth)
            suivi_I0[evid] = np.append(suivi_I0[evid], I0)
            if mini_iteration > 3:
                if critere_arret_HI0(suivi_depth, suivi_I0):
                    break
    return ObsBin_plus


def initialize_C1C2(ObsBin_plus):
    """
    Give initial values for C1 and C2 coefficient for the followign equation:
        I = C1 + C2.Mag + beta.log10(hypo) + gamma.hypo
    where I is the intensity at a given hypocentral distance hypo,
    Mag is the magnitude of the earthquake and beta and gamma are attenuation coefficients, supposed known.
    

    Parameters
    ----------
    ObsBin_plus : pandas.DataFrame
        DataFrame with all isoseismal radii (intensity bin) used in the inversion.
        The mandatory columns are:
            EVID : ID of the earthquake
            Depth : initial depth of the earthquake
            I : intensity value of the isoseismal radii
            Depi : epcientral distance of the isoseismal radii
            beta : the attenuation coefficient beta
            gamma : the attenuation coefficient gamma
            eqStdM : the equivalent standard deviation used to weight the data used in the inversion.
                     The weights are equal to 1/obsgp.eqStdM.values**2

    Returns
    -------
    C1 : float
        C1 coefficient value
    C2 : float
        C2 coefficient value

    """
    ObsBin_plus.loc[:, 'intercept'] = 1
    ObsBin_plus.loc[:, 'Hypo'] = ObsBin_plus.apply(
        lambda row: np.sqrt(row['Depi']**2+row['Depth']**2), axis=1)
    ObsBin_plus.loc[:, 'X'] = ObsBin_plus.apply(
        lambda row: row['I'] - row['beta']*np.log10(row['Hypo'] - row['gamma']*row['Hypo']), axis=1)
    ObsBin_plus = ObsBin_plus.astype({'Mag': float})
    obsgp = ObsBin_plus[['EVID', 'X', 'intercept',
                         'eqStdM', 'Mag']].groupby('EVID').mean()
    resultCaCb = sm.WLS(obsgp.Mag, obsgp[['intercept', 'X']],
                        weights=1/obsgp.eqStdM.values**2).fit()
    # print(resultCaCb)
    Ca = resultCaCb.params[1]
    Cb = resultCaCb.params[0]
    C2 = 1/Ca
    C1 = -Cb/Ca
    return C1, C2


def check_ifHlim_ok(obsbin, beta, C1, C2):
    """
    Function that check if the depth bounds are compatible with the epicentral intensity
    and its uncertainty. If not the function returns new depth bounds compatible
    with the epicentral intensity and its uncertainty. the following equation is used:
        I = C1 + C2.Mag + beta.log10(hypo) 
    where I is the intensity at a given hypocentral distance hypo,
    Mag is the magnitude of the earthquake, beta, C1 and C2 are coefficients provided by the user.

    Parameters
    ----------
    obsbin : pandas.Dataframe
        DataFrame with all isoseismal radii (intensity bin) of one earthquake. The mandatory columns are:
            Io_ini : initial epicentral intensity of the earthquake
            Io_std : uncertainty associated to the epicentral intensity value 
            Mag : magnitude of the earthquake
            Hmin_ini : initial minimal depth bounds
            Hmax_ini : initial maximal depth bounds
    beta : float
        beta coefficient in the equation used to check the depth bounds.
    C1 : float
        C1 coefficient in the equation used to check the depth bounds.
    C2 : float
        C2 coefficient in the equation used to check the depth bounds.

    Returns
    -------
    Hmin : float
        if needed, new minimal depth bounds compatible
        with the epicentral intensity and its uncertainty. 
        Else, the output value is equal to the initial minimal depth bounds.
    Hmax : float
        if needed, new maximal depth bounds compatible
        with the epicentral intensity and its uncertainty. 
        Else, the output value is equal to the initial maximal depth bounds.

    """
    # Verif des Hlim  - compatible avec I0+-2std
    # print(obsbin.columns)
    Ioinf = obsbin.Io_ini.values[0] - 2*obsbin.Io_std.values[0]
    Iosup = obsbin.Io_ini.values[0] + 2*obsbin.Io_std.values[0]
    Hmin_ini = obsbin.Hmin_ini.values[0]
    Hmax_ini = obsbin.Hmax_ini.values[0]
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


def calib_C1C2H(liste_evt, ObsBin_plus, 
                NmaxIter=50, add_I0=True):
    """
    Function that calibrate the C1 and the C2 coefficient in the following equation:
        I = C1 + C2.Mag + beta.log10(hypo) + gamma.hypo
    where I is the intensity at a given hypocentral distance hypo,
    Mag is the magnitude of the earthquake, beta, gamma, C1 and C2 are coefficients.    
    Depth of the calibration earthquakes is also inverted, within its uncertainties. The 
    depth parameter is in the hypo term:
        hypo = sqrt(depi**2 + depth**2)
    Regionalization of C1 is possible, with a maximal limit of 4 regions.

    Parameters
    ----------
    liste_evt : list
        list the earthquake whose macroseismic data and metadat will be used to calibrate the C1 and C2 coefficients.
    ObsBin_plus : pandas.DataFrame
        DataFrame with all isoseismal radii (intensity bin) used in the inversion. The dataframe
        should contain the isoseismal radii of each earthquake listed in liste_evt. The mandatory columns are:
            EVID : ID of the earthquake
            RegID : region ID of the location of the earthquake
            Depth : initial depth of the earthquake
            I : intensity value of the isoseismal radii
            StdI : uncertainty associated to I
            Depi : epicentral distance of the isoseismal radii
            beta : the attenuation coefficient beta, corresponding to the region of the location of the earthquake
            gamma : the attenuation coefficient gamma, corresponding to the region of the location of the earthquake
            Mag : magnitude of the earthquake
            eqStdM : the equivalent standard deviation used to weight the data used in the inversion.
                     The weights are equal to 1/obsgp.eqStdM.values**2
            
    
    NmaxIter : int, optional
       Maximal number of iteration allowed. The default is 50.
    add_I0 : boolean, optional
        option that add the epicentral intensity to the intensity data used in ivnersion.
        The default is True.

    Returns
    -------
    ObsBin_plus : pandas.DataFrame
        Same as ObsBin_plus input, but with the Depth column updated.
    resC1regC2 : array
        array with two elements. First element is the popt output of scipy.optimize.curve_fit,
        i.e. the optimal values for the parameters so that the sum of the squared residuals
        of f(xdata, *popt) - ydata is minimized. Second element is pcov output of scipy.optimize.curve_fit,
        i.e. the estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
        To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov))

    """
    liste_region = np.unique(ObsBin_plus.RegID.values.astype(float))
    for regID in liste_region:
        beta = ObsBin_plus[ObsBin_plus.RegID == regID].beta.values[0]
        liste_evt = ObsBin_plus[ObsBin_plus.RegID == regID].EVID.values
        ObsBin_plus[ObsBin_plus.RegID == regID] = initialize_HI0(
            ObsBin_plus[ObsBin_plus.RegID == regID], liste_evt, beta, 0)
    C1ini, C2 = initialize_C1C2(ObsBin_plus)
    C1_dict = {}
    for regID in liste_region:
        C1_dict[regID] = C1ini
    # if add_I0:
    #     ObsBin_plus = add_I0as_datapoint(ObsBin_plus, liste_evt)
    #     ObsBin_plus.sort_values(by=['EVID', 'I'], inplace=True)

   
    suivi_depth = {}
    mini_iteration = 0
    while mini_iteration < NmaxIter:
        for evid in liste_evt:
            obsbin = ObsBin_plus[ObsBin_plus.EVID == evid].copy()
            beta = obsbin.beta.values[0]
            gamma = obsbin.gamma.values[0]
            depth = obsbin.Depth.values[0]
            mag = obsbin.Mag.values[0]
            Hmin = obsbin.Hmin.values[0]
            Hmax = obsbin.Hmax.values[0]
            regID = obsbin.RegID.values[0]
            C1 = C1_dict[regID]
            # Verification Hlim compatible avec I0
            Hmin, Hmax = check_ifHlim_ok(obsbin, beta, C1, C2)
            obsbin.loc[:, 'Hypo'] = np.sqrt(
                obsbin['Depi'].values.astype(float)**2 + depth**2)
            suivi_depth[evid] = np.array([depth])
            # Inversion de la profondeur 
            resH = WLSIC.WLSIC_oneEvt(
                obsbin, depth, mag, beta, gamma, C1, C2).do_wlsic_depth(Hmin, Hmax)
            new_depth = resH[0][0]
            ObsBin_plus.loc[ObsBin_plus.EVID == evid, 'Depth'] = new_depth

            suivi_depth[evid] = np.append(suivi_depth[evid], new_depth)
        poids = ((1/(ObsBin_plus.StdI.values**2)) *
                 (1/(ObsBin_plus.eqStdM.values**2))).astype(float)
        resC1regC2 = WLSIC.WLS(ObsBin_plus, C1, C2, beta, gamma).do_linregressC1regC2(ftol=2e-3,
                                                                                      max_nfev=100,
                                                                                      sigma=np.sqrt(1/poids))
        mini_iteration += 1
    return ObsBin_plus, resC1regC2


def calib_C1C2(liste_evt, ObsBin_plus, 
               NmaxIter=50, add_I0=True):
    """
    Function that calibrate the C1 and the C2 coefficient in the following equation:
        I = C1 + C2.Mag + beta.log10(hypo) + gamma.hypo
    where I is the intensity at a given hypocentral distance hypo,
    Mag is the magnitude of the earthquake, beta, gamma, C1 and C2 are coefficients.
    Regionalization of C1 is possible, with a maximal limit of 4 regions.

    Parameters
    ----------
    liste_evt : list
        list the earthquake whose macroseismic data and metadat will be used to calibrate the C1 and C2 coefficients.
    ObsBin_plus : pandas.DataFrame
        DataFrame with all isoseismal radii (intensity bin) used in the inversion. The dataframe
        should contain the isoseismal radii of each earthquake listed in liste_evt. The mandatory columns are:
            EVID : ID of the earthquake
            RegID : region ID of the location of the earthquake
            Depth : initial depth of the earthquake
            I : intensity value of the isoseismal radii
            StdI : uncertainty associated to I
            Depi : epicentral distance of the isoseismal radii
            beta : the attenuation coefficient beta, corresponding to the region of the location of the earthquake
            gamma : the attenuation coefficient gamma, corresponding to the region of the location of the earthquake
            Mag : magnitude of the earthquake
            eqStdM : the equivalent standard deviation used to weight the data used in the inversion.
                     The weights are equal to 1/obsgp.eqStdM.values**2
    NmaxIter : int, optional
       Maximal number of iteration allowed. The default is 50.
    add_I0 : boolean, optional
        option that add the epicentral intensity to the intensity data used in inversion.
        The default is True.

    Returns
    -------
    ObsBin_plus : pandas.DataFrame
        Same as ObsBin_plus input
    resC1regC2 : array
        array with two elements. First element is the popt output of scipy.optimize.curve_fit,
        i.e. the optimal values for the parameters so that the sum of the squared residuals
        of f(xdata, *popt) - ydata is minimized. Second element is pcov output of scipy.optimize.curve_fit,
        i.e. the estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
        To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov))

    """
    # print(ObsBin_plus.columns)
    liste_region = np.unique(ObsBin_plus.RegID.values.astype(float))
    for regID in liste_region:
        beta = ObsBin_plus[ObsBin_plus.RegID == regID].beta.values[0]
        gamma = ObsBin_plus[ObsBin_plus.RegID == regID].gamma.values[0]
        liste_evt = ObsBin_plus[ObsBin_plus.RegID == regID].EVID.values
        ObsBin_plus[ObsBin_plus.RegID == regID] = initialize_HI0(
            ObsBin_plus[ObsBin_plus.RegID == regID], liste_evt, beta, gamma)
    #ObsBin_plus = initialize_HI0(ObsBin_plus, liste_evt, beta, 0)
    C1, C2 = initialize_C1C2(ObsBin_plus)
    # if add_I0:
    #     ObsBin_plus = add_I0as_datapoint(ObsBin_plus, liste_evt)
    #     ObsBin_plus.sort_values(by=['EVID', 'I'], inplace=True)

    poids = ((1/(ObsBin_plus.StdI.values**2)) *
             (1/(ObsBin_plus.eqStdM.values**2))).astype(float)

    resC1regC2 = WLSIC.WLS(ObsBin_plus, C1, C2, beta, gamma).do_linregressC1regC2(ftol=2e-3,
                                                                                  max_nfev=100,
                                                                                  sigma=np.sqrt(1/poids))
    return ObsBin_plus, resC1regC2


def update_depth(ObsBin_plus, depths, liste_evt):
    """
    function that updates the depth in the ObsBin_plus dataframe

    Parameters
    ----------
    ObsBin_plus : pandas.DataFrame
        DataFrame with all isoseismal radii (intensity bin) used in the inversion. The dataframe
        should contain the isoseismal radii of each earthquake listed in liste_evt. 
        The mandatories columns for this function are:
            EVID : ID of the earthquake
            Depth : depth of the earthquake
            
        TThe ObsBin_plus dataframe should at least contains information about the isoseismal radii, i.e.
        I (intensity value of the isoseismal radii) and Depi (epicentral distance of the isoseismal radii) but
        those columns are not necessary in this function.
    depths : list
        list of depths of the earthquakes listed in liste_evt. Order in depths and liste_evt must be the same.
    liste_evt : list
        list of earthquakes listed. Order in depths and liste_evt must be the same.

    Returns
    -------
    ObsBin_plus : pandas.DataFrame
        Same as ObsBin_plus input but with the column Depth updated with the depths input.

    """
    for compt, evid in enumerate(liste_evt):
        depth = depths[compt]
        ObsBin_plus.loc[ObsBin_plus.EVID == evid, 'Depth'] = depth
    return ObsBin_plus


def calib_C1C2betaH(liste_evt, ObsBin_plus, C1, C2, beta,
                    NmaxIter=50, add_I0=True):
    """
    Function that calibrate the C1, the C2and the beta coefficients in the following equation:
        I = C1 + C2.Mag + beta.log10(hypo) 
    where I is the intensity at a given hypocentral distance hypo,
    Mag is the magnitude of the earthquake, beta, gamma, C1 and C2 are coefficients.
    Depth of the calibration earthquakes is also inverted, within its uncertainties. The 
    depth parameter is in the hypo term:
        hypo = sqrt(depi**2 + depth**2)

    Parameters
    ----------
    liste_evt : list
        list the earthquake whose macroseismic data and metadat will be used to calibrate the C1, C2 and beta coefficients.
    ObsBin_plus : pandas.DataFrame
        DataFrame with all isoseismal radii (intensity bin) used in the inversion. The dataframe
        should contain the isoseismal radii of each earthquake listed in liste_evt. The mandatory columns are:
            EVID : ID of the earthquake
            RegID : region ID of the location of the earthquake
            Depth : initial depth of the earthquake
            I : intensity value of the isoseismal radii
            StdI : uncertainty associated to I
            Depi : epicentral distance of the isoseismal radii
            beta : the attenuation coefficient beta, corresponding to the region of the location of the earthquake
            gamma : the attenuation coefficient gamma, corresponding to the region of the location of the earthquake
            Mag : magnitude of the earthquake
            eqStdM : the equivalent standard deviation used to weight the data used in the inversion.
                     The weights are equal to 1/obsgp.eqStdM.values**2
            Hmin : lower bound of uncertainty associated to depth
            Hmax : upper bound of uncertainty associated to depth
    C1 : float
        Initial value of the C1 coefficient.
    C2 : float
        Initial value of the C2 coefficient.
    beta : float
        Initial value of the beta coefficient.
    NmaxIter : int, optional
        Maximal number of iteration allowed. The default is 50.
    add_I0 : boolean, optional
        option that add the epicentral intensity to the intensity data used in inversion.
        The default is True.

    Returns
    -------
    ObsBin_plus : pandas.DataFrame
        Same as the input, but with the depth column updated.
    C1C2BetaH : array
       array with two array elements. First element is the popt output of scipy.optimize.curve_fit,
       i.e. the optimal values for the parameters so that the sum of the squared residuals
       of f(xdata, *popt) - ydata is minimized. The three first elments in this array are the C1, the C2 and the beta
       coefficient. The other elements are the inverted depths, in the same order as liste_evt.
       Second element is pcov output of scipy.optimize.curve_fit,
       i.e. the estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
       To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov))

    """
    ObsBin_plus = initialize_HI0(ObsBin_plus, liste_evt, beta, 0)
    ObsBin_plus.loc[:, 'Hmin_ini'] = ObsBin_plus.loc[:, 'Hmin']
    ObsBin_plus.loc[:, 'Hmax_ini'] = ObsBin_plus.loc[:, 'Hmax']

    if add_I0:
        ObsBin_plus = add_I0as_datapoint(ObsBin_plus, liste_evt)
        ObsBin_plus.sort_values(by=['EVID', 'I'], inplace=True)
    eta_values_tested = [0]

    for eta in eta_values_tested:
        sigma = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).compute_2Dsigma(eta)
        C1C2BetaH = WLSIC.WLS(ObsBin_plus, C1, C2, beta,
                              0).do_wls_C1C2BetaH(sigma=sigma)
    ObsBin_plus = update_depth(ObsBin_plus, C1C2BetaH[0][3:], liste_evt)
    return ObsBin_plus, C1C2BetaH


def calib_C1C2betagammaH(liste_evt, ObsBin_plus, C1, C2, beta, gamma,
                         NmaxIter=50, add_I0=True):
    """
    Function that calibrate the C1, the C2, the beta and the gamma coefficients in the following equation:
        I = C1 + C2.Mag + beta.log10(hypo) + gamma.hypo
    where I is the intensity at a given hypocentral distance hypo,
    Mag is the magnitude of the earthquake, beta, gamma, C1 and C2 are coefficients.
    Depth of the calibration earthquakes is also inverted, within its uncertainties. The 
    depth parameter is in the hypo term:
        hypo = sqrt(depi**2 + depth**2)

    Parameters
    ----------
    liste_evt : list
        list the earthquake whose macroseismic data and metadat will be used to calibrate the C1, C2 and beta coefficients.
    ObsBin_plus : pandas.DataFrame
        DataFrame with all isoseismal radii (intensity bin) used in the inversion. The dataframe
        should contain the isoseismal radii of each earthquake listed in liste_evt. The mandatory columns are:
            EVID : ID of the earthquake
            RegID : region ID of the location of the earthquake
            Depth : initial depth of the earthquake
            I : intensity value of the isoseismal radii
            StdI : uncertainty associated to I
            Depi : epicentral distance of the isoseismal radii
            beta : the attenuation coefficient beta, corresponding to the region of the location of the earthquake
            gamma : the attenuation coefficient gamma, corresponding to the region of the location of the earthquake
            Mag : magnitude of the earthquake
            eqStdM : the equivalent standard deviation used to weight the data used in the inversion.
                     The weights are equal to 1/obsgp.eqStdM.values**2
            Hmin : lower bound of uncertainty associated to depth
            Hmax : upper bound of uncertainty associated to depth
    C1 : float
        Initial value of the C1 coefficient.
    C2 : float
        Initial value of the C2 coefficient.
    beta : float
        Initial value of the beta coefficient.
    gamma : float
        Initial value of the gamma coefficient.
    NmaxIter : int optional
        Maximal number of iteration allowed. The default is 50.
    add_I0 : boolean, optional
        option that add the epicentral intensity to the intensity data used in inversion. The default is True.

    Returns
    -------
    ObsBin_plus : pandas.DataFrame
        Same as the input, but with the depth column updated.
    C1C2BetaH : array
       array with two array elements. First element is the popt output of scipy.optimize.curve_fit,
       i.e. the optimal values for the parameters so that the sum of the squared residuals
       of f(xdata, *popt) - ydata is minimized. The four first elments in this array are the C1, the C2, the beta
       and the gamma coefficients. The other elements are the inverted depths, in the same order as liste_evt.
       Second element is pcov output of scipy.optimize.curve_fit,
       i.e. the estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
       To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov))

    """
    ObsBin_plus = initialize_HI0(ObsBin_plus, liste_evt, beta, 0)
    ObsBin_plus.loc[:, 'Hmin_ini'] = ObsBin_plus.loc[:, 'Hmin']
    ObsBin_plus.loc[:, 'Hmax_ini'] = ObsBin_plus.loc[:, 'Hmax']

    if add_I0:
        ObsBin_plus = add_I0as_datapoint(ObsBin_plus, liste_evt)
        ObsBin_plus.sort_values(by=['EVID', 'I'], inplace=True)
    #eta_values_tested = np.arange(0, 1.01, 0.01)
    eta_values_tested = [0]

    for eta in eta_values_tested:
        #        sigma = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).compute_2Dsigma(eta, col='StdI')
        #        C1C2BetaH = WLSIC.WLS(ObsBin_plus, C1, C2, beta, 0).do_wls_C1C2BetaH(sigma=sigma)
        sigma = WLSIC.WLS(ObsBin_plus, C1, C2, beta,
                          gamma).compute_2Dsigma(eta)
        C1C2BetaH = WLSIC.WLS(ObsBin_plus, C1, C2, beta,
                              gamma).do_wls_C1C2BetaGammaH(sigma=sigma)
    ObsBin_plus = update_depth(ObsBin_plus, C1C2BetaH[0][3:], liste_evt)
    return ObsBin_plus, C1C2BetaH
