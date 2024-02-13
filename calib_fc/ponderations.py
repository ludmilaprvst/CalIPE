# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:25:33 2021

@author: PROVOST-LUD
"""

import pandas as pd
import numpy as np
import matplotlib.path as mpltPath
import os.path


def evt_weights(obsbin_plus, option_ponderation):
    """
    Function that attribute a weight to the intensity data
    :param obsbin_plus:dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have a column named StdI.
                        The other mandatory columns depends on the chosen weighting
                        scheme:
                            - StdI: the associated standard deviation,
                            - EVID: the earthquake ID,
                            - StdM: standard deviation associated to the magnitude
                            value
                            - Hmin: minimal (shallow) limit of known depth
                            - Hmax: maximal (deep) limit of known depth
                            - RegID: the region ID in which the earthquake
                            is located
    :param option_ponderation: type of weighting whished. Possible values:
                               - 'IStdI',
                               - 'IStdI_evtUni',
                               - 'IStdI_evtStdM',
                               - 'IStdI_evtStdH',
                               - 'IStdI_evtStdM_gRegion'
                               - 'IStdI_evtStdM_gMclass'
    :type obsbin_plus: pandas.DataFrame
    :type option_ponderation: str
    
    :return: a completed obsbin_plus DataFrame, with a eqStd column that contains
             the associated inverse square root of the weights
    """
    # if option_ponderation == 'Ponderation dI':
    #     obsbin_plus = Kovatt_ponderation_dI(obsbin_plus)
    # elif option_ponderation == 'Ponderation evt-uniforme':
    #     obsbin_plus = Kovatt_ponderation_evt_uniforme(obsbin_plus)
    # elif option_ponderation == 'Ponderation evt-reg':
    #     obsbin_plus = Kovatt_ponderation_evt_reg(obsbin_plus)
    # elif option_ponderation == 'Ponderation evt-depth':
    #     obsbin_plus = Kovatt_ponderation_evt_depth(obsbin_plus)
    print(obsbin_plus.columns)
    if option_ponderation == 'IStdI':
        obsbin_plus = weight_IStdI(obsbin_plus)
    elif option_ponderation == 'IStdI_evtUni':
        obsbin_plus = weight_IStdI_evtUni(obsbin_plus)
    elif option_ponderation == 'IStdI_evtStdM':
        obsbin_plus = weight_IStdI_evtStdM(obsbin_plus)
    elif option_ponderation == 'IStdI_evtStdH':
        obsbin_plus = weight_IStdI_evtStdH(obsbin_plus)
    elif option_ponderation == 'IStdI_evtStdM_gRegion':
        obsbin_plus = weight_IStdI_evtStdM_gRegion(obsbin_plus)
    elif option_ponderation == 'IStdI_evtStdM_gMclass':
        obsbin_plus = weight_IStdI_evtStdM_gMclass(obsbin_plus)
    else:
        print('No such ponderation option:')
        print(option_ponderation)
    return obsbin_plus


def weight_IStdI(obsbin_plus):
    """
    Function that compute a weight based on intensity standard deviation
    :param obsbin_plus: dataframe with the binned intensity data for all
                        calibration earthquakes.This dataframe should at least
                        have a column named 'StdI', which is the  standard deviation
                        associated to the binned intensity.
    :type obsbin_plus: pandas.DataFrame
    
    :return: a completed obsbin_plus DataFrame, with a column called 'eqStd'
             that contains the associated inverse square root of the weights
    """
    obsbin_plus.loc[:, 'Poids_int'] = 1/obsbin_plus.loc[:, 'StdI']**2
    obsbin_plus.loc[:, 'eqStd'] = obsbin_plus.loc[:, 'StdI']
    return obsbin_plus


def normaliser_poids_par_evt(evid, obsbin_plus):
    """
    Function which normalizes the weights associated to the binned intensity
    within one event.
    
    :param evid: id of the earthquake for which weights should be normalized
    :param obsbin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 'EVID' and 'Poids_int', which are respectively id
                        of the chosen earthquake and the weight
                        associated to the binned intensity data.
    :type evid: str or float
    :type obsbin_plus: pandas.DataFrame
    :return: a completed obsbin_plus DataFrame, with a Poids_inevt column that contains
             normalized weigths per earthquake. Sum of the weights of one earthquake
             data is be equal to one.
    """
    ind = obsbin_plus[obsbin_plus.EVID==evid].index
    somme_poids_par_evt = np.sum(obsbin_plus.loc[ind, 'Poids_int'])
    obsbin_plus.loc[ind, 'Poids_int_norm'] = obsbin_plus.loc[ind, 'Poids_int']/somme_poids_par_evt
    #obsbin_plus.drop(['Poids_int'], axis=1, inplace=True)
    return obsbin_plus


def weight_evtUni(obsbin_plus):
    """
    Attribute a uniform weight to each event.

    Parameters
    ----------
    obsbin_plus : pandas.DataFrame
        Dataframe with the binned intensity data for all calibration earthquakes.
        This dataframe should at least have a column called "EVID", which contains
        the ID of each event.

    Returns
    -------
    Completed input dataframe, with an additional column 'Poids_evt', which 
    contains the uniform weight of the different event, equal to 1/n, where n
    is the number of event in the input dataframe.

    """
    liste_evt = np.unique(obsbin_plus.EVID.values)
    poids_evt = 1/len(liste_evt)
    obsbin_plus.loc[:, 'Poids_evt'] = poids_evt
    return obsbin_plus

def weight_IStdI_evtUni(obsbin_plus):
    """
    Attribute a weight to each binned intensity data that combine a weight based
    on the intensity associated standard deviation and a event-uniform
    weight. In this scheme, each event has the same weight. In one event, the data
    have a weight based on the intensity standard deviation StdI.
    In practice, the weight_IStdI weighting scheme is applied, then the weights
    are normalized within each event: after normalization, the sum of the weights
    based on StdI within one event is equal to 1. Then the uniform-event weight is
    applied.

    Parameters
    ----------
    obsbin_plus : pandas.DataFrame
        Dataframe with the binned intensity data for all calibration earthquakes.
        This dataframe should at least have the following columns:
            - "EVID": contains the ID of each event,
            - "StdI": the standard deviation associated to each binned intensity
            data.

    Returns
    -------
    obsbin_plus : pandas.DataFrame
        Completed input dataframe, with the following additional columns:
            - 'Poids_evt': the uniform weight of the different event, equal to
            1/n, where n is the number of event in the input dataframe.
            - 'Poids_int_norm': the normalized by event weights associated to StdI,
            - 'Poids': the combined weight
            - 'eqStd': the equivalent standard deviation corresponding to Poids,
            used in all inversion functions.

    """
    obsbin_plus = weight_IStdI(obsbin_plus)
    liste_evt = np.unique(obsbin_plus.EVID.values)
    for evid in liste_evt:
        obsbin_plus = normaliser_poids_par_evt(evid, obsbin_plus)
    obsbin_plus = weight_evtUni(obsbin_plus)
    obsbin_plus.loc[:, 'Poids'] = obsbin_plus.loc[:, 'Poids_int_norm'].astype(float)*obsbin_plus.loc[:, 'Poids_evt']
    obsbin_plus.loc[:, 'Poids'] = obsbin_plus.loc[:, 'Poids']
    obsbin_plus.loc[:, 'eqStd'] = 1/np.sqrt(obsbin_plus.loc[:, 'Poids'])
    return obsbin_plus


def weight_evtStdM(obsbin_plus):
    """
    Attribute a weight based on the magnitude standard deviation StdM to each event.
    Waight = 1/StdM**2, with a minimal value of StdM equal to 0.1.

    Parameters
    ----------
    obsbin_plus : pandas.DataFrame
        Dataframe with the binned intensity data for all calibration earthquakes.
        This dataframe should at least have the following columns:
            - "EVID": the ID of each event,
            - "StdM": the magnitude standard deviation,

    Returns
    -------
    Completed input dataframe, with an additional column 'Poids_evt', which 
    contains the weight based on the magnitude standard deviation StdM to each
    event.

    """
    print(obsbin_plus.columns)
    #obsbin_plus['StdM'] = obsbin_plus['StdM'].astype(float)
    #obsbin_plus_tmp = obsbin_plus[['EVID', 'StdM']]
    print(obsbin_plus.StdM.dtypes)
    gp_evid_obsbinplus = obsbin_plus.groupby('EVID').mean()
    print(gp_evid_obsbinplus.columns)
    poids = 1/(gp_evid_obsbinplus.StdM.values**2)
    max_poids = 1/(0.1**2)*np.ones(len(poids))
    poids = np.min([poids, max_poids], axis=0)
    gp_evid_obsbinplus.loc[:, 'poids'] = poids
    print(gp_evid_obsbinplus)
    dict_poids = gp_evid_obsbinplus['poids'].to_dict()
    print(dict_poids)
    obsbin_plus.loc[:, 'Poids_evt'] = obsbin_plus.apply(lambda row: dict_poids[row['EVID']], axis=1)
    # liste_evt = np.unique(obsbin_plus.EVID.values)
    # tous_lespoidsStdM = []
    # for evid in liste_evt:
    #     # Compute the weight linked to StdM for each event
    #     poids = 1/(obsbin_plus[obsbin_plus.EVID==evid]['StdM'].values[0]**2)
    #     max_poids = 1/(0.1**2)
    #     poids = np.min([max_poids, poids])
    #     tous_lespoidsStdM.append(poids)
    #     obsbin_plus.loc[obsbin_plus.EVID==evid, 'Poids_evt'] = poids
    # # Normalize the event weight   
    # obsbin_plus.loc[:, 'Poids_evt_norm'] = obsbin_plus.loc[:, 'Poids_evt']/sum(tous_lespoidsStdM)
    return obsbin_plus

def weight_IStdI_evtStdM(obsbin_plus):
    """
    Attribute a weight to each binned intensity data that combine a weight based
    on the intensity associated standard deviation and a event-weight based on
    magnitude standard deviation (StdM). In one event, the data
    have a weight based on the intensity standard deviation StdI.
    In practice, the weight_IStdI weighting scheme is applied, then the weights
    are normalized within each event: after normalization, the sum of the weights
    based on StdI within one event is equal to 1. Then the StdM-event weight is
    applied.

    Parameters
    ----------
    obsbin_plus : pandas.DataFrame
        Dataframe with the binned intensity data for all calibration earthquakes.
        This dataframe should at least have the following columns:
            - "EVID": contains the ID of each event,
            - "StdI": the standard deviation associated to each binned intensity
            data,
            - "StdM": the standard deviation associated to the magnitude of each
            event.

    Returns
    -------
    obsbin_plus : pandas.DataFrame
        Completed input dataframe, with the following additional columns:
            - 'Poids_evt': the uniform weight of the different event, equal to
            1/n, where n is the number of event in the input dataframe.
            - 'Poids_int_norm': the normalized by event weights associated to StdI,
            - 'Poids': the combined weight
            - 'eqStd': the equivalent standard deviation corresponding to Poids,
            used in all inversion functions.

    """
    
    obsbin_plus = weight_IStdI(obsbin_plus)
    liste_evt = np.unique(obsbin_plus.EVID.values)
    for evid in liste_evt:
        # Normalized by event the intensity weight
        obsbin_plus = normaliser_poids_par_evt(evid, obsbin_plus)
        
    obsbin_plus = weight_evtStdM(obsbin_plus)
    # Combine the intensity weight and the event weight
    obsbin_plus.loc[:, 'Poids'] = obsbin_plus.loc[:, 'Poids_int_norm'].astype(float)*obsbin_plus.loc[:, 'Poids_evt']   
    # Compute equivalent Std for the inversion process
    obsbin_plus.loc[:, 'eqStd'] = np.sqrt(1/obsbin_plus.loc[:, 'Poids'])
    return obsbin_plus


def weight_evtStdH(obsbin_plus):
    """
    Attribute a weight based on the hypocentral depth uncertainty to each event.
    Depth uncertainty is characterized here by a minimal (shallow) and a maximal
    (deep) possible depth. 
    Weight = 1/((Hmax-Hmin)/2)**2

    Parameters
    ----------
    obsbin_plus : pandas.DataFrame
        Dataframe with the binned intensity data for all calibration earthquakes.
        This dataframe should at least have the following columns:
            - "EVID": the ID of each event,
            - "Hmin": minimal (shallow) possible depth,
            - "Hmax": maximal (shallow) possible depth,

    Returns
    -------
    Completed input dataframe, with an additional column 'Poids_evt', which 
    contains the weight based on the hypocentral depth uncertainty to each event.

    """
    gp_evid_obsbinplus = obsbin_plus[['EVID', 'Hmax', 'Hmin']].groupby('EVID').mean()
    StdH = (gp_evid_obsbinplus.Hmax-gp_evid_obsbinplus.Hmin)/2
    poids = 1/(StdH**2)
    gp_evid_obsbinplus.loc[:, 'poids'] = poids
    dict_poids = gp_evid_obsbinplus['poids'].to_dict()
    obsbin_plus.loc[:, 'Poids_evt'] = obsbin_plus.apply(lambda row: dict_poids[row['EVID']], axis=1)
    return obsbin_plus


def weight_IStdI_evtStdH(obsbin_plus):
    """
    Function that attribute a weight based on instrumental depth uncertainties.
    
    :param obsbin_plus: dataframe with the binned intensity data for all
                        calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 
                            - EVID: id of the considered earthquake
                            - Hmax: upper (deep) limit of the depth uncertainties
                            - Hmin: lower (shallow) limit of the depth uncertainties
                            - StdI: the standard deviation associated to the binned intensity data
    :type obsbin_plus: pandas.DataFrame                    
    :return: a completed obsbin_plus DataFrame, with a eqStd column that contains
             the inverse of the root square of a weight based on instrumental depth uncertainties. 
    """
    obsbin_plus = weight_IStdI(obsbin_plus)
    liste_evt = np.unique(obsbin_plus.EVID.values)
    for evid in liste_evt:
        # Normalized by event the intensity weight
        obsbin_plus = normaliser_poids_par_evt(evid, obsbin_plus)
    obsbin_plus = weight_evtStdH(obsbin_plus)
    obsbin_plus.loc[:, 'Poids'] = obsbin_plus.loc[:, 'Poids_int_norm'].astype(float)*obsbin_plus.loc[:, 'Poids_evt']
    obsbin_plus.loc[:, 'eqStd'] = 1/np.sqrt(obsbin_plus.loc[:, 'Poids'])
    #obsbin_plus.drop(['Poids_inevt_norm', 'Poids'], axis=1, inplace=True)
    return obsbin_plus


def normaliser_par_MClass(obsbin_plus, bin_width=0.5):
    """
    Function which normalizes weights of each event within a given area. Sum
    of the weights within one region is equal to one.
    
    :param regid: id of the region for which weights should be normalized
    :param obsbin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 
                            - I:  the binned intensity,
                            - StdI: the associated standard deviation,
                            - EVID: the earthquake ID,
                            - RegID: the region ID in which the earthquake
                            is located
    :type regid: str or float
    :type obsbin_plus: pandas.DataFrame                    
    :return: a completed obsbin_plus DataFrame, with a Poids_inreg_norm column that contains
             normalized weigths per region. 
    """
    minMag = obsbin_plus.Mag.min()
    maxMag = obsbin_plus.Mag.max()
    mag_bins = np.arange(minMag, maxMag+bin_width, bin_width)
    obsbin_plus.loc[:,'range1'] = pd.cut(obsbin_plus.Mag, mag_bins, include_lowest=True)
    mag_bins_df = np.unique(obsbin_plus.range1.values)
    for bins in mag_bins_df:
        ind = obsbin_plus.range1==bins
        obsbin_gp_tmp = obsbin_plus.loc[ind, :].groupby('EVID').mean()
        poids = obsbin_gp_tmp.Poids_evt.values/obsbin_gp_tmp.Poids_evt.sum()
        obsbin_gp_tmp.loc[:, 'poids_evt_norm'] = poids
        dict_poids = obsbin_gp_tmp['poids_evt_norm'].to_dict()
        obsbin_plus.loc[ind, 'Poids_evt_norm'] = obsbin_plus.loc[ind, :].apply(lambda row: dict_poids[row['EVID']], axis=1)
        obsbin_plus.loc[ind, 'poids_grp'] = obsbin_plus.loc[ind, 'Poids_int_norm']*obsbin_plus.loc[ind, 'Poids_evt_norm']
        obsbin_plus.loc[ind, 'poids_grp'] = obsbin_plus.loc[ind, 'poids_grp']/obsbin_plus.loc[ind, 'poids_grp'].sum()
    obsbin_plus.loc[:, 'Poids'] = obsbin_plus.loc[:, 'poids_grp']
    obsbin_plus.drop(['range1'], axis=1, inplace=True)
    return obsbin_plus, len(mag_bins_df)


def weight_IStdI_evtStdM_gMclass(obsbin_plus):
    """
    Attribute a weight that balance the effect of the different magnitude classes.
    A calibration dataset is composed by events with different magnitude. Often,
    events with small and great magnitudes are underrepresented in comparison
    to medium magnitude events. For example, in France, the number of calibration
    earthquake with magnitude greater than 5 is small. This weighting scheme
    aims to balance the contribution of the different magnitude classes of the
    calibration dataset. The magnitude bin width is equal to 0.5 and will begin
    with the smallest magnitude of the calibration dataset.
    Weight of the intensity bins, associated to the intensity uncertainty within
    each event and weight of the event, associated to the magnitude uncertainty
    within each magnitude class are propagated and included in the final weight.

    Parameters
    ----------
    obsbin_plus : pandas.DataFrame
        Dataframe with the binned intensity data for all calibration earthquakes.
        This dataframe should at least have the following columns:
            - "EVID": the ID of each event,
            - "Mag": magnitude of the event,
            - "StdM": uncertainty associated to the magnitude,
            - "StdI": uncertainty associated to the intensity bin,
            

    Returns
    -------
    obsbin_plus : pandas.DataFrame
        completed obsbin_plus DataFrame, with a eqStd column that contains
        the weight that balance the effect of the different magnitude classes.

    """
    # Attribution poids evt-StdM    
    obsbin_plus = weight_IStdI_evtStdM(obsbin_plus)
    # Normalisation des StdM par bin de magnitude
    obsbin_plus, nbre_bin = normaliser_par_MClass(obsbin_plus, bin_width=0.5)
    # Attribuer un poids uniforme par classe de magnitude
    obsbin_plus.loc[:, 'Poids'] = obsbin_plus.loc[:, 'Poids']/nbre_bin
    obsbin_plus.loc[:, 'eqStd'] = np.sqrt(1/obsbin_plus.Poids.values)
    #obsbin_plus.drop(['range1'], axis=1, inplace=True)
    return obsbin_plus
    

def lecture_fichierregions(fichier_regions):
    """
    Function that read the .txt file with the region limits
    
    :param fichier_regions: .txt file with the region limits. This file should have three columns
            'ID_region', 'Lon', 'Lat'. Each region is delimited by a polygon.
            Lon and Lat columns described the longitude and latitude of each
            point of the polygon. The ID_region column indicate which region is
            described by the corresponding Lon and Lat.
    :type fichier_regions: str
    
    :return: a dictionnary in which each key represent a region and the associated
             object in a matplotlib polygon object (matplotlib.path.Path())
    """
#    if os.path.isflile(fichier_regions):
    data_regions = pd.read_csv(fichier_regions, sep=';')
    regions = np.unique(data_regions.ID_region.values)
    dict_regions = {}
    for regid in regions:
        coord = data_regions[data_regions.ID_region==regid][['Lon', 'Lat']]
        dict_regions[regid] = mpltPath.Path(coord.to_numpy())
#    else:
#        raise FileNotFoundError('File does not exist: ' + fichier_regions)
    return dict_regions
    
    
def attribute_region(data_evt, obsbin_plus, fichier_regions):
    """
    Function that attribute a region ID to each line of the obsbin_plus input parameter.
    
    
    :param data_evt: dataframe with the following columns: 'EVID', the earthquake ID,
                     'Lon' and 'Lat' the longitude and the latitude of the earthquake epicenter.
    :param obsbin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns :
                            - EVID: the id of the earthquake associated
                        to each intensity data.
    :param fichier_regions: .txt file with the region limits. This file should have three columns
            'ID_region', 'Lon', 'Lat'. Each region is delimited by a polygon.
            Lon and Lat columns described the longitude and latitude of each
            point of the polygon. The ID_region column indicate which region is
            described by the corresponding Lon and Lat.
    :type data_evt: pandas.DataFrame
    :type obsbin_plus: pandas.DataFrame
    :type fichier_regions: str
    
    :return: a completed obsbin_plus DataFrame, with a RegID column indicating
             in which region occured the earthquake
    """
    dict_regions = lecture_fichierregions(fichier_regions)
    for regid in dict_regions.keys():
        data_evt.loc[:, 'in_reg'] = data_evt.apply(lambda row: dict_regions[regid].contains_points([[row['Lon'], row['Lat']]])[0], axis=1)
        #ind_in = data_evt[data_evt['in_reg']==True].index
        data_evt.loc[data_evt['in_reg']==True, 'RegID'] = regid
        dict_evtreg = data_evt.set_index('EVID')['RegID'].to_dict()
    obsbin_plus.loc[:, 'RegID'] = obsbin_plus.apply(lambda row: dict_evtreg[row['EVID']], axis=1)
    obsbin_plus['RegID'].fillna(-99, inplace=True)
    return obsbin_plus

def normaliser_par_region(regid, obsbin_plus):
    """
    Function which normalizes weights associated to the event within a given area.
    Sum of the weights within one region is equal to one.
    
    :param regid: id of the region for which weights should be normalized
    :param obsbin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 
                            - I:  the binned intensity,
                            - StdI: the associated standard deviation,
                            - EVID: the earthquake ID,
                            - RegID: the region ID in which the earthquake
                            is located
    :type regid: str or float
    :type obsbin_plus: pandas.DataFrame                    
    :return: a completed obsbin_plus DataFrame, with a Poids_inreg_norm column that contains
             normalized weigths per region. 
    """
    
    ind = obsbin_plus[obsbin_plus.RegID==regid].index
    somme_poids_par_reg= np.sum(obsbin_plus.loc[ind, 'Poids_int_norm'])
    obsbin_plus.loc[ind, 'Poids_inreg_norm'] = obsbin_plus.loc[ind, 'Poids_int_norm']/somme_poids_par_reg
    return obsbin_plus

def weight_IStdI_evtStdM_gRegion(obsbin_plus):
    """
    Function that attribute an equal weight to each region.
    
    :param obsbin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns :
                            - I:  the binned intensity,
                            - StdI: the associated standard deviation,
                            - EVID: the earthquake ID,
                            - RegID: the region ID in which the earthquake
                            is located
    :type obsbin_plus: pandas.DataFrame                    
    :return: a completed obsbin_plus DataFrame, with a eqStd column that contains
             the inverse of the root square of the normalized weigths per region. 
    """
    regions = np.unique(obsbin_plus.RegID.values)
    poids_reg = 1/len(regions)
    # Attribution poids evt-StdM    
    obsbin_plus = weight_IStdI_evtStdM(obsbin_plus)
    for regid in regions:
        ind = obsbin_plus[obsbin_plus.RegID==regid].index
        obsbin_plus =  normaliser_par_region(regid, obsbin_plus)
        obsbin_plus.loc[ind, 'Poids'] = obsbin_plus.loc[ind, 'Poids_inreg_norm']*poids_reg
    obsbin_plus.loc[:, 'eqStd'] = 1/np.sqrt(obsbin_plus.loc[:, 'Poids'])
    #obsbin_plus.loc[:, 'eqStd'] = obsbin_plus.loc[:, 'eqStd']/obsbin_plus['eqStd'].max()
    #obsbin_plus.drop(['Poids_int_norm', 'Poids_inreg_norm', 'Poids'], axis=1, inplace=True)
    return obsbin_plus

def savename_weights(option_ponderation):
    """
    Function that gives the savename id of the chosen wieghting scheme
    :param option_ponderation: type of weighting whished. Possible values:
                               'Ponderation dI', 'Ponderation evt-uniforme',
                               'Ponderation evt-reg', 'Ponderation evt-depth'
    :type option_ponderation: str
    :return: a str with the savename id. For 'Ponderation dI', 'wStdI',
             for 'Ponderation evt-uniforme', 'wdevt-uni', for'Ponderation evt-reg',
             'wdevt-reg', and for 'Ponderation evt-depth', 'wdevt-depth'.
             
    """
    return 'w' + option_ponderation


# def evt_weights_C1C2(obsbin_plus, option_ponderation):
#     """
#     Function that attribute a weight to the intensity data
#     :param obsbin_plus:dataframe with the binned intensity data for all calibration earthquakes.
#                         This dataframe should at least have have the following
#                         columns : 
#                             - Mag: the magnitude of the earthquake
#                             - StdM: the associated standard deviation,
#                             - EVID: the earthquake ID
#     :param option_ponderation: type of weighting whished. Possible values:
#                                 'Ponderation evt-uniforme',
#                                'Ponderation evt-stdM', 'Ponderation mag-class'
#     :type obsbin_plus: pandas.DataFrame
#     :type option_ponderation: str
    
#     :return: a completed obsbin_plus DataFrame, with a eqStdM column that contains
#              the associated inverse square root of the weights
#     """
#     if option_ponderation == 'Ponderation evt-uniforme':
#         obsbin_plus = C1C2_ponderation_evt_uniforme(obsbin_plus)
#     elif option_ponderation == 'Ponderation evt-stdM':
#         obsbin_plus = C1C2_ponderation_evt_sdtM(obsbin_plus)
#     elif option_ponderation == 'Ponderation mag_class':
#         obsbin_plus = C1C2_ponderation_mag_class(obsbin_plus)
#     else:
#         print('No such ponderation option:')
#         print(option_ponderation)
#     return obsbin_plus

def Kovatt_ponderation_evt_uniforme(obsbin_plus):
    """
    Function that attribute an equal weight to each earthquake.
    
    :param obsbin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 'EVID' and 'StdI', which are respectively id
                        of the chosen earthquake and the standard deviation
                        associated to the binned intensity data
    :type obsbin_plus: pandas.DataFrame                    
    :return: a completed obsbin_plus DataFrame, with a eqStd column that contains
             the inverse of the root square of the normalized weigths per earthquake. 
    """
    liste_evt = np.unique(obsbin_plus.EVID.values)
    poids_evt = 1/len(liste_evt)
    for evid in liste_evt:
        obsbin_plus = normaliser_poids_par_evt(evid, obsbin_plus)
    obsbin_plus.loc[:, 'Poids'] = obsbin_plus.loc[:, 'Poids_inevt_norm'].astype(float)*poids_evt
    obsbin_plus.loc[:, 'eqStd'] = 1/np.sqrt(obsbin_plus.loc[:, 'Poids'])
    
    obsbin_plus.drop(['Poids_inevt_norm', 'Poids'], axis=1, inplace=True)
    return obsbin_plus