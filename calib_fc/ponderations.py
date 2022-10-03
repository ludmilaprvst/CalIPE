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
                        This dataframe should at least have have the following
                        columns : 'I', 'StdI', 'EVID' and 'RegID'
                        which are respectively the binned intensity, the associated standard deviation,
                        the earthquake ID and the region ID in which the earthquake
                        is located.
    :param option_ponderation: type of weighting whished. Possible values:
                               'Ponderation dI', 'Ponderation evt-uniforme',
                               'Ponderation evt-reg', 'Ponderation evt-depth'
    :type obsbin_plus: pandas.DataFrame
    :type option_ponderation: str
    
    :return: a completed obsbin_plus DataFrame, with a eqStd column that contains
             the associated inverse square root of the weights
    """
    if option_ponderation == 'Ponderation dI':
        obsbin_plus = Kovatt_ponderation_dI(obsbin_plus)
    elif option_ponderation == 'Ponderation evt-uniforme':
        obsbin_plus = Kovatt_ponderation_evt_uniforme(obsbin_plus)
    elif option_ponderation == 'Ponderation evt-reg':
        obsbin_plus = Kovatt_ponderation_evt_reg(obsbin_plus)
    elif option_ponderation == 'Ponderation evt-depth':
        obsbin_plus = Kovatt_ponderation_evt_depth(obsbin_plus)
    else:
        print('No such ponderation option:')
        print(option_ponderation)
    return obsbin_plus


def evt_weights_C1C2(obsbin_plus, option_ponderation):
    """
    Function that attribute a weight to the intensity data
    :param obsbin_plus:dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 'Mag', 'StdM' and 'EVID' 
                        which are respectively the binned intensity, the associated standard deviation,
                        the earthquake ID and the region ID in which the earthquake
                        is located.
    :param option_ponderation: type of weighting whished. Possible values:
                                'Ponderation evt-uniforme',
                               'Ponderation evt-stdM', 'Ponderation mag-class'
    :type obsbin_plus: pandas.DataFrame
    :type option_ponderation: str
    
    :return: a completed obsbin_plus DataFrame, with a eqStdM column that contains
             the associated inverse square root of the weights
    """
    if option_ponderation == 'Ponderation evt-uniforme':
        obsbin_plus = C1C2_ponderation_evt_uniforme(obsbin_plus)
    elif option_ponderation == 'Ponderation evt-stdM':
        obsbin_plus = C1C2_ponderation_evt_sdtM(obsbin_plus)
    elif option_ponderation == 'Ponderation mag_class':
        obsbin_plus = C1C2_ponderation_mag_class(obsbin_plus)
    else:
        print('No such ponderation option:')
        print(option_ponderation)
    return obsbin_plus

def C1C2_ponderation_evt_uniforme(obsbin_plus):
    """
    eqStdM column will be meaned by EVID before being used for C1/C2 inversion
    """
    liste_evt = np.unique(obsbin_plus.EVID.values)
    poids = 1/len(liste_evt)
    eqStdM = np.sqrt(1/poids)
    obsbin_plus.loc[:, 'eqStdM'] = eqStdM
    return obsbin_plus

def C1C2_ponderation_evt_sdtM(obsbin_plus):
    """
    eqStdM column will be meaned by EVID before being used for C1/C2 inversion:
        only one data per earthquake is used for C1/C2 inversion
    """
    liste_evt = np.unique(obsbin_plus.EVID.values)
    liste_poidsM = []
    for evid in liste_evt:
        poids = 1/(obsbin_plus[obsbin_plus.EVID==evid]['StdM'].values[0]**2)
        min_poids = 1/0.1
        poids = np.max([min_poids, poids])
        #eqStdM = np.sqrt(1/poids)
        #print(evid, poids,eqStdM )
        obsbin_plus.loc[obsbin_plus.EVID==evid, 'poids'] = poids
        liste_poidsM.append(poids)
        
    obsbin_plus.loc[:, 'poids'] = obsbin_plus.loc[:, 'poids']/np.sum(liste_poidsM)
    obsbin_plus.loc[:, 'eqStdM'] = np.sqrt(1/obsbin_plus.loc[:, 'poids'])
    return obsbin_plus

def C1C2_ponderation_mag_class(obsbin_plus):
    """
    eqStdM column will be meaned by EVID before being used for C1/C2 inversion:
        only one data per earthquake is used for C1/C2 inversion
    Bin of 0.5 magnitude unit width
    """
    liste_evt = np.unique(obsbin_plus.EVID.values)
    for evid in liste_evt:
        poids = 1/(obsbin_plus[obsbin_plus.EVID==evid]['StdM'].values[0]**2)
        min_poids = 1/0.1
        poids = np.max([min_poids, poids])
        
        obsbin_plus.loc[obsbin_plus.EVID==evid, 'poids_indiv'] = poids
    minMag = obsbin_plus.Mag.min()
    maxMag = obsbin_plus.Mag.max()
    mag_bins = np.arange(minMag, maxMag+0.5, 0.5)
    obsbin_plus.loc[:,'range1'] = pd.cut(obsbin_plus.Mag, mag_bins, include_lowest=True)
    mag_bins_df = np.unique( obsbin_plus.range1.values)
    # Normalisation des StdM par bin de magnitude
    liste_poids_class = []
    for bins in mag_bins_df:
        ind = obsbin_plus.range1==bins
        obsbin_gp_tmp = obsbin_plus.loc[ind, :].groupby('EVID').mean()
        obsbin_plus.loc[ind, 'poids_class'] = obsbin_plus.loc[ind, 'poids_indiv']/obsbin_gp_tmp.poids_indiv.sum()
        obsbin_gp_tmp = obsbin_plus.loc[ind, :].groupby('EVID').mean()
        #poids_class = obsbin_plus[ind]['poids_class'].values[0]
        liste_poids_class.append(obsbin_gp_tmp['poids_class'].sum())
        
    obsbin_plus.loc[:, 'poids'] = obsbin_plus.loc[:, 'poids_class']/np.sum(liste_poids_class)
    obsbin_plus.loc[:, 'eqStdM'] = np.sqrt(1/obsbin_plus.poids.values)
    obsbin_plus.drop(['poids_indiv', 'poids_class', 'poids', 'range1'], axis=1, inplace=True)  
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
    if option_ponderation == 'Ponderation dI':
        name = 'wStdI'
    elif option_ponderation == 'Ponderation evt-uniforme':
        name = 'wdevt-uni'
    elif option_ponderation == 'Ponderation evt-reg':
        name = 'wdevt-reg'
    elif option_ponderation == 'Ponderation evt-depth':
        name = 'wdevt-depth'
    else:
        print('No such ponderation option:')
        print(option_ponderation)
    return name

def Kovatt_ponderation_dI(obsbin_plus):
    """
    Function that compute a weight based on intensity standard deviation
    :param obsbin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 'I' and 'StdI', which are respectively the binned
                        intensity and the associated standard deviation.
    :type obsbin_plus: pandas.DataFrame
    
    :return: a completed obsbin_plus DataFrame, with a eqStd column that contains
             the associated inverse square root of the weights
    """
    obsbin_plus.loc[:, 'eqStd'] = obsbin_plus.loc[:, 'StdI']
    return obsbin_plus

def normaliser_poids_par_evt(evid, obsbin_plus):
    """
    Function which normalizes weights of each binned intensity within one event
    Weight of one binned intensity is equal to the inverse of the square of
    the standard deviation associated to the intensity bin.
    
    :param evid: id of the earthquake for which weights should be normalized
    :param obsbin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 'EVID' and 'StdI', which are respectively id
                        of the chosen earthquake and the standard deviation
                        associated to the binned intensity data
    :type evid: str or float
    :type obsbin_plus: pandas.DataFrame
    :return: a completed obsbin_plus DataFrame, with a Poids_inevt column that contains
             normalized weigths per earthquake. Sum of the weights of one earthquake
             data is be equal to one.
    """
    ind = obsbin_plus[obsbin_plus.EVID==evid].index
    obsbin_plus.loc[ind, 'Poids_inevt'] = 1/obsbin_plus.loc[ind, 'StdI']**2
    somme_poids_par_evt = np.sum(obsbin_plus.loc[ind, 'Poids_inevt'])
    obsbin_plus.loc[ind, 'Poids_inevt_norm'] = obsbin_plus.loc[ind, 'Poids_inevt']/somme_poids_par_evt
    obsbin_plus.drop(['Poids_inevt'], axis=1, inplace=True)
    return obsbin_plus

def normaliser_par_region(regid, obsbin_plus):
    """
    Function which normalizes weights of each event within a given area. Sum
    of the weights within one region is equal to one.
    
    :param regid: id of the region for which weights should be normalized
    :param obsbin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 'EVID' and 'StdI', which are respectively id
                        of the chosen earthquake and the standard deviation
                        associated to the binned intensity data
    :type regid: str or float
    :type obsbin_plus: pandas.DataFrame                    
    :return: a completed obsbin_plus DataFrame, with a Poids_inreg_norm column that contains
             normalized weigths per region. 
    """
    
    ind = obsbin_plus[obsbin_plus.RegID==regid].index
    somme_poids_par_reg= np.sum(obsbin_plus.loc[ind, 'Poids_inevt_norm'])
    obsbin_plus.loc[ind, 'Poids_inreg_norm'] = obsbin_plus.loc[ind, 'Poids_inevt_norm']/somme_poids_par_reg
    return obsbin_plus

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
    obsbin_plus.loc[:, 'Poids'] = obsbin_plus.loc[:, 'Poids_inevt_norm']*poids_evt
    obsbin_plus.loc[:, 'eqStd'] = 1/np.sqrt(obsbin_plus.loc[:, 'Poids'])
    
    obsbin_plus.drop(['Poids_inevt_norm', 'Poids'], axis=1, inplace=True)
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
                        columns : 'EVID' which is the id of the earthquake associated
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


def Kovatt_ponderation_evt_reg(obsbin_plus):
    """
    Function that attribute an equal weight to each region.
    
    :param obsbin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 'EVID', 'RegID' and 'StdI', which are respectively id
                        of the chosen earthquake, the ID of the region within the
                        earthquake is located and the standard deviation
                        associated to the binned intensity data.
    :type obsbin_plus: pandas.DataFrame                    
    :return: a completed obsbin_plus DataFrame, with a eqStd column that contains
             the inverse of the root square of the normalized weigths per region. 
    """
    regions = np.unique(obsbin_plus.RegID.values)
    poids_reg = 1/len(regions)
    liste_evt = np.unique(obsbin_plus.EVID.values)
    for evid in liste_evt:
        obsbin_plus = normaliser_poids_par_evt(evid, obsbin_plus)
    for regid in regions:
        ind = obsbin_plus[obsbin_plus.RegID==regid].index
        obsbin_plus =  normaliser_par_region(regid, obsbin_plus)
        obsbin_plus.loc[ind, 'Poids'] = obsbin_plus.loc[ind, 'Poids_inreg_norm']*poids_reg
    obsbin_plus.loc[:, 'eqStd'] = 1/np.sqrt(obsbin_plus.loc[:, 'Poids'])
    #obsbin_plus.loc[:, 'eqStd'] = obsbin_plus.loc[:, 'eqStd']/obsbin_plus['eqStd'].max()
    obsbin_plus.drop(['Poids_inevt_norm', 'Poids_inreg_norm', 'Poids'], axis=1, inplace=True)
    return obsbin_plus

def Kovatt_ponderation_evt_depth(obsbin_plus):
    """
    Function that attribute a weight based on instrumental depth uncertainties.
    
    :param obsbin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 'EVID', 'Hmax', 'Hmin' and 'StdI', which are
                        respectively id of the chosen earthquake, upper limit of
                        the depth uncertainties, lower limit of the depth uncertainties
                        and the standard deviation associated to the binned intensity data.
    :type obsbin_plus: pandas.DataFrame                    
    :return: a completed obsbin_plus DataFrame, with a eqStd column that contains
             the inverse of the root square of a weight based on instrumental depth uncertainties. 
    """
    liste_evt = np.unique(obsbin_plus.EVID.values)
    poids_evt = 1/len(liste_evt)
    for evid in liste_evt:
        obsbin_plus = normaliser_poids_par_evt(evid, obsbin_plus)
    obsbin_plus.loc[:, 'Poids'] = obsbin_plus.loc[:, 'Poids_inevt_norm']*poids_evt
    obsbin_plus.loc[:, 'Poids'] = obsbin_plus.loc[:, 'Poids']/(obsbin_plus.loc[:, 'Hmax']-obsbin_plus.loc[:, 'Hmin'])
    obsbin_plus.loc[:, 'eqStd'] = 1/np.sqrt(obsbin_plus.loc[:, 'Poids'])
    obsbin_plus.drop(['Poids_inevt_norm', 'Poids'], axis=1, inplace=True)
    return obsbin_plus
