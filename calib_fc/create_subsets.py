# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:56:53 2021

@author: PROVOST-LUD
"""

import pandas as pd
import numpy as np
from prepa_data import prepare_input4calibration
import os


def same_values_2array(array1, array2):
    """
    Check if two arrays of the same length contains the same values

    Parameters
    ----------
    array1 : array
        first array to check.
    array2 : array
        second array to check.

    Returns
    -------
    bool
        is the two arrays contains the same values, return True. Else return False

    """
    intersect = np.intersect1d(array1, array2)
    if (len(intersect) == len(array1))and(len(intersect) == len(array2)):
        return True
    else:
        return False

def create_basicdb_criteria(evtdata_name, obsdata_name,
                            binning_type='ROBS',
                            outputfolder='',
                            regiondata_name='',
                            ponderation='Ponderation evt-uniforme',
                            ):
    """
    Complete the evt file with information about the number of intensity class (Nclass)
    after binning (to create isoseismal radii) by earthquake and the total number of 
    intensity data points (IDP) used to compute theisoseismal radii (Ndata) by earthquake. 

    Parameters
    ----------
    evtdata_name : str
        name of the evt data file. The list of the calibration earthquakes and associated 
        metadata are stored in this file. This .txt file contains 16 columns, separated by the ";" string:
            EVID: ID of the earthquake
            Year : year of occurence of the earthquake
            Month : month of occurence of the earthquake
            Day: day of occurence of the earthquake
            Lon: longitude in WGS84 of the epicenter
            Lat: latitude in WGS84 of the epicenter
            QPos: quality of the epicenter location
            I0 : epicentral intensity
            QI0 : quality of the epicentral intensity value
            Ic : intensity of completeness
            Dc : distance of completeness
            Mag: magnitude of the earthquake
            StdM : uncertainty associated with the magnitude
            Depth: hypocentral depth of the earthquake
            Hmin : lower bound of uncertainty associated to depth
            Hmax : upper bound of uncertainty associated to depth
    obsdata_name : str
        name of the obs data file. The IDPs of the calibration earthquakes are stored in this file.
        This .txt file contains 5 columns, separated by the ";" string:
            EVID : ID of the earthquake
            Lon: longitude in WGS84 of the IDP
            Lat: latitude in WGS84 of the IDP
            Iobs: value of intensity of the IDP
            QIobs : quality associated to Iobs
    binning_type : str, optional
        Name of the method applied to the calibration earthquakes intensity date to compute 
        isoseismal radii. The isoseismal radii are the intensity data used in the 
        inversion process Availaible values are : 'ROBS', 'RAVG', 'RP50' and 'RP84'. 
        The default is 'ROBS'.
    outputfolder : str, optional
        path to the outputfolder where the output dataframe will be saved in a excel file.
        The default is '' (file saved in the folder where the function is executed)
    regiondata_name : str, optional
        Name of the .txt file which contains the contour of the different regions
        defined by the user. Contour is described by a polygon. The coordinates of
        the polygon points are in WGS84 longitude and latitude.
        The three columns are separeted by ";":
            ID_region : ID of the considered region
            Lon: Longitude of the polygon points
            Lat: Latitude of the polygon points.
        The default is ''.
    ponderation : TYPE, optional
        Name of the ponderation applied to the calibration earthquakes intensity data used
        in the inversion process of the beta coefficient. The default is 'Ponderation evt-uniforme'.

    Returns
    -------
    criteria : pandas.dataframe
       dataframe with the same columns as the evt file with two supplementary columns
       NClass (number of isoseismal radii) per earthquake and Ndata (number of IDP used to
    compute the isoseismal radii) per earthquake.

    """
    evtdata = pd.read_csv(evtdata_name, sep=';')
    obsbin_plus = prepare_input4calibration(obsdata_name, evtdata_name, ponderation,
                                            regiondata_name, binning_type)
    print(obsbin_plus.head())
    obsbin_plus_gp_count = obsbin_plus[['EVID', 'I']].groupby('EVID').count()
    obsbin_plus_gp_count = obsbin_plus_gp_count.reset_index()
    obsbin_plus_gp_count.columns = ['EVID', 'NClass']
    count_ndataEff = obsbin_plus[['EVID', 'Ndata']].groupby('EVID').sum()
    count_ndataEff = count_ndataEff.reset_index()
    criteria = evtdata.merge(count_ndataEff, on='EVID')
    criteria = criteria.merge(obsbin_plus_gp_count, on='EVID')
    head, basename = os.path.split(evtdata_name)
    basename = basename[:-4]
    savename = "subset_criteria_" + basename
    criteria.to_excel(outputfolder + '/' + savename + '.xlsx', index=False)
    return criteria


def create_liste_subset(criteria, year_inf, year_sup,
                        QH, NClass, Nobs, Dc):
    """
    Create subsets from a calibration dataset (called hereafter basic calibration database)
    and the following criteria:
        year of occurence of the earthquake
        Quality of the depth estimate
        Number of intensity class of the isoseismal radii of the earthquake
        Number of data used to compute the isoseismal radii of the earthquake
        Distance of completeness of the earthquake's macroseismic field
        

    Parameters
    ----------
    criteria : pandas.DataFrame
        dataframe with metadata about the earthquakes of the basic calibration. The metadata required are:
            EVID : ID of the earthquake
            Year : year of occurence of the earthquake
            Dc : distance of completeness of the earthquake's macroseismic field
            NClass : number of intensity class of the isoseismal radii of the earthquake
            Ndata : number of data used to compute the isoseismal radii of the earthquake
            QH : quality of the depth estimate
    year_inf : list of int
        list of the minimal year of occurence allowed.
    year_sup : list int
        list of the maximal year of occurence allowed.
    QH : list
        List of list of the quality of the depth estimate in the metadata wished.
        Example : QH = [['A', 'B'], ['A', 'B', 'C']]
        The quality of the depth estimate in the metadata is based on expert opinion. 
        A is the best quality factor and E the worst. E quality means that we have no
        idea about the depth of the earthquake. In the example, the subsets created will contain
        earthquakes with 'A' and 'B' or 'A', 'B' and 'C' depth quality.    
    NClass : list of int
        list of the minimal number of intensity class allowed.
    Nobs : list of int
        list of the minimal number of data used to compute isoseismal radii allowed.
    Dc : list of float
        list of the minimal distance of completeness of the earthquake's macroseismic field
        allowed.

    Returns
    -------
    global_liste : list
        list of the different list of calibration earthquakes.
    criteres : pandas.DataFrame
        dataframe that traces the criteria used to create each of the subsets.

    """
    if not year_inf:
        min_year = criteria.Year.min()
        year_inf = [min_year]
    if not year_sup:
        max_year = criteria.Year.max()
        year_sup = [max_year]
    if not NClass:
        NClass = [2]
    if not QH:
        QH = [np.unique(criteria.QH.values)]
    if not Nobs:
        min_nobs = criteria.Ndata.min()
        Nobs = [min_nobs]
    if not Dc:
        min_Dc = criteria.Dc.min()
        Dc = [min_Dc]
    global_liste = []
    criteres = pd.DataFrame(columns=['Year_inf', 'Year_sup', 'NClass', 'QH', 'Ndata', 'Dc'])
    compt = 0
    for min_year in year_inf:
        for max_year in year_sup:
            for nclass in NClass:
                for qh in QH:
                    for ndata in Nobs:
                        for dc in Dc:
                            tmp = criteria[np.logical_and(criteria.Year>=min_year, criteria.Year<=max_year)]
                            tmp = tmp[np.logical_and(tmp.NClass>=nclass, tmp.QH.isin(qh))]
                            tmp = tmp[np.logical_and(tmp.Ndata>=ndata, tmp.Dc>=dc)]
                            liste_evid_tmp = tmp.EVID.values
                            global_liste.append(liste_evid_tmp)
                            qh_save = ','.join(qh)
                            criteres.loc[compt,:] = [min_year, max_year, nclass, qh_save, ndata, dc]
                            compt += 1
    return global_liste, criteres

def filter_by_nevt(global_liste, criteres, nmin=10):
    """
    Filter a list of calibration sets and keep only the calibration sets with 
    a minimal number of earthquakes.

    Parameters
    ----------
    global_liste : list
       list of the different list of calibration earthquakes, identified by their ID.
    criteres : pandas.DataFrame
        dataframe that traces the criteria used to create each of the subsets listed in global_list.
    nmin : int, optional
        minimal number of earthquake allowed in the calibration datasubsets. The default is 10.

    Returns
    -------
    filt_nevt_liste : list
        global_liste filtered from the datasets with a number of earthquakes smaller than nmin.
    filt_criteres : TYPE
        criteres filtered from the datasets with a number of earthquakes smaller than nmin.

    """
    filt_nevt_liste = []
    filt_criteres = pd.DataFrame(columns=criteres.columns)
    for ind, liste in enumerate(global_liste):
        if len(liste)>= nmin:
            filt_nevt_liste.append(liste)
            crit_to_append = pd.DataFrame(columns=criteres.columns)
            crit_to_append.loc[0, :] = criteres.loc[ind, :].values
            filt_criteres = filt_criteres.append(crit_to_append)
    filt_criteres.reset_index(inplace=True)
    filt_criteres = filt_criteres[criteres.columns]
    return filt_nevt_liste, filt_criteres

def check_duplicate(global_liste, criteres):
    """
    Check in a list of calibration datasets if two calibration datasets are identical.
    Duplicate calibration datasets are deleted.

    Parameters
    ----------
    global_liste : list
        list of the different list of calibration earthquakes, identified by their ID.
    criteres : pandas.DataFrame
        dataframe that traces the criteria used to create each of the subsets listed in global_list.

    Returns
    -------
    no_duplicate_list : list
        list of the different list of calibration earthquakes with no duplicated calibration datasets.
    no_duplicat_criteres : TYPE
        dataframe that traces the criteria used to create each of the subsets listed in no_duplicate_list.

    """
    no_duplicate_list = []
    no_duplicat_criteres = pd.DataFrame(columns=criteres.columns)
    for ind, liste in enumerate(global_liste):
        unique = True
        for liste_check in global_liste[ind+1:]:
            if same_values_2array(liste_check, liste):
                unique = False
                break
        if unique:
            no_duplicate_list.append(liste)
            crit_to_append = pd.DataFrame(columns=criteres.columns)
            crit_to_append.loc[0, :] = criteres.loc[ind, :].values
            no_duplicat_criteres = no_duplicat_criteres.append(crit_to_append)
        else:
            pass    
    no_duplicat_criteres.reset_index(inplace=True)
    no_duplicat_criteres = no_duplicat_criteres[criteres.columns]
    return no_duplicate_list, no_duplicat_criteres

def create_bootstrap_notrandom(liste_base):
    """
    Create subsets useable for a bootstrap study for a basic calibration dataset.
    The subsets are created by deleting one earthquake from the basic calibration dataset.
    Each earthquake of the basic calibration dataset will be deleted to create the subsets:
        for example, if the basic calibration dataset contains 10 earthquakes,
        the function will create 10 subsets, with in each subset one earthquake deleted from
        the calibration list.
    

    Parameters
    ----------
    liste_base : list
        list of calibration earthquake, identified by their ID.

    Returns
    -------
    bootstrap_list : list
        list of the different bootstrap susbets 
    bootstrap_criteres : pandas.DataFrame
        dataframe containing the deleted earthquake ID for each subset.

    """
    bootstrap_list = []
    bootstrap_criteres = pd.DataFrame(columns=['Deleted event id'])
    for ind, evid in enumerate(liste_base):
        tmp_list = np.setdiff1d(liste_base, [evid])
        bootstrap_list.append(tmp_list)
        bootstrap_criteres.loc[ind, :] = [evid]
    return bootstrap_list, bootstrap_criteres

def create_subsets(global_liste, criteres, evtdata_name, folder='', basename='',
                   column_criteria=['Year_inf', 'Year_sup', 'NClass', 'QH', 'Ndata', 'Dc'] ):
    """
    Create calibration Evt files from a list of calibration susbsets.  

    Parameters
    ----------
    global_liste : list
        list of list of earthquakes identified by their ID of calibration subsets.
    criteres : pandas.DataFrame
        dataframe with the criteria used to define the subsets. Each line corresponds to
        one subset. The order is the same as the list of subsets of global_liste.
    evtdata_name : str
        Name of the Evt file of the basic calibration dataset. The subsets are created from
        the basic dataset.
    folder : str, optional
        Name of the folder where the subset files will be saved. The default is ''.
    basename : str, optional
        basename of the basic calibration dataset. Used to name the folder where the
        subset files will be saved. The default is ''. In this case, the evtdata_name will
        be used.
    column_criteria : list, optional
        List of the columns of the criteria used to define the subsets from the
        basic calibration database. The default is ['Year_inf', 'Year_sup', 'NClass', 'QH', 'Ndata', 'Dc'].

    Returns
    -------
    None.

    """
    evtdata = pd.read_csv(evtdata_name, sep=';')
    nombre_subset = len(global_liste)
    len_nbresubset = len(str(nombre_subset))
    if basename == '':
        head, basename = os.path.split(evtdata_name)
        basename = basename[:-4]
    column_criteria.insert(0, 'Datasubsetname')
    datasubset_table = pd.DataFrame(columns=column_criteria)
    for ind, liste in enumerate(global_liste):
        subset_name = "Datasubset"+str(ind+1).zfill(len_nbresubset)
        sub_evtdata = evtdata[evtdata.EVID.isin(liste)]
        subset_folder = folder + '/' + basename
        if not os.path.exists(subset_folder):
            os.makedirs(subset_folder)
        sub_evtdata.to_csv(subset_folder+'/'+subset_name+'.csv', sep=';', index=False)
        if len(column_criteria)-1>1:
            datasubset_table.loc[ind, :] = np.insert(criteres.loc[ind,:].values, 0, subset_name)
        else:
            datasubset_table.loc[ind, :] = [subset_name, criteres.loc[ind,:].values[0]]
    datasubset_table.to_excel(subset_folder+'/'+'dataset_list.xlsx', index=False)
    return

