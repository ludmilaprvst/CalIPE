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
    intersect = np.intersect1d(array1, array2)
    if (len(intersect) == len(array1))and(len(intersect) == len(array2)):
        return True
    else:
        return False

def create_basicdb_criteria(evtdata_name, obsdata_name,
                            binning_type='RAVG',
                            outputfolder='',
                            regiondata_name='',
                            ponderation='Ponderation evt-uniforme',
                            ):
    evtdata = pd.read_csv(evtdata_name, sep=';')
    obsbin_plus = prepare_input4calibration(obsdata_name, evtdata_name, ponderation,
                                            regiondata_name, binning_type)
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
    criteria.to_excel(outputfolder + '/' + savename + '.xlsx')
    return criteria


def create_liste_subset(criteria, year_inf, year_sup,
                        QH, NClass, Nobs, Dc):
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
    bootstrap_list = []
    bootstrap_criteres = pd.DataFrame(columns=['Deleted event id'])
    for ind, evid in enumerate(liste_base):
        tmp_list = np.setdiff1d(liste_base, [evid])
        bootstrap_list.append(tmp_list)
        bootstrap_criteres.loc[ind, :] = [evid]
    return bootstrap_list, bootstrap_criteres

def create_subsets(global_liste, criteres, evtdata_name, folder='', basename='',
                   column_criteria=['Year_inf', 'Year_sup', 'NClass', 'QH', 'Ndata', 'Dc'] ):
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

