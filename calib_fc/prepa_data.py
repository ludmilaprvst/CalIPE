# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:52:50 2021

@author: PROVOST-LUD
"""

import pandas as pd
import numpy as np
import EvtIntensityObject as eio
from ponderations import evt_weights, attribute_region

class fichier_input:
    """
    Class used to create an object input to the class Evt in EvtIntensityObject.py
    """
    def __init__(self, obsdata, evtdata):
        """
        Initialization and creation of the object used to initialize the Evt object

        Parameters
        ----------
        obsdata : pandas.DataFrame
            obsdata contains the macroseismic fields corresponding to the earthquakes stored in evtdata.
        evtdata : pandas.DataFrame
            evtdata contains the metadata of different earthquakes. 
            Mandatory columns for evtdata:
                - EVID: ID of the earthquake
                - Lon: longitude in WGS84 of the earthquake location
                - Lat: latitude in WGS84 of the earthquake location
                - Qpos: quality associated to the earthquake location (A for very good quality, E for bad quality, i.e. more than 50 km of possible error)
                - I0: epicentral intensity of the earthquake 
                - QI0: quality associated to the value of I0 (A to E)
                - Year: year of occurence of the earthquake
                - Month: month of occurence of the earthquake
                - Day: day of occurence of the earthquake
                - Ic: intensity of completeness of the earthquake. The intensities smaller than Ic in a macroseismic field are considered as incomplete
                     In the isoseismal radii based on intensity bins, intensities smaller than Ic are not taken into account
                     to compute the isoseismal radii.
            Optional columns for EvtFile:
                - Dc: distance of completeness of the earthquake. The macroseismic field located at greater epicentral distance than Dc is considered as incomplete
                - Depth: hypocentral depth of the earthquake
                - Hmin: lower bound of depth uncertainty
                - Hmax: upper bound of depth uncertainty
                - Mag: magnitude of the earthquake
                - StdM: uncertainty associated to magnitude
            Mandatory columns are mandatory to use the Evt class. However, to use the other functions
            of CalIPE, the optional columns are mandatory.
            Mandatory columns for the Obs file:
                - EVID: ID of the earthquake
                - Iobs: intenstity in the locality (coordinates Lon/Lat)
                - QIobs: quality of the value of Iobs. Possible values: A (very good quality), B (fair quality) and C (bad quality)
                - Lon: Longitude in WGS84 of the locality
                - Lat: Latitude in WGS84 of the locality

        Returns
        -------
        None.

        """
        self.EvtFile = evtdata
        self.ObsFile = obsdata

def ensure_floatcoltype(obsbin_plus):
    obsbin_plus['StdM'] = obsbin_plus['StdM'].astype(float)
    obsbin_plus['I'] = obsbin_plus['I'].astype(float)
    obsbin_plus['StdI'] = obsbin_plus['StdI'].astype(float)
    obsbin_plus['Io'] = obsbin_plus['Io'].astype(float)
    obsbin_plus['Io_std'] = obsbin_plus['Io_std'].astype(float)
    obsbin_plus['Depi'] = obsbin_plus['Depi'].astype(float)
    obsbin_plus['Ndata'] = obsbin_plus['Ndata'].astype(int)
    obsbin_plus['Mag'] = obsbin_plus['Mag'].astype(float)
    return obsbin_plus

def prepare_input4calibration(obsdata_name, evtdata_name, ponderation,
                              regiondata_name='',
                              binning_type = 'ROBS',
                              Beta=-3.5,
                              Gamma=0):
    """
    Prepare the data stored in evt file and obs file into a dataframe useable by the CalIPE
    tool.

    Parameters
    ----------
    obsdata_name : str
        Name of the obs file that contains the macroseismic fields of the calibration
        earthquakes.
        Mandatory columns for the Obs file:
            - EVID: ID of the earthquake
            - Iobs: intenstity in the locality (coordinates Lon/Lat)
            - QIobs: quality of the value of Iobs. Possible values: A (very good quality), B (fair quality) and C (bad quality)
            - Lon: Longitude in WGS84 of the locality
            - Lat: Latitude in WGS84 of the locality
    evtdata_name : str
        Name of the evt file that contains the metadata of different earthquakes. 
        Mandatory columns for evtdata:
            - EVID: ID of the earthquake
            - Lon: longitude in WGS84 of the earthquake location
            - Lat: latitude in WGS84 of the earthquake location
            - Qpos: quality associated to the earthquake location (A for very good quality, E for bad quality, i.e. more than 50 km of possible error)
            - I0: epicentral intensity of the earthquake 
            - QI0: quality associated to the value of I0 (A to E)
            - Year: year of occurence of the earthquake
            - Month: month of occurence of the earthquake
            - Day: day of occurence of the earthquake
            - Ic: intensity of completeness of the earthquake. The intensities smaller than Ic in a macroseismic field are considered as incomplete
                 In the isoseismal radii based on intensity bins, intensities smaller than Ic are not taken into account
                 to compute the isoseismal radii.
        Optional columns for EvtFile:
            - Dc: distance of completeness of the earthquake. The macroseismic field located at greater epicentral distance than Dc is considered as incomplete
            - Depth: hypocentral depth of the earthquake
            - Hmin: lower bound of depth uncertainty
            - Hmax: upper bound of depth uncertainty
            - Mag: magnitude of the earthquake
            - StdM: uncertainty associated to magnitude.
    ponderation : str
        Name of the ponderation option that weight the earthquakes in the attenuation calibration.
        Available values are 'Ponderation dI', 'Ponderation evt-uniforme', 'Ponderation evt-reg' and
        'Ponderation evt-depth'. See ponderations.py documentation of more information.
        process (2 steps method).
    regiondata_name : str, optional
        Name of the file that contains the contour of different regions. Each region is identifed by an ID.
        Coordinates of the points that define the contour are in WGS84. The default is ''.
    binning_type : str, optional
        Name of the method used to bin the intensity data points (define isoseismal radii).
        The default is 'ROBS'.
    Beta : float, optional
        DESCRIPTION. The default is -3.5.
    Gamma : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    obsbin_plus : pandas.DataFrame
        Dataframe that contains metadata of the calibration earthquakes and the
        isoseismal radii of the calibration earthquakes.
        Columns of the dataframe:
            - EVID: earthquake ID
            - I: intensity value of the isoseismal
            - StdI: uncertainty associated to I
            - Depi: epicentral distance (km) of the isoseismal
            - Ndata: number of intensity data point used to compute the isoseismal
            - Io: epicentral intensity
            - Io_std: uncertainty associated to Io
            - Mag: magnitude of the earthquake
            - StdM: uncertainty associated to the magnitude
            - Depth: hypocentral depth of the earthquake
            - Hmin: lower bound of the depth uncertainty
            - Hmax: upper bound of the depth uncertainty
            - Io_ini: initial value of Io in inversion process. Equal to Io in this function. Used in the calibration process 
            - Hmin_ini: initial value of Hmin in inversion process. Equal to Hmin in this function. Used in the calibration process
            - Hmax_ini: initial value of Hmax in inversion process. Equal to Hmax in this function. Used in the calibration process
            - StdIo_inv: Value of the standard deviation associated to Io after inversion. In this function, equal to Io_std
            - eqStd: equivalent standard deviation used to weight the earthquakes in the calibration process (weight=1/std**2)

    """
    obsdata = pd.read_csv(obsdata_name, sep=';')
    evtdata = pd.read_csv(evtdata_name, sep=';')
    
    fichiers = fichier_input(obsdata, evtdata)
    data = eio.Evt(fichiers)
    
    columns_obsbinplus = ['EVID', 'I', 'StdI', 'Io', 'Io_std', 'Io_ini', 'Depi','Ndata', 'Mag',
                                        'StdM', 'Depth', 'Hmin', 'Hmax', 'Hmin_ini', 'Hmax_ini',
                                        'StdIo_inv']
    obsbin_plus = pd.DataFrame(columns=columns_obsbinplus)
    for evid in evtdata.EVID.values:
        # Attribuer un Depi aux Iobs
        #data.build(int(evid))
        data.build(evid)
        data.I0 = data.Io_ini
        #print(evid)
        #print(data.I0)
        #print(data.Obsevid.head())
        # Bin des Iobs
        data.Binning_Obs(0, data.Ic, method_bin=binning_type)
        #print(data.ObsBinn)
        evt_obsbin = data.ObsBinn
        #print(evt_obsbin.Depi.dtype)
        evt_obsbin.loc[:, 'Depth'] = data.depth
        evt_obsbin.loc[:, 'Hmin'] = data.Hmin
        evt_obsbin.loc[:, 'Hmax'] = data.Hmax
        evt_obsbin.loc[:, 'Hmin_ini'] = data.Hmin
        evt_obsbin.loc[:, 'Hmax_ini'] = data.Hmax
        evt_obsbin.loc[:, 'Mag'] = data.Mag
        evt_obsbin.loc[:, 'StdM'] = data.StdM
        evt_obsbin.loc[:, 'Io_ini'] = data.I0
        evt_obsbin.loc[:, 'StdIo_inv'] = data.QI0
        #hypo_tmp = np.sqrt(evt_obsbin.Depi.values**2 + data.depth**2)
        # X_tmp = evt_obsbin.I.values - Beta*np.log10(hypo_tmp) - Gamma*hypo_tmp
        # evt_obsbin.loc[:, 'X'] = np.average(X_tmp, weights=1/evt_obsbin.StdI.values**2)
#        evt_obsbin.loc[:, 'Io'] = data.I0
        evt_obsbin = evt_obsbin[columns_obsbinplus]
        #obsbin_plus = obsbin_plus.append(evt_obsbin)
        obsbin_plus = pd.concat([obsbin_plus, evt_obsbin])
    if regiondata_name != '':
        obsbin_plus = attribute_region(evtdata, obsbin_plus, regiondata_name)
    else:
        obsbin_plus.loc[:, 'RegID'] = -99
    obsbin_plus = ensure_floatcoltype(obsbin_plus)
    obsbin_plus = evt_weights(obsbin_plus, ponderation)
    return obsbin_plus

def get_eqStd4I0(StdIo_inv):
    Std ={'A':0.25,'B':0.375,'C':0.5,'E':0.750, 'K':0.5}
    if StdIo_inv <= Std['A']:
        StdIo_inv = Std['A']
    elif StdIo_inv <= Std['B']:
        StdIo_inv = Std['B']
    elif StdIo_inv <= Std['C']:
        StdIo_inv = Std['C']
    else:
        StdIo_inv = Std['K']
    StdI_eq = np.sqrt(StdIo_inv/(0.1*Std['A']))
    return StdI_eq

def add_I0as_datapoint(obsbin_plus, liste_evt):
    """
    Add the epicentral intensity in the isoseismal radii

    Parameters
    ----------
    obsbin_plus : pandas.DataFrame
        See prepare_input4calibration.
    liste_evt : list
        List of the calibration earthquakes in obsbin_plus.

    Returns
    -------
    obsbin_plus with a supplementary isoseismal for each earthquake 
    with intensity equal to eipcentral intensity and epicentral distance
    equal to 0.

    """
    
    #Std ={'A':0.25,'B':0.375,'C':0.5,'E':0.750, 'K':0.5}
    last_index = len(obsbin_plus)+1
    #print(obsbin_plus.columns)
    for evid in liste_evt:
        depth = obsbin_plus[obsbin_plus.EVID==evid]['Depth'].values[0]
        Hmin = obsbin_plus[obsbin_plus.EVID==evid]['Hmin'].values[0]
        Hmax = obsbin_plus[obsbin_plus.EVID==evid]['Hmax'].values[0]
        Hmin_ini = obsbin_plus[obsbin_plus.EVID==evid]['Hmin_ini'].values[0]
        Hmax_ini = obsbin_plus[obsbin_plus.EVID==evid]['Hmax_ini'].values[0]
        Mag = obsbin_plus[obsbin_plus.EVID==evid]['Mag'].values[0]
        StdM = obsbin_plus[obsbin_plus.EVID==evid]['StdM'].values[0]
        Io_Std = obsbin_plus[obsbin_plus.EVID==evid]['Io_std'].values[0]
        StdIo_inv = obsbin_plus[obsbin_plus.EVID==evid]['StdIo_inv'].values[0]
        I0 = obsbin_plus[obsbin_plus.EVID==evid]['Io'].values[0]
        #I0_ini = obsbin_plus[obsbin_plus.EVID==evid]['Io_ini'].values[0]
        regID = obsbin_plus[obsbin_plus.EVID==evid]['RegID'].values[0]
        if 'beta' in obsbin_plus.columns:
            beta = obsbin_plus[obsbin_plus.EVID==evid]['beta'].values[0]
        if 'gamma' in obsbin_plus.columns:
            gamma = obsbin_plus[obsbin_plus.EVID==evid]['gamma'].values[0]
        StdI_eq = get_eqStd4I0(StdIo_inv)
        Depi = 0
        Ndata = 0   
        if ('beta' in obsbin_plus.columns)and('gamma' in obsbin_plus.columns):

            obsbin_plus = pd.concat([obsbin_plus, pd.DataFrame.from_dict({'EVID' : [evid],
                                              'Depi' : [Depi],
                                              'Hypo': [depth],
                                              'I': [I0],
                                              'StdI': [StdI_eq],
                                              'Io': [I0],
                                              'Io_std': [Io_Std],
                                              'Ndata': [Ndata],
                                              'Depth': [depth],
                                              'Hmin': [Hmin],
                                              'Hmax': [Hmax],
                                              'Hmin_ini': [Hmin_ini],
                                              'Hmax_ini': [Hmax_ini],
                                              'Mag': [Mag],
                                              'StdM': [StdM],
                                              'StdIo_inv': [StdIo_inv],
                                              'RegID': [regID],
                                              'beta': [beta],
                                              'gamma': [gamma]})])
        else:
          
            obsbin_plus = pd.concat([obsbin_plus, pd.DataFrame.from_dict({'EVID' : [evid],
                                              'Depi' : [Depi],
                                              'Hypo': [depth],
                                              'I': [I0],
                                              'StdI': [StdI_eq],
                                              'Io': [I0],
                                              'Io_std': [Io_Std],
                                              'Ndata': [Ndata],
                                              'Depth': [depth],
                                              'Hmin': [Hmin],
                                              'Hmax': [Hmax],
                                              'Hmin_ini': [Hmin_ini],
                                              'Hmax_ini': [Hmax_ini],
                                              'Mag': [Mag],
                                              'StdM': [StdM],
                                              'StdIo_inv': [StdIo_inv],
                                              'RegID': [regID]})])

        last_index +=1
    obsbin_plus.sort_values(by=['EVID', 'I'], inplace=True)
    try:
        obsbin_plus.drop(['index'], axis=1)
    except:
        pass
        
    obsbin_plus.reset_index(inplace=True)
    if 'level_0' in obsbin_plus.columns:
        obsbin_plus.drop(['level_0'], axis=1)
    obsbin_plus.drop(['index'], axis=1, inplace=True)
    return obsbin_plus
        
