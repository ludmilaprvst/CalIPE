# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 13:51:29 2021

@author: PROVOST-LUD
"""

from prepa_data import prepare_input4calibration
from combinaison_calib import calib_attBeta_Kov, calib_attBetaGamma_Kov
from ponderations import savename_weights
import numpy as np
import pandas as pd
import sys
import os


def write_betaresults(name, outputfolder, beta, suivi_beta, beta_ini,
                      cov_beta, NminIter, NmaxIter,
                      nbre_evt, nbreI, evtdata_name, obsdata_name,
                      obsBin_plus_end):
    """
    Function that writes the output of the attenuation coefficient beta calibration
    The outputs will be written in three files. The first file contains:
        - the final beta value of the inversion process (median value of the gaussian),
        - its associated standard deviation based on the uncertainties of the intensity data,
        - the initial value of the coefficient beta before inversion,
        - the number of iteration needed to converge,
        - the maximal number of iteration allowed,
        - the minimal number of iteration allowed,
        - the number of calibration earthquake used,
        - the number of intensity data used,
        - the Evt file used for the calibration process,
        - the Obs file used for the calibration process,
        - the name of the python file where this function is called
    The beginning of the first file name is betaFinal. The second file contains
    the values of the beta coefficient of each iteration. The two columns of the file
    are respectively the number of the iteration and the corresponding value of beta.
    The beginning of the second file name is betaWay.
    The third file contains the intensities data used in the inversion process, along
    with the depth and epicentral intensity after inversion for each earthquake.
    The columns of this .csv file are:
        - EVID : earthquake ID
        - I: value of the intensity of the isoseismal radii
        - StdI: Uncertainty associated to I
        - Io: epicentral intensity after the inversion process
        - Io_std : uncertainty associated to the inital value of Io
        - Io_ini: initial value of epicentral intensity
        - Depi : value of the epicentral distance of the isoseismal radii
        - Ndata: number of data used to compute the isoseismal radii
        - Mag: magnitude of the earthquake
        - StdM : uncertainty associated to the magnitude
        - Depth: hypocentral depth after the inversion process
        - Hmin : minimal bound for the depth inversion
        - Hmax : maximal bound for the depth inversion
        - RegID: ID of the geographical region of the considered earthquake
        - eqStd: the equivalent standard deviation used in the inversion process
        to compute weight for each isoseismal radii. (see the weights function)
        - Hmin_ini : identical to Hmin
        - Hmax_ini : identical to Hmax
    The beginning of the third file name is obsbinEnd.
    :param name: core name given to the outputs. For example, for the first output
                 file, the output name would be 'betaFinal_''+ name +'_betaini.txt'
    :param outputfolder: folder where the outputs will be saved
    :param beta: the value of the beta coefficient after inversion
    :param suivi_beta: array with the values of the beta coefficient of each iteration in the inversion process
    :param beta_ini: initial value of the beta coefficient
    :param cov_beta: covariance matrix of the beta coefficient after the inversion process
    :param NminIter: minimal number of iteration allowed
    :param NmaxIter: maximal number of iteration allowed
    :param nbre_evt: number of earthquakes used in the inversion process
    :param nbreI:  number of intensity data used in the inversion process
    :param evtdata_name: name of the Evt file used in the calibration process
    :param obsdata_name: name of the Obs file used in the calibration process
    :param obsBin_plus_end: dataframe containing the intensity data used in the calibration process. 
                            The columns of the dataframe are the same as the obsbinEnd output file.
    
    :type name: str
    :type outputfolder: str 
    :type beta: float
    :type suivi_beta: numpy.darray
    :type beta_ini: float
    :type cov_beta:2-D array
    :type NminIter: int
    :type NmaxIter: int
    :type nbre_evt: int
    :type nbreI: int
    :type evtdata_name: str
    :type obsdata_name: str
    :type obsBin_plus_end: pandas.dataframe
    """
    str_betaini = str(int(-10*beta_ini))
    fullname01 = outputfolder + '/betaFinal_' + name + str_betaini + '.txt'
    fullname02 = outputfolder + '/betaWay_' + name + str_betaini + '.txt'
    fullname03 = outputfolder + '/obsbinEnd_' + name + str_betaini + '.csv'
    output_file = open(fullname01, 'w')
    output_file.write('Beta:' +str(beta)+'\n')
    output_file.write('Beta Std:' +str(np.sqrt(np.diag(cov_beta))[0])+'\n')
    output_file.write('Beta ini:' +str(beta_ini)+'\n')
    output_file.write('Nbre iteration:' +str(len(suivi_beta)) + ' \n')
    output_file.write('Nbre iteration max:' +str(NmaxIter) + ' \n')
    output_file.write('Nbre iteration min:' +str(NminIter) + ' \n')
    output_file.write('Nbre evt:' +str(int(nbre_evt))+'\n')
    output_file.write('Nbre data:' +str(len(obsBin_plus_end))+'\n')
    output_file.write('Fichier Evt:' +evtdata_name+'\n')
    output_file.write('Fichier Obs:' +obsdata_name+'\n')

    output_file.write('Nom fichier execute:' +sys.argv[0]+'\n')
    output_file.close()
    d = {"Beta values during inversion" : suivi_beta}
    output02 = pd.DataFrame(data=d)
    output02.to_csv(fullname02, sep=';')
    obsBin_plus_end.to_csv(fullname03, index=False, sep=';')
    
def write_betagammaresults(name, outputfolder,
                           beta, suivi_beta, beta_ini, 
                           gamma, suivi_gamma, gamma_ini, cov_betagamma,
                           NminIter, NmaxIter,
                           nbre_evt, nbreI, evtdata_name, obsdata_name,
                           obsBin_plus_end):
    """
    Function that writes the output of the attenuation coefficients beta and gamma calibration
    The outputs will be written in three files. The first file contains:
        - the final beta value of the inversion process (median value of the gaussian),
        - its associated standard deviation based on the uncertainties of the intensity data,
        - the initial value of the coefficient beta before inversion,
        - the final gamma value of the inversion process (median value of the gaussian),
        - its associated standard deviation based on the uncertainties of the intensity data,
        - the initial value of the coefficient gamma before inversion,
        - the number of iteration needed to converge,
        - the maximal number of iteration allowed,
        - the minimal number of iteration allowed,
        - the number of calibration earthquake used,
        - the number of intensity data used,
        - the Evt file used for the calibration process,
        - the Obs file used for the calibration process,
        - the name of the python file where this function is called
    The beginning of the first file name is betagammaFinal. The second file contains
    the values of the beta coefficient of each iteration. The two columns of the file
    are respectively the number of the iteration and the corresponding value of beta.
    The beginning of the second file name is betaWay.
    The third file contains the intensities data used in the inversion process, along
    with the depth and epicentral intensity after inversion for each earthquake.
    The columns of this .csv file are:
        - EVID: earthquake ID
        - I: value of the intensity of the isoseismal radii
        - StdI: Uncertainty associated to I
        - Io: epicentral intensity after the inversion process
        - Io_std: uncertainty associated to the inital value of Io
        - Io_ini: initial value of epicentral intensity
        - Depi: value of the epicentral distance of the isoseismal radii
        - Ndata: number of data used to compute the isoseismal radii
        - Mag: magnitude of the earthquake
        - StdM: uncertainty associated to the magnitude
        - Depth: hypocentral depth after the inversion process
        - Hmin: minimal bound for the depth inversion
        - Hmax: maximal bound for the depth inversion
        - RegID: ID of the geographical region of the considered earthquake
        - eqStd: the equivalent standard deviation used in the inversion process
        to compute weight for each isoseismal radii. (see the weights function)
        - Hmin_ini : identical to Hmin
        - Hmax_ini : identical to Hmax
    The beginning of the third file name is obsbinEnd.

    :param name: core name given to the outputs. For example, for the first output
                 file, the output name would be 'betaFinal_'+ name +'_betaini.txt'
    :param outputfolder: folder where the outputs will be saved
    :param beta: the value of the beta coefficient after inversion
    :param suivi_beta: array with the values of the beta coefficient of each iteration in the inversion process
    :param beta_ini: initial value of the beta coefficient
    :param gamma: the value of the gamma coefficient after inversion
    :param suivi_gamma: array with the values of the gamma coefficient of each iteration in the inversion process
    :param gamma_ini: initial value of the gamma coefficient
    :param cov_betagamma: covariance matrix of the beta and gamma coefficients after the inversion process
    :param NminIter: minimal number of iteration allowed
    :param NmaxIter: maximal number of iteration allowed
    :param nbre_evt: number of earthquakes used in the inversion process
    :param nbreI:  number of intensity data used in the inversion process
    :param evtdata_name: name of the Evt file used in the calibration process
    :param obsdata_name: name of the Obs file used in the calibration process
    :param obsBin_plus_end: dataframe containing the intensity data used in the calibration process. 
                            The columns of the dataframe are the same as the obsbinEnd output file.
    
    :type name: str
    :type outputfolder: str 
    :type beta: float
    :type suivi_beta: numpy.darray
    :type beta_ini: float
    :type gamma: float
    :type suivi_gamma: numpy.darray
    :type gamma_ini: float
    :type cov_betagamma: 2-D array
    :type NminIter: int
    :type NmaxIter: int
    :type nbre_evt: int
    :type nbreI: int
    :type evtdata_name: str
    :type obsdata_name: str
    :type obsBin_plus_end: pandas.dataframe

    """
    
    str_betaini = str(int(-10*beta_ini))
    str_gammaini = str(abs(gamma_ini))
    betagammaini_name = str_betaini + '_gammaini' + str_gammaini
    fullname01 = outputfolder + '/betagammaFinal_' + name + betagammaini_name + '.txt'
    fullname02 = outputfolder + '/betaWay_' + name + betagammaini_name + '.txt'
    fullname03 = outputfolder + '/obsbinEnd_' + name + betagammaini_name + '.csv'
    output_file = open(fullname01, 'w')
    output_file.write('Beta:' +str(beta)+'\n')
    output_file.write('Beta Std:' +str(np.sqrt(np.diag(cov_betagamma))[0])+'\n')
    output_file.write('Beta ini:' +str(beta_ini)+'\n')
    output_file.write('Gamma:' +str(gamma)+'\n')
    output_file.write('Gamma Std:' +str(np.sqrt(np.diag(cov_betagamma))[1])+'\n')
    output_file.write('Gamma ini:' +str(gamma_ini)+'\n')
    output_file.write('Nbre iteration:' +str(len(suivi_beta)) + ' \n')
    output_file.write('Nbre iteration max:' +str(NmaxIter) + ' \n')
    output_file.write('Nbre iteration min:' +str(NminIter) + ' \n')
    output_file.write('Nbre evt:' +str(int(nbre_evt))+'\n')
    output_file.write('Nbre data:' +str(len(obsBin_plus_end))+'\n')
    output_file.write('Fichier Evt:' +evtdata_name+'\n')
    output_file.write('Fichier Obs:' +obsdata_name+'\n')

    output_file.write('Nom fichier execute:' +sys.argv[0]+'\n')
    output_file.close()
    d = {"Beta values during inversion" : suivi_beta,
         "Gamma values during inversion": suivi_gamma}
    output02 = pd.DataFrame(data=d)
    output02.to_csv(fullname02, sep=';')
    obsBin_plus_end.to_csv(fullname03, index=False, sep=';')
   

def Kovbeta_onedataset(evtdata_name, obsdata_name, outputfolder,
                       liste_beta_ini, ponderation,
                       binning_type, regiondata_name,
                       NminIter, NmaxIter):
    """
    Function that calibrates the attenuation coefficient beta in the Koveslighety
    mathematical formula, with the hypothesis gamma coefficient equal to 0, for a given
    list of earthquake, its associated IDPs, depth and its uncertainties and epicentral
    intensity and its uncertainties. IDPs are grouped into isoseismal radii, with the
    method given by the user. Depth and epicentral intensity are inverted sequentially
    with the beta coefficient, within their defined uncertainties.
    See Provost, CalIPE (in writing) for more information
    
    Parameters
    ----------
    evtdata_name : str
        name of the evt data file. The list of the calibration earthquakes and associated 
        metadata are stored in this file. This .txt file contains 16 columns, separated by the ";" string:
            - EVID: ID of the earthquake
            - Year: year of occurence of the earthquake
            - Month: month of occurence of the earthquake
            - Day: day of occurence of the earthquake
            - Lon: longitude in WGS84 of the epicenter
            - Lat: latitude in WGS84 of the epicenter
            - QPos: quality of the epicenter location
            - I0 : epicentral intensity
            - QI0 : quality of the epicentral intensity value
            - Ic : intensity of completeness
            - Dc : distance of completeness
            - Mag: magnitude of the earthquake
            - StdM : uncertainty associated with the magnitude
            - Depth: hypocentral depth of the earthquake
            - Hmin : lower bound of uncertainty associated to depth
            - Hmax : upper bound of uncertainty associated to depth
    obsdata_name : str
        name of the obs data file. The IDPs of the calibration earthquakes are stored in this file.
        This .txt file contains 5 columns, separated by the ";" string:
            - EVID : ID of the earthquake
            - Lon: longitude in WGS84 of the IDP
            - Lat: latitude in WGS84 of the IDP
            - Iobs: value of intensity of the IDP
            - QIobs : quality associated to Iobs
    outputfolder : str
        Folder name where the outputs will be saved. See function write_betaresults()
    liste_beta_ini : list of float
        list with different initial beta values
    ponderation : str
        Name of the ponderation applied to the calibration earthquakes intensity data used
        in the inversion process of the beta coefficient
    binning_type : str
        Name of the method applied to the calibration earthquakes intensity date to compute 
        isoseismal radii. The isoseismal radii are the intensity data used in the 
        inversion process
    regiondata_name : str
        Name of the .txt file which contains the contour of the different regions
        defined by the user. Contour is described by a polygon. The coordinates of
        the polygon points are in WGS84 longitude and latitude.
        The three columns are separeted by ";":
            - ID_region: ID of the considered region
            - Lon: Longitude of the polygon points
            - Lat: Latitude of the polygon points
    NminIter : int
        Minimal number of iteration allowed
    NmaxIter : int
        Maximal number of iteration allowed.

    Returns
    -------
    None.

    """
    
    head, basename = os.path.split(evtdata_name)
    databasename = basename[:-4]
    weightname = savename_weights(ponderation)
    repere =  '_'.join([databasename, binning_type, weightname, 'betaini'])
    obsbin_plus = prepare_input4calibration(obsdata_name, evtdata_name,
                                            ponderation,
                                            regiondata_name, binning_type)
    liste_evt = np.unique(obsbin_plus.EVID.values)
    nbre_evt = len(liste_evt)
    nbreI = len(obsbin_plus)
    for beta_ini in liste_beta_ini:
        print(beta_ini)
        obsBin_plus_end, beta, cov_beta, suivi_beta = calib_attBeta_Kov(liste_evt, obsbin_plus, beta_ini, 
                                                              NminIter=NminIter, NmaxIter=NmaxIter, suivi_inversion=False,
                                                              dossier_suivi='../Outputs/suivi_inv_par_evt_ALPSPYREST_lesplusbeaux')
        write_betaresults(repere, outputfolder, beta, suivi_beta, beta_ini,
                          cov_beta, NminIter, NmaxIter,
                          nbre_evt, nbreI, evtdata_name, obsdata_name,
                          obsBin_plus_end)
        
def Kovbetagamma_onedataset(evtdata_name, obsdata_name, outputfolder,
                       liste_beta_ini, liste_gamma_ini, ponderation,
                       binning_type, regiondata_name,
                       NminIter, NmaxIter):
    """
    Function that calibrates the attenuation coefficients beta and gamma in the Koveslighety
    mathematical formula,for a given list of earthquake, its associated IDPs
    depth and its uncertainties and epicentral intensity and its uncertainties. 
    IDPs are grouped into isoseismal radii, with the method given by the user. 
    Depth and epicentral intensity are inverted sequentiallyvwith the beta and gamma coefficients, within their defined uncertainties.
    See Provost, CalIPE (in writing) for more information

    Parameters
    ----------
    evtdata_name : str
        name of the evt data file. The list of the calibration earthquakes and associated 
        metadata are stored in this file. This .txt file contains 16 columns, separated by the ";" string:
            - EVID: ID of the earthquake
            - Year: year of occurence of the earthquake
            - Month: month of occurence of the earthquake
            - Day: day of occurence of the earthquake
            - Lon: longitude in WGS84 of the epicenter
            - Lat: latitude in WGS84 of the epicenter
            - QPos: quality of the epicenter location
            - I0: epicentral intensity
            - QI0: quality of the epicentral intensity value
            - Ic: intensity of completeness
            - Dc: distance of completeness
            - Mag: magnitude of the earthquake
            - StdM: uncertainty associated with the magnitude
            - Depth: hypocentral depth of the earthquake
            - Hmin: lower bound of uncertainty associated to depth
            - Hmax: upper bound of uncertainty associated to depth
    obsdata_name : str
        name of the obs data file. The IDPs of the calibration earthquakes are stored in this file.
        This .txt file contains 5 columns, separated by the ";" string:
            - EVID: ID of the earthquake
            - Lon: longitude in WGS84 of the IDP
            - Lat: latitude in WGS84 of the IDP
            - Iobs: value of intensity of the IDP
            - QIobs: quality associated to Iobs
    outputfolder : str
        Folder name where the outputs will be saved. See function write_betagammaresults()
    liste_beta_ini : list of float
        list with different initial beta values
    liste_gamma_ini : list of float
        list with different initial gamma values
    ponderation : str
        Name of the ponderation applied to the calibration earthquakes intensity data used
        in the inversion process of the beta and gamma coefficients
    binning_type : str
        Name of the method applied to the calibration earthquakes intensity date to compute 
        isoseismal radii. The isoseismal radii are the intensity data used in the 
        inversion process
    regiondata_name : str
        Name of the .txt file which contains the contour of the different regions
        defined by the user. Contour is described by a polygon. The coordinates of
        the polygon points are in WGS84 longitude and latitude.
        The three columns are separeted by ";":
            - ID_region: ID of the considered region
            - Lon: Longitude of the polygon points
            - Lat: Latitude of the polygon points
    NminIter : int
        Minimal number of iteration allowed
    NmaxIter : int
        Maximal number of iteration allowed

    Returns
    -------
    None.

    """
    
    head, basename = os.path.split(evtdata_name)
    databasename = basename[:-4]
    weightname = savename_weights(ponderation)
    repere =  '_'.join([databasename, binning_type, weightname, 'betaini'])
    obsbin_plus = prepare_input4calibration(obsdata_name, evtdata_name,
                                            ponderation,
                                            regiondata_name, binning_type)
    liste_evt = np.unique(obsbin_plus.EVID.values)
    nbre_evt = len(liste_evt)
    nbreI = len(obsbin_plus)
    for beta_ini in liste_beta_ini:
        for gamma_ini in liste_gamma_ini:
            print(beta_ini)
            print(gamma_ini)
            (obsBin_plus_end,
             beta, gamma,
             cov_betagamma,
             suivi_beta, suivi_gamma) = calib_attBetaGamma_Kov(liste_evt, obsbin_plus,
                                                               beta_ini, gamma_ini,
                                                               NminIter=NminIter, NmaxIter=NmaxIter)
            write_betagammaresults(repere, outputfolder,
                                   beta, suivi_beta, beta_ini,
                                   gamma, suivi_gamma, gamma_ini,
                                   cov_betagamma, NminIter, NmaxIter,
                                   nbre_evt, nbreI, evtdata_name, obsdata_name,
                                   obsBin_plus_end)