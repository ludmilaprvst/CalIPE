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
    d = {"Beta values during inversion" : suivi_beta}
    output02 = pd.DataFrame(data=d)
    output02.to_csv(fullname02, sep=';')
    obsBin_plus_end.to_csv(fullname03, index=False, sep=';')
   

def Kovbeta_onedataset(evtdata_name, obsdata_name, outputfolder,
                       liste_beta_ini, ponderation,
                       binning_type, regiondata_name,
                       NminIter, NmaxIter):
    
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