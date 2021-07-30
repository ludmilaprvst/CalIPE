# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:48:09 2021

@author: PROVOST-LUD
"""



def readBetaFile(nomFichier):
    fichier = open(nomFichier, 'r')
    contenuFichier = fichier.read()
    contenuFichier = contenuFichier.split('\n')
    
    
    ligneBeta = contenuFichier[0]
    beta = ligneBeta[ligneBeta.index(':')+1:]
    beta = float(beta)
    
    ligneBetaStd = contenuFichier[1]
    stdbeta = ligneBetaStd.split(':')[-1]
    stdbeta = float(stdbeta)
    
    ligneBetaIni = contenuFichier[2]
    beta_ini = ligneBetaIni.split(':')[-1]
    beta_ini = float(beta_ini)
    
    ligneNbreIteration = contenuFichier[3]
    nbre_iteration = ligneNbreIteration.split(':')[-1]
    nbre_iteration = float(nbre_iteration)
    
    ligneNbreIterationMax = contenuFichier[4]
    nbre_iterationMax = ligneNbreIterationMax.split(':')[-1]
    nbre_iterationMax = float(nbre_iterationMax)
    
    ligneNbreIterationMin = contenuFichier[5]
    nbre_iterationMin = ligneNbreIterationMin.split(':')[-1]
    nbre_iterationMin = float(nbre_iterationMin)
    
    ligneNbreEvt = contenuFichier[6]
    nbreEvt = ligneNbreEvt.split(':')[-1]
    nbreEvt = float(nbreEvt)
    
    ligneNbreI = contenuFichier[7]
    nbreI = ligneNbreI.split(':')[-1]
    nbreI = float(nbreI)
    
    ligneEvtfile = contenuFichier[8]
    evtfile = ligneEvtfile.split(':')[-1]
    
    ligneObsfile = contenuFichier[9]
    obsfile = ligneObsfile.split(':')[-1]

    fichier.close()
    
    return beta, stdbeta, beta_ini, nbre_iteration, nbre_iterationMax, nbre_iterationMin, nbreEvt, nbreI, evtfile, obsfile