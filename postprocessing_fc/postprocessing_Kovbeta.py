# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:48:09 2021

@author: PROVOST-LUD
"""
import pandas as pd
import numpy as np
import os
from wrms import calcul_wrms_beta, plot_wrms_beta_1evt, plot_wrms_withHI0lines
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats.stats import pearsonr


def apply_KovBetaeq(I0, beta, Depi, H):
    hypo = np.sqrt(Depi**2 + H**2)
    return I0 + beta*np.log10(hypo/H)

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

def create_postprocessing_savedir(run_name, path):
    directory = path + '/' + run_name
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory

def plot_dI_Depi(run_name, path, ax, evthighlight='None'):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    
    obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                            obsbin.Depi.values, obsbin.Depth.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    mean_dI = obsbin.dI.mean()
    ax.scatter(obsbin.Depi.values, obsbin.dI.values)
    ax.axhline(y=mean_dI, color='k', label='Mean residual')
    if not evthighlight == 'None':
        liste_evt = np.unique(obsbin.EVID.values)
        if evthighlight in liste_evt:
            tmp = obsbin[obsbin.EVID==evthighlight]
            ax.scatter(tmp.Depi.values, tmp.dI.values, label=evthighlight)
    ax.set_xlabel('Epicentral distance [km]')
    ax.set_ylabel('dI = Iobs - Ipred')
    outsiders = obsbin[np.abs(obsbin.dI)>0.5]
    return outsiders


def plot_dI_Iobs(run_name, path, ax, evthighlight='None'):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    
    obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                            obsbin.Depi.values, obsbin.Depth.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    mean_dI = obsbin.dI.mean()
    ax.scatter(obsbin.I.values, obsbin.dI.values)
    ax.axhline(y=mean_dI, color='k', label='Mean residual')
    if not evthighlight == 'None':
        liste_evt = np.unique(obsbin.EVID.values)
        if evthighlight in liste_evt:
            tmp = obsbin[obsbin.EVID==evthighlight]
            ax.scatter(tmp.I.values, tmp.dI.values, label=evthighlight)
    ax.set_xlabel('Iobs')
    ax.set_ylabel('dI = Iobs - Ipred')
    outsiders = obsbin[np.abs(obsbin.dI)>0.5]
    return outsiders

def plot_dIH(run_name, path, ax):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                            obsbin.Depi.values, obsbin.Depth.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    obsbin_gp = obsbin.groupby('EVID').mean()
    ax.scatter(obsbin_gp.Depth.values, obsbin_gp.dI.values)
    ax.axhline(y=obsbin_gp.dI.mean(), color='k', label='Mean event residual')
    ax.set_xlabel('Depth [km]')
    ax.set_ylabel('dI = Iobs - Ipred')
    outsiders = obsbin_gp[np.abs(obsbin_gp.dI)>0.5]
    return outsiders

def plot_dII0(run_name, path, ax):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                            obsbin.Depi.values, obsbin.Depth.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    obsbin_gp = obsbin.groupby('EVID').mean()
    ax.scatter(obsbin_gp.Io_ini.values, obsbin_gp.dI.values)
    ax.axhline(y=obsbin_gp.dI.mean(), color='k', label='Mean event residual')
    ax.set_xlabel('I0 (from database)')
    ax.set_ylabel('dI = Iobs - Ipred')
    outsiders = obsbin_gp[np.abs(obsbin_gp.dI)>0.5]
    return outsiders

def plot_dIMag(run_name, path, ax):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                            obsbin.Depi.values, obsbin.Depth.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    obsbin_gp = obsbin.groupby('EVID').mean()
    ax.scatter(obsbin_gp.Mag.values, obsbin_gp.dI.values)
    ax.axhline(y=obsbin_gp.dI.mean(), color='k', label='Mean event residual')
    ax.set_xlabel('I0 (from database)')
    ax.set_ylabel('dI = Iobs - Ipred')
    outsiders = obsbin_gp[np.abs(obsbin_gp.dI)>0.1]
    return outsiders

def compute_withinbetweenevt_sigma(obsbin, beta):
    obsbin.loc[:, 'Hypo'] = np.sqrt(obsbin.Depi.values**2 + obsbin.Depth.values**2)
    obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                            obsbin.Depi.values, obsbin.Depth.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    obsbin_gp = obsbin.groupby('EVID').mean()
    dict_meandIevt = obsbin_gp.to_dict(orient='index')
    obsbin.loc[:, 'dImeanevt'] = obsbin.apply(lambda row: dict_meandIevt[row['EVID']]['dI'], axis=1)
    obsbin.loc[:, 'within_residuals'] = obsbin.I.values - (obsbin.Ipred.values - obsbin.dImeanevt.values)
    obsbin_gp = obsbin.groupby('EVID').mean()
    
    betweennevt_sigma = np.std(obsbin_gp.within_residuals.values)
    withinevt_sigma = np.std(obsbin.within_residuals.values)
    sigma = np.sqrt(betweennevt_sigma**2 + withinevt_sigma**2)
    return withinevt_sigma, betweennevt_sigma, sigma
 
def compute_stats(run_name, path):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    obsbin.loc[:, 'Hypo'] = np.sqrt(obsbin.Depi.values**2 + obsbin.Depth.values**2)
    obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                            obsbin.Depi.values, obsbin.Depth.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    mean_dI = obsbin.dI.mean()
    poids = 1/obsbin.StdI**2
    variance = np.sum(poids*(mean_dI - obsbin.loc[:, 'dI'])**2)/np.sum(poids)
    
    withinevt_sigma, betweennevt_sigma, sigma = compute_withinbetweenevt_sigma(obsbin, beta)
    wrms = calcul_wrms_beta(obsbin, beta)[0]
    return mean_dI, variance, withinevt_sigma, betweennevt_sigma, sigma, wrms


def plot_sigma_diffDB(output_folder, ax):
    liste_fichiers = os.listdir(output_folder)
    compt = 0
    withinevt_sigma_list = []
    betweennevt_sigma_list = []
    sigma_list = []
    beta_list = []
    for fichier in liste_fichiers:
        if 'betaFinal' in fichier:
            compt += 1
            run_name = fichier[10:-4]
            
            beta_file = 'betaFinal_' + run_name + '.txt'
            obsbinfile = 'obsbinEnd_' + run_name + '.csv'
            beta = readBetaFile(output_folder + '/' + beta_file)[0]
            obsbin = pd.read_csv(output_folder + '/' + obsbinfile, sep=';')
            (withinevt_sigma,
             betweennevt_sigma, sigma) = compute_withinbetweenevt_sigma(obsbin, beta)
            if betweennevt_sigma<0.16:
                print(run_name)
            withinevt_sigma_list.append(withinevt_sigma)
            betweennevt_sigma_list.append(betweennevt_sigma)
            sigma_list.append(sigma)
            beta_list.append(beta)
#        print('out of beta')
#    print(len(sigma_list))
    ax.scatter(beta_list, sigma_list, c='Gray', marker='s', label='Sigma')
    ax.scatter(beta_list, withinevt_sigma_list, c='FireBrick', marker='>',
                label='Within event sigma')
    ax.scatter(beta_list, betweennevt_sigma_list, c='DarkBlue', marker='<',
                label='Between event sigma')
    ax.set_xlabel('Beta value')
    ax.set_ylabel('Standard deviation')
    print('collection done')
    return withinevt_sigma_list, betweennevt_sigma_list, sigma_list, beta_list

def plot_meandI_diffDB(output_folder, ax):
    liste_fichiers = os.listdir(output_folder)
    compt = 0

    meandI_list = []
    beta_list = []
    for fichier in liste_fichiers:
        if 'betaFinal' in fichier:
            compt += 1
            run_name = fichier[10:-4]
            beta_file = 'betaFinal_' + run_name + '.txt'
            obsbinfile = 'obsbinEnd_' + run_name + '.csv'
            beta = readBetaFile(output_folder + '/' + beta_file)[0]
            obsbin = pd.read_csv(output_folder + '/' + obsbinfile, sep=';')
            obsbin.loc[:, 'Hypo'] = np.sqrt(obsbin.Depi.values**2 + obsbin.Depth.values**2)
            obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                                    obsbin.Depi.values, obsbin.Depth.values)
            obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
            mean_dI = obsbin.dI.mean()
            meandI_list.append(mean_dI)
            beta_list.append(beta)
    ax.scatter(beta_list, meandI_list, label='Mean intensity residual (Iobs-Ipred)')
    ax.set_xlabel('Beta value')
    ax.set_ylabel('Mean intensity residual')
    return mean_dI, beta_list
            
def plot_stats_diffDB(output_folder, ax1, ax2):
    liste_fichiers = os.listdir(output_folder)
    compt = 0
    withinevt_sigma_list = []
    betweennevt_sigma_list = []
    sigma_list = []
    meandI_list = []
    beta_list = []
    for fichier in liste_fichiers:
        if 'betaFinal' in fichier:
            compt += 1
            run_name = fichier[10:-4]
            beta_file = 'betaFinal_' + run_name + '.txt'
            beta = readBetaFile(output_folder + '/' + beta_file)[0]
            (mean_dI, variance, withinevt_sigma,
             betweennevt_sigma, sigma, wrms) = compute_stats(run_name, output_folder)
            withinevt_sigma_list.append(withinevt_sigma)
            betweennevt_sigma_list.append(betweennevt_sigma)
            sigma_list.append(sigma_list)
            meandI_list.append(mean_dI)
            beta_list.append(beta)
    ax1.scatter(beta_list, sigma_list, c='Gray', marker='s', label='Sigma')
    ax1.scatter(beta_list, withinevt_sigma_list, c='FireBrick', marker='>',
                label='Within event sigma')
    ax1.scatter(beta_list, betweennevt_sigma_list, c='DarkBkue', marker='<',
                label='Between event sigma')
    ax1.set_xlabel('Beta value')
    ax1.set_ylabel('Standard deviation')
    ax2.scatter(beta_list, meandI_list)
    ax2.set_xlabel('Beta value')
    ax2.set_ylabel('Mean intensity residual (Iobs- Ipred)')

def plot_dI_Dhypo(run_name, path, ax, evthighlight='None'):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    obsbin.loc[:, 'Hypo'] = np.sqrt(obsbin.Depi.values**2 + obsbin.Depth.values**2)
    obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                            obsbin.Depi.values, obsbin.Depth.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    mean_dI = obsbin.dI.mean()
    ax.scatter(obsbin.Hypo.values, obsbin.dI.values)
    ax.axhline(y=mean_dI, color='k', label='Mean residual')
    if not evthighlight == 'None':
        liste_evt = np.unique(obsbin.EVID.values)
        if evthighlight in liste_evt:
            tmp = obsbin[obsbin.EVID==evthighlight]
            ax.scatter(tmp.Hypo.values, tmp.dI.values, label=evthighlight)
    ax.set_xlabel('Hypocentral distance [km]')
    ax.set_ylabel('dI = Iobs - Ipred')
    outsiders = obsbin[np.abs(obsbin.dI)>0.5]
    return outsiders

def compute_pearsonDepi_1db(run_name, path):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                            obsbin.Depi.values, obsbin.Depth.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    return pearsonr(obsbin.Depi.values, obsbin.dI.values), beta

def compute_pearsonDhypo_1db(run_name, path):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                            obsbin.Depi.values, obsbin.Depth.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    obsbin.loc[:, 'Dhypo'] = np.sqrt(obsbin.Depi.values**2 + obsbin.Depth**2)
    return pearsonr(obsbin.Dhypo.values, obsbin.dI.values), beta

def compute_pearsonI_1db(run_name, path):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                            obsbin.Depi.values, obsbin.Depth.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    return pearsonr(obsbin.I.values, obsbin.dI.values), beta

def compute_pearsonI0_1db(run_name, path):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                            obsbin.Depi.values, obsbin.Depth.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    obsbin_gp = obsbin.groupby('EVID').mean()
    return pearsonr(obsbin_gp.Io_ini.values, obsbin_gp.dI.values), beta

def compute_pearsonH_1db(run_name, path):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                            obsbin.Depi.values, obsbin.Depth.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    obsbin_gp = obsbin.groupby('EVID').mean()
    return pearsonr(obsbin_gp.Depth.values, obsbin_gp.dI.values), beta

def compute_pearsonMag_1db(run_name, path):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    obsbin.loc[:, 'Ipred'] = apply_KovBetaeq(obsbin.Io.values, beta,
                            obsbin.Depi.values, obsbin.Depth.values)
    obsbin.loc[:, 'dI'] = obsbin.I - obsbin.Ipred
    obsbin_gp = obsbin.groupby('EVID').mean()
    return pearsonr(obsbin_gp.Mag.values, obsbin_gp.dI.values), beta

def plot_pearson_diffDB(output_folder, ax, xvalue_tested, **kwargs):
    liste_fichiers = os.listdir(output_folder)
    compt = 0
    pearson_list = []
    beta_list = []
    for fichier in liste_fichiers:
        if 'betaFinal' in fichier:
            compt += 1
            run_name = fichier[10:-4]
            if xvalue_tested == 'Iobs':
                pearson_coeff, beta = compute_pearsonI_1db(run_name, output_folder)
            elif xvalue_tested == 'Depi':
                pearson_coeff, beta = compute_pearsonDepi_1db(run_name, output_folder)
            elif xvalue_tested == 'Dhypo':
                pearson_coeff, beta = compute_pearsonDhypo_1db(run_name, output_folder)
            elif xvalue_tested == 'Mag':
                pearson_coeff, beta = compute_pearsonMag_1db(run_name, output_folder)
            elif xvalue_tested == 'I0':
                pearson_coeff, beta = compute_pearsonI0_1db(run_name, output_folder)
            elif xvalue_tested == 'Depth':
                pearson_coeff, beta = compute_pearsonH_1db(run_name, output_folder)
            else:
                raise KeyError("xvalue_tested "+ xvalue_tested + " does not exist")
            pearson_list.append(pearson_coeff[0])
            beta_list.append(beta)
    ax.scatter(beta_list, pearson_list, **kwargs)
    ax.set_ylim([-1, 1])
    ax.axhline(y=-0.5, ls='--', color='k')
    ax.axhline(y=0.5, ls='--', color='k')
    ax.axhline(y=0, ls='-', color='k')
    ax.set_xlabel('Beta value')
    ax.set_ylabel('Pearson coefficient value')
    if xvalue_tested == 'Iobs':
        xtitle =  "Intensity value"
    elif xvalue_tested == 'Depi':
        xtitle =  "Epicentral distance [km]"
    elif xvalue_tested == 'Dhypo':
        xtitle =  "Hypocentral distance [km]"
    elif xvalue_tested == 'Mag':
        xtitle =  "Magnitude"
    elif xvalue_tested == 'I0':
        xtitle =  "Epicentral intensity"
    elif xvalue_tested == 'Depth':
        xtitle =  "Depth [km]"
    ax.set_title('Intensity residual = f('+ xtitle+')')
    ax.grid(which='both')
    return pearson_list, beta_list

def plot_Hlim(run_name, path, ax):
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    
    grouped_data = obsbin.groupby('EVID').mean()
    grouped_data.reset_index(level=0, inplace=True)
    
    deeperror = grouped_data.Hmax-grouped_data.Depth
    shallowerror = grouped_data.Depth-grouped_data.Hmin

    ax.errorbar(range(len(grouped_data)), grouped_data.Depth.values,
                 yerr=[shallowerror.values, deeperror.values], ls='',
                 fmt='none', color='DimGray')
    ax.invert_yaxis()
    ax.grid(which='both')
    ax.set_ylabel('Depth [km]')
    ax.set_xticks(range(len(grouped_data)))
    ticks_label = grouped_data['EVID'].values
    #ticks_label[0] = ' '
    ax.set_xticklabels(ticks_label, rotation=80)
    
def plot_endH_oneDataset(run_name, path, basicdatabase,
                         ax, beta_min=-4, beta_max=-2,
                         colorbar=True):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    grouped_data = obsbin.groupby('EVID').mean()
    grouped_data.reset_index(level=0, inplace=True)
    grouped_data.loc[:, 'Beta'] = beta
    list_evt = grouped_data.EVID.values
    xdata = basicdatabase[basicdatabase.EVID.isin(list_evt)].index.values
    sc = ax.scatter(xdata, grouped_data.Depth.values,
               c=grouped_data.Beta.values, vmin=beta_min, vmax=beta_max,
               zorder=10)
    if colorbar:
        plt.colorbar(sc, label="Beta value")
    
def plot_endH_diffSubsets(output_folder, basicdatabasename,
                          ax, beta_min=-4, beta_max=-2):
    basicdatabase = pd.read_csv(basicdatabasename, sep=';')
    grouped_basicdb = basicdatabase.groupby('EVID').mean()
    grouped_basicdb.reset_index(inplace=True)
    grouped_basicdb.reset_index(inplace=True)
    grouped_basicdb = grouped_basicdb[['index', 'EVID']]
    liste_fichiers = os.listdir(output_folder)
    compt = 0
    for fichier in liste_fichiers:
        if 'betaFinal' in fichier:
            compt += 1
            run_name = fichier[10:-4]
            if compt == 1:
                plot_endH_oneDataset(run_name, output_folder, grouped_basicdb, ax,
                                     beta_min=beta_min, beta_max=beta_max)
            else:
                plot_endH_oneDataset(run_name, output_folder, grouped_basicdb, ax,
                                     beta_min=beta_min, beta_max=beta_max,
                                     colorbar=False)
                
def plot_Iolim(run_name, path, ax):
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    
    grouped_data = obsbin.groupby('EVID').mean()
    grouped_data.reset_index(level=0, inplace=True)
    
    error = grouped_data.Io_std.values*2

    ax.errorbar(range(len(grouped_data)), grouped_data.Io_ini.values,
                 yerr=error, ls='',
                 fmt='none', color='DimGray')
    ax.grid(which='both')
    ax.set_ylabel('Epicentral intensity [km]')
    ax.set_xticks(range(len(grouped_data)))
    ticks_label = grouped_data['EVID'].values
    #ticks_label[0] = ' '
    ax.set_xticklabels(ticks_label, rotation=80)
    
def plot_endIo_oneDataset(run_name, path, basicdatabase,
                         ax, beta_min=-4, beta_max=-2,
                         colorbar=True):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    grouped_data = obsbin.groupby('EVID').mean()
    grouped_data.reset_index(level=0, inplace=True)
    grouped_data.loc[:, 'Beta'] = beta
    list_evt = grouped_data.EVID.values
    xdata = basicdatabase[basicdatabase.EVID.isin(list_evt)].index.values
    sc = ax.scatter(xdata, grouped_data.Io.values,
               c=grouped_data.Beta.values, vmin=beta_min, vmax=beta_max,
               zorder=10)
    if colorbar:
        plt.colorbar(sc, label="Beta value")
    
def plot_endIo_diffSubsets(output_folder, basicdatabasename,
                          ax, beta_min=-4, beta_max=-2):
    basicdatabase = pd.read_csv(basicdatabasename, sep=';')
    grouped_basicdb = basicdatabase.groupby('EVID').mean()
    grouped_basicdb.reset_index(inplace=True)
    grouped_basicdb.reset_index(inplace=True)
    grouped_basicdb = grouped_basicdb[['index', 'EVID']]
    liste_fichiers = os.listdir(output_folder)
    compt = 0
    for fichier in liste_fichiers:
        if 'betaFinal' in fichier:
            compt += 1
            run_name = fichier[10:-4]
            if compt == 1:
                plot_endIo_oneDataset(run_name, output_folder, grouped_basicdb, ax,
                                     beta_min=beta_min, beta_max=beta_max)
            else:
                plot_endIo_oneDataset(run_name, output_folder, grouped_basicdb, ax,
                                     beta_min=beta_min, beta_max=beta_max,
                                     colorbar=False)
                
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

def define_plotstyle(list_evt):
    df_plotstyle = pd.DataFrame(columns=['EVID', 'color', 'linestyle'])
    for ind, evid in enumerate(list_evt):
        ls, color = define_ls_color_byevt(ind)
        df_plotstyle.loc[ind, :] = [evid, color, ls]
    return df_plotstyle

def plot_wmrsbeta_byevt(run_name, path, ax,
                    minbeta, maxbeta, pasbeta,
                    custom_color_ls='none'):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    list_evt = np.unique(obsbin.EVID.values)
    if custom_color_ls == 'none':
        custom_color_ls = define_plotstyle(list_evt)
    for ind, row in custom_color_ls.iterrows():
        plot_wrms_beta_1evt(ax, obsbin, row['EVID'],
                 minbeta, maxbeta, pasbeta, color=row['color'],
                 ls=row['linestyle'])
        ax.axvline(x=beta, color='k')
    
def plot_wmrsHI0_byevt(run_name, path, evid,
                       fig_wrms,
                       minH=1, maxH=25, pasH=0.25,
                       minI0=2, maxI0=10, pasI0=0.1,
                       vmax=-99):
    beta_file = 'betaFinal_' + run_name + '.txt'
    obsbinfile = 'obsbinEnd_' + run_name + '.csv'
    beta = readBetaFile(path + '/' + beta_file)[0]
    obsbin = pd.read_csv(path + '/' + obsbinfile, sep=';')
    
    plot_wrms_withHI0lines(fig_wrms,
                           obsbin, evid, beta,
                           minH=minH, maxH=maxH, pasH=pasH,
                           minI0=minI0, maxI0=maxI0, pasI0=pasI0,
                           vmax=vmax)
    
    
        
    