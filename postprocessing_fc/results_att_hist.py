# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:41:44 2021

@author: PROVOST-LUD
"""

import numpy as np
import pandas as pd
import ntpath
import os
from postprocessing_Kovbeta import readBetaFile, readBetaGammaFile


def write_betahistfile(outputname,
                       beta_values, bins,
                       path_subsets, datasetlist,
                       specific_comment):
    """
    Function that write a file with the attenuation coefficient beta values
    histogram. The coefficient beta values results by the inversion of the data
    of the different subsets. 

    Parameters
    ----------
    outputname : str
        Name of the output file.
    beta_values : numpy.darray or list
        Output beta values.
    bins : int or sequence of scalars or str
        If bins is an int, it defines the number of equal-width bins in the
        given range (10, by default). If bins is a sequence, it defines a
        monotonically increasing array of bin edges, including the rightmost
        edge, allowing for non-uniform bin widths. If bins is a string,
        it defines the method used to calculate the optimal bin width,
        as defined by numpy.histogram_bin_edges.
    path_subsets : str
        Path to the folder where the inversion output with the beta values are stored.
    datasetlist : str
        path and name of the file where the subsets names are stored.
    specific_comment : str
        Add a specific comment to the histogram file.

    Returns
    -------
    bin_edges : array of dtype float
        Return the bin edges (length(hist)+1).
    hist : array
        The values of the histogram. See density and weights for a description
        of the possible semantics.
    normedHist : array
        Normed values of the histogram.

    """
    
    hist, bin_edges = np.histogram(beta_values,
                                   bins=bins)
    normedHist = hist/np.sum(hist)
    outputFRinstru = pd.DataFrame({'Beta value' : bin_edges[:-1]+np.diff(bin_edges)/2,
                                   'Probability' : normedHist})
    fichier = open(outputname, 'w')
    fichier.write('# Subsets path : ' + path_subsets + '\n')
    fichier.write('# Datasetlist : ' + datasetlist + '\n')
    fichier.write('# Other comments : ' + specific_comment + '\n')
    fichier.close()
    #df.to_csv(path, mode='a')
    outputFRinstru.to_csv(outputname, mode='a',index=False, sep=';', float_format='%.4f')
    fichier.close()
    return bin_edges, hist, normedHist



def get_hist_bootstrap(output_folder):
    """
    

    Parameters
    ----------
    output_folder : TYPE
        DESCRIPTION.

    Returns
    -------
    liste_beta : TYPE
        DESCRIPTION.
    beta_baseDB : TYPE
        DESCRIPTION.

    """
    
    liste_fichiers = os.listdir(output_folder)
    liste_beta = []
    for fichier in liste_fichiers:
        if 'betaFinal' in fichier:
            if 'FR' in fichier:
                beta_baseDB = readBetaFile(output_folder+'/'+fichier)[0]
            else:
                
                beta = readBetaFile(output_folder+'/'+fichier)[0]
                liste_beta.append(beta)
                #print(fichier, beta)
    return liste_beta, beta_baseDB

def plot_hist_bootstrap(ax, liste, beta_base,
                        xlim=[-4, -2], ylim=[0, 15],
                        bbins=np.arange(-4, -2, 0.1)):
    """
    

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    liste : TYPE
        DESCRIPTION.
    beta_base : TYPE
        DESCRIPTION.
    xlim : TYPE, optional
        DESCRIPTION. The default is [-4, -2].
    ylim : TYPE, optional
        DESCRIPTION. The default is [0, 15].
    bbins : TYPE, optional
        DESCRIPTION. The default is np.arange(-4, -2, 0.1).

    Returns
    -------
    None.

    """
    
    ax.hist(liste, bins=bbins, density=False, alpha=1, label='subsets beta results')
    ax.axvline(x=beta_base, color='k', lw=1,ls='--', label='beta value from the base database')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Beta values')
    #ax.legend(loc=2)
    ax.grid(which='both')
    
def get_table_inoutputbeta(output_folder):
    """
    

    Parameters
    ----------
    output_folder : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    liste_fichiers = os.listdir(output_folder)
    liste_beta = []
    liste_evtfile = []
    liste_betaini = []
    for fichier in liste_fichiers:
        if 'betaFinal' in fichier:
            if 'FR' in fichier:
                beta_baseDB = readBetaFile(output_folder+'/'+fichier)[0]
                liste_beta.append(beta_baseDB)
                evtfile = ntpath.basename(readBetaFile(output_folder+'/'+fichier)[8])
                basename = evtfile.split('.')[0]
                liste_evtfile.append(basename)
                liste_betaini.append(readBetaFile(output_folder+'/'+fichier)[2])
            else:
                
                beta = readBetaFile(output_folder+'/'+fichier)[0]
                liste_beta.append(beta)
                evtfile = ntpath.basename(readBetaFile(output_folder+'/'+fichier)[8])
                basename = evtfile.split('.')[0]
                liste_evtfile.append(basename)
                liste_betaini.append(readBetaFile(output_folder+'/'+fichier)[2])
                #print(fichier, beta)
                
    return pd.DataFrame.from_dict({'beta' : liste_beta,
                                   'beta_ini' : liste_betaini,
                                   'database' : liste_evtfile})

def get_list_beta_folder(output_folder):
    """
    

    Parameters
    ----------
    output_folder : TYPE
        DESCRIPTION.

    Returns
    -------
    liste_beta : TYPE
        DESCRIPTION.
    fichiers_beta : TYPE
        DESCRIPTION.

    """
    
    liste_fichiers = os.listdir(output_folder)
    liste_beta = []
    fichiers_beta = []
    for fichier in liste_fichiers:
        if 'betaFinal' in fichier:
            beta = readBetaFile(output_folder+'/'+fichier)[0]
            liste_beta.append(beta)
            fichiers_beta.append(fichier)
    return liste_beta, fichiers_beta

def get_list_betagamma_folder(output_folder):
    """
    

    Parameters
    ----------
    output_folder : TYPE
        DESCRIPTION.

    Returns
    -------
    liste_beta : TYPE
        DESCRIPTION.
    liste_gamma : TYPE
        DESCRIPTION.
    fichiers_betagamma : TYPE
        DESCRIPTION.

    """
    
    liste_fichiers = os.listdir(output_folder)
    liste_beta = []
    liste_gamma = []
    fichiers_betagamma = []
    for fichier in liste_fichiers:
        if 'betagammaFinal' in fichier:
            beta = readBetaGammaFile(output_folder+'/'+fichier)[0]
            liste_beta.append(beta)
            gamma = readBetaGammaFile(output_folder+'/'+fichier)[3]
            liste_gamma.append(gamma)
            fichiers_betagamma.append(fichier)
    return liste_beta, liste_gamma, fichiers_betagamma



def get_betagamma_hist(output_folder,
                       xedges=np.arange(-5, -1.9, 0.1),
                       yedges=np.arange(-0.01, 0.00002, 0.00001)):
    """
    

    Parameters
    ----------
    output_folder : TYPE
        DESCRIPTION.
    xedges : TYPE, optional
        DESCRIPTION. The default is np.arange(-5, -1.9, 0.1).
    yedges : TYPE, optional
        DESCRIPTION. The default is np.arange(-0.01, 0.00002, 0.00001).

    Returns
    -------
    hist : TYPE
        DESCRIPTION.
    xedges : TYPE
        DESCRIPTION.
    yedges : TYPE
        DESCRIPTION.

    """
    
    liste_beta, liste_gamma, fichiers_betagamma = get_list_betagamma_folder(output_folder)
    hist, xedges, yedges = np.histogram2d(liste_beta, liste_gamma, bins=(xedges, yedges))
    hist = hist/sum(hist)
    return hist, xedges, yedges