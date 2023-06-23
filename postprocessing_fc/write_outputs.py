# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:59:06 2023

@author: PROVOST-LUD
"""



def write_IPEfile4QUakeMD(coeff_dataframe, outname, comments='No comments', C1_col='C1', C2_col='C2', beta_col='beta', gamma_col='gamma'):
    """
    Write the ouputs of IPE calibration, i.e. C1, C2, beta and gamma values, from a dataframe to an IPE file compatible with QUake-MD tool
    and the CalIPE post-processing tools

    Parameters
    ----------
    coeff_dataframe : pandas.DataFrame
        DataFrame that contains the coefficients of the IPE and the probability/weight associated to each IPE. This dataframe
        should have a column named 'proba', that contains the probability/weight associated to each IPE. It should also contains
        columns with the C1, C2, beta and gamma coefficient values. The name of these columns is up to the user (see the other input parameters)
    outname : str
        Name of the output file.
    comments : str, optional
        Description of the IPEs written in the output file. The default is 'No comments'.
    C1_col : str, optional
        Name of the columns that contains the C1 values. The default is 'C1'.
    C2_col : str, optional
        Name of the columns that contains the C2 values. The default is 'C2'.
    beta_col : str, optional
        Name of the columns that contains the beta values. The default is 'beta'.
    gamma_col : str, optional
        Name of the columns that contains the gamme values. The default is 'gamma'.

    Returns
    -------
    None.

    """
    fichier_resultats = open(outname + '.txt', 'w')
    fichier_resultats.write(comments +'\n')
    fichier_resultats.write('\n')
    fichier_resultats.write('Weigth\tC1\tC2\tBeta\tGamma\n')
    fichier_resultats.write('\n')
    for ind, row in coeff_dataframe.iterrows():
        fichier_resultats.write('%0.5f\t%0.5f\t%0.5f\t%0.5f\t%0.5f\n' % (row['proba'], row[C1_col], row[C2_col], row[beta_col], row[gamma_col]))
    fichier_resultats.close()
    return