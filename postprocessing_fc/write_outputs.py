# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:59:06 2023

@author: PROVOST-LUD
"""



def write_IPEfile4QUakeMD(coeff_dataframe, outname, comments='No comments', C1_col='C1', C2_col='C2', beta_col='beta', gamma_col='gamma'):
    fichier_resultats = open(outname + '.txt', 'w')
    fichier_resultats.write(comments +'\n')
    fichier_resultats.write('\n')
    fichier_resultats.write('Weigth\tC1\tC2\tBeta\tGamma\n')
    fichier_resultats.write('\n')
    for ind, row in coeff_dataframe.iterrows():
        fichier_resultats.write('%0.5f\t%0.5f\t%0.5f\t%0.5f\t%0.5f\n' % (row['proba'], row[C1_col], row[C2_col], row[beta_col], row[gamma_col]))
    fichier_resultats.close()
    return