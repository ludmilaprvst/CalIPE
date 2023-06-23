#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:31:20 2018

@author: PROVOLU
"""
import numpy as np
import pandas as pd
import sys

from scipy.optimize import curve_fit
#import statsmodels.api as sm
import matplotlib.pyplot as plt

#try:
#    import Fonctions_bin as a
#except:
#    import Fonctions_bin_p37 as a
#from Modules_Getbeta import CalcDist, read_obsfile, read_evtfile, read_critfile


class WLSIC_Kov_oneEvt():
    """
    Set of functions that inverse depth and epicentral intensity
    from macroseismic data for a given earthquake.
    The mathematical formulation is the Koveslighety equation:
        
        I = I0 + BETA.log10(Hypo/H) + GAMMA.(Hypo-H)
    
    where I is the intensity value, I0 the epicentral intensity, BETA the
    geometric attenuation coefficient, Hypo the hypocentral distance, H the
    hypocentral depth and GAMMA the intrisic attenuation coefficient.
    The endog data are the epicentral distance and the exog data the associated
    epicentral distance.
    """
    def __init__(self, ObsBinn, depth, Beta, Gamma, I0):
        """
        :param Obsbinn: dataframe with the binned intensity data for one earthquake.
                        This dataframe should at least have have the following
                        columns : 'I', 'Depi', 'StdI', which are respectively
                        the binned intensity, the associated epicentral distance
                        and the associated standard deviation.
        :param depth: hypocenter's depth of the considered earthquake.
                      Should be greater than 0
        :param Beta: the geometric attenuation coefficient of the Koveslighety equation
        :param Gamma: the intresic attenuation coefficient of the Koveslighety equation
        :param I0: the epicentral intensity of the considered earthquake
        :type Obsbinn: pandas.DataFrame
        :type depth: float
        :type Beta: float
        :type Gamma: float
        :type I0: float 
        """
        self.beta = Beta
        self.gamma = Gamma
        self.Obsbin = ObsBinn
        self.depth = depth
        self.I0 = I0
        
    def EMIPE_H(self, Depi, H):
        """
        Function used to inverse depth
        :param Depi: epicentral distances associated to the binned intensity data
        :param depth: hypocenter's depth of the considered earthquake.
                      Should be greater than 0
        :type Depi: numpy.array
        :type depth: float
        """
        I = self.I0 + self.beta*np.log10(np.sqrt(Depi**2+H**2)/H) + self.gamma*(np.sqrt(Depi**2+H**2)-H)
        return np.array(I, dtype=float)

    def EMIPE_JACdH(self, Depi, H):
        """
        The jacobian fuunction associated to EMIPE_H
        :param Depi: epicentral distances associated to the binned intensity data
        :param depth: hypocenter's depth of the considered earthquake.
                      Should be greater than 0
        :type Depi: numpy.array
        :type depth: float
        """
        Hypo = np.sqrt(Depi**2+H**2)
        tmpValue = H/Hypo
        g = self.beta*(tmpValue**2-1.)/(H*np.log(10)) +  self.gamma*(tmpValue-1)
        return g.reshape(len(Depi),1)
    
    def EMIPE_I0(self, Depi, I0):
        """
        Function used to inverse epicentral intensity.
        :param Depi: epicentral distances associated to the binned intensity data
        :param I0: epicentral intensity
        :type Depi: numpy.array
        :type I0: float
        """
        I = I0 + self.beta*np.log10(np.sqrt(Depi**2+self.depth**2)/self.depth)+ self.gamma*(np.sqrt(Depi**2+self.depth**2)-self.depth)
        return I

    def EMIPE_JACdI0(self, Depi, I0):
        """
        The jacobian fuunction associated to EMIPE_I0
        :param Depi: epicentral distances associated to the binned intensity data
        :param I0: epicentral intensity
        :type Depi: numpy.array
        :type I0: float
        """
        g = np.ones(len(Depi))
        return g.reshape(len(Depi), 1)
    
    def do_wlsic_depth(self, depth_inf, depth_sup):
        """
        Function used to launch depth inversion within limits.
        
        :param depth_inf: lower depth limit of inversion
        :param depth_sup: upper depth limit of inversion
        :type depth_inf: float
        :type depth_sup: float
        :return: [popt, pcov] list with popt a (1, 1) shape array containing
                the inverted depth. pcov is a 2-D array and 
                the estimated covariance of popt. The diagonals provide the
                variance of the parameter estimate. 
                To compute one standard deviation errors on the parameters use
                perr = np.sqrt(np.diag(pcov)). 
        """
        Ibin = self.Obsbin['I'].values.astype(float)
        Depi = self.Obsbin['Depi'].values.astype(float)
        StdI = self.Obsbin['StdI'].values.astype(float)
        resH = curve_fit(self.EMIPE_H, Depi, Ibin, p0=self.depth,
                                 jac= self.EMIPE_JACdH, bounds=(depth_inf, depth_sup),
                                 sigma=StdI, absolute_sigma=True,
                                 ftol=5e-2)
        return resH
    
    def do_wlsic_I0(self, I0_inf, I0_sup):
        """
        Function used to launch epicentral intensity inversion within limits
        :param depth_inf: lower epicentral intensity limit of inversion
        :param depth_sup: upper epicentral intensity limit of inversion
        :type I0_inf: float
        :type I0_sup: float
        :return: [popt, pcov] list with popt a (1, 1) shape array containing
                the inverted epicentral intensity. pcov is a 2-D array and 
                the estimated covariance of popt. The diagonals provide the
                variance of the parameter estimate. 
                To compute one standard deviation errors on the parameters use
                perr = np.sqrt(np.diag(pcov)).
        """
        Ibin = self.Obsbin['I'].values.astype(float)
        Depi = self.Obsbin['Depi'].values.astype(float)
        StdI = self.Obsbin['StdI'].values.astype(float)
        resI0 = curve_fit(self.EMIPE_I0, Depi, Ibin, p0=self.I0,
                                  jac= self.EMIPE_JACdI0, bounds=(I0_inf, I0_sup),
                                  sigma=StdI, absolute_sigma=True,
                                  xtol=1e-2, loss='soft_l1')
        return resI0
    
    
class WLS_Kov():
    """
    Set of functions that inverse the coefficients of the Koveslighety equation:
    
        I = I0 + BETA.log10(Hypo/H) + GAMMA.(Hypo-H)
    
    where I is the intensity value, I0 the epicentral intensity, BETA the
    geometric attenuation coefficient, Hypo the hypocentral distance, H the
    hypocentral depth and GAMMA the intrinsic attenuation coefficient.
    The endog data are the epicentral distance and the exog data the associated
    epicentral distance.
    """
    def __init__(self, ObsBin_plus, Beta, Gamma):
        """
        :param ObsBin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 'I', 'Depi', 'Depth', 'Io',  'StdI 'and 'eqStd'
                        which are respectively the binned intensity, the associated epicentral distance,
                        the associated depth, the associated epicentral intensity,
                        the associated standard deviation and the associated
                        inverse square root of the weights used in the inversion.
        :param Beta: the geometric attenuation coefficient of the Koveslighety equation
        :param Gamma: the intresic attenuation coefficient of the Koveslighety equation
        :type ObsBin_plus: pandas.DataFrame
        :type Beta: float
        :type Gamma: float
        """
        #Variable ObsBin_plus doit contenir les Obsbin de tous les evts de la calibration, avec une colonne depth en plus
        self.ObsBin_plus = ObsBin_plus
        self.beta = Beta
        self.gamma = Gamma
        
    def EMIPE_beta(self, X, beta):
        """
        Function used to inverse the geometric attenuation coefficient
        :param X: matrix that contains epicentral distance, depth and epicentral
                  intensity associated to the binned intensity
        :param beta: geometric attenuation coefficient
        :type X: numpy.array
        :type beta: float
        """
        Depi, depths, I0s = X
        try:
            logterm = np.sqrt(Depi**2+depths**2)/depths
        except AttributeError:
            tmp = (Depi**2+depths**2)**0.5/depths
            logterm = np.array([])
            for tt in tmp:
                logterm = np.append(logterm, tt)
            
        I = I0s + beta*np.log10(logterm)
        I = np.array(I, dtype=float)
        return I
    
    
    def EMIPE_beta_gamma(self, X, beta, gamma):
        """
        Function used to inverse the attenuation coefficients
        :param X: matrix that contains epicentral distance, depth and epicentral
                  intensity associated to the binned intensity
        :param beta: geometric attenuation coefficient
        :param gamma: intrisic attenuation coefficient
        :type X: numpy.array
        :type gamma: float
        :type beta: float
        """
        Depi, depths, I0s = X
        I = I0s + beta*np.log10(np.sqrt(Depi**2+depths**2)/depths) + gamma*(np.sqrt(Depi**2+depths**2)-depths)
        return I.astype(float)
        
    def do_wls_beta(self):
        """
        Function used to launch geometric attenuation coefficient inversion
        :return: [popt, pcov] list with popt a (1, 1) shape array containing
                 the inverted beta. pcov is a 2-D array and 
                 the estimated covariance of popt. The diagonals provide the
                 variance of the parameter estimate. 
                 To compute one standard deviation errors on the parameters use
                 perr = np.sqrt(np.diag(pcov)).
                 In the case of a weighting scheme different as 'Ponderation dI',
                 please use the do_wls_beta_std() function to retrieve
                 the covariance matrix based on the intensity data standard
                 deviation just after inverting beta with the present function.
        """
        Ibin = self.ObsBin_plus['I'].values.astype(float)
        Depi = self.ObsBin_plus['Depi'].values.astype(float)
        depths = self.ObsBin_plus['Depth'].values.astype(float)
        I0s = self.ObsBin_plus['Io'].values.astype(float)
        X = [np.array(Depi), np.array(depths), np.array(I0s)]
        resBeta = curve_fit(self.EMIPE_beta, X, Ibin, p0=self.beta,
                                  sigma=self.ObsBin_plus['eqStd'].values.astype(float), absolute_sigma=True,
                                  xtol=1e-3)
        return resBeta
    
    def do_wls_beta_std(self):
        """
        Function used to compute the covariance matrix associated to the 
        geomteric attenuation coefficient based on the standard deviations associated
        to the intensity data.
        :return: [popt, pcov] list with popt a (1, 1) shape array containing
                 the inverted beta. pcov is a 2-D array and 
                 the estimated covariance of popt. The diagonals provide the
                 variance of the parameter estimate. 
                 To compute one standard deviation errors on the parameters use
                 perr = np.sqrt(np.diag(pcov)).
        """
        Ibin = self.ObsBin_plus['I'].values.astype(float)
        Depi = self.ObsBin_plus['Depi'].values.astype(float)
        depths = self.ObsBin_plus['Depth'].values.astype(float)
        I0s = self.ObsBin_plus['Io'].values.astype(float)
        X = [np.array(Depi, dtype=float), np.array(depths, dtype=float), np.array(I0s, dtype=float)]
        resBeta = curve_fit(self.EMIPE_beta, X, Ibin, p0=self.beta, bounds=(self.beta-0.0001, self.beta+0.0001),
                                  sigma=self.ObsBin_plus['StdI'].values.astype(float), absolute_sigma=True,
                                  xtol=1e-3)
        # resBeta = curve_fit(self.EMIPE_beta, X, Ibin, p0=self.beta, 
        #                           sigma=self.ObsBin_plus['StdI'].values, absolute_sigma=True,
        #                           xtol=1e-3)
        return resBeta
    
    
    def do_wls_betagamma(self):
        """
        Function used to launch geometric and intrisic attenuation coefficients inversion
        
        :return: [popt, pcov] list with popt a (2, 1) shape array containing
                 the inverted beta (popt[0]) and gamma (popt[1]). pcov is a 2-D array and 
                 the estimated covariance of popt. The diagonals provide the
                 variance of the parameter estimate. 
                 To compute one standard deviation errors on the parameters use
                 perr = np.sqrt(np.diag(pcov)).
                 In the case of a weighting scheme different as 'Ponderation dI',
                 please use the do_wls_betagamma_std() function to retrieve
                 the covariance matrix based on the intensity data standard
                 deviation just after inverting beta with the present function.
        """
        Ibin = self.ObsBin_plus['I'].values.astype(float)
        Depi = self.ObsBin_plus['Depi'].values.astype(float)
        depths = self.ObsBin_plus['Depth'].values.astype(float)
        I0s = self.ObsBin_plus['Io'].values.astype(float)
        X = [Depi, depths, I0s]
        resBetaGamma = curve_fit(self.EMIPE_beta_gamma, X, Ibin, p0=[self.beta, self.gamma],
                                  bounds=([-np.inf, -np.inf], [np.inf, 0]),
                                  sigma=self.ObsBin_plus['eqStd'].values.astype(float), absolute_sigma=True,
                                  xtol=1e-3)
        return resBetaGamma
    
    def do_wls_betagamma_std(self):
        """
        Function used to compute the covariance matrix associated to the 
        geomteric and intrinsic attenuation coefficient based on the standard deviations associated
        to the intensity data.
        :return: [popt, pcov] list with popt a (2, 1) shape array containing
                 the inverted beta (popt[0]) and gamma (popt[1]).
                 pcov is a 2-D array and the estimated covariance of popt.
                 The diagonals provide the variance of the parameter estimate. 
                 To compute one standard deviation errors on the parameters use
                 perr = np.sqrt(np.diag(pcov)).
        """
        
        Ibin = self.ObsBin_plus['I'].values.astype(float)
        Depi = self.ObsBin_plus['Depi'].values.astype(float)
        depths = self.ObsBin_plus['Depth'].values.astype(float)
        I0s = self.ObsBin_plus['Io'].values.astype(float)
        X = [Depi, depths, I0s]
        resBetaGamma = curve_fit(self.EMIPE_beta_gamma, X, Ibin, p0=[self.beta, self.gamma],
                                  bounds=([self.beta-0.0001, self.gamma-1e-6], [self.beta+0.0001, self.gamma+1e-6]),
                                  sigma=self.ObsBin_plus['StdI'].values.astype(float), absolute_sigma=True,
                                  xtol=1e-3)
        return resBetaGamma


class WLSIC_oneEvt():
    """
    Set of functions that inverse depth and magnitude
    from macroseismic data for a given earthquake.
    The mathematical formulation is :
        
        I = C1 + C2.Mag + BETA.log10(Hypo) + GAMMA.Hypo
    
    where I is the intensity value, C1 and C2 the magnitude coefficients, M the
    magnitude, BETA the geometric attenuation coefficient, Hypo the hypocentral
    distance and GAMMA the intrisic attenuation coefficient.
    The endog data are the epicentral distance and the exog data the associated
    epicentral distance.
    """
    def __init__(self, ObsBinn, depth, mag, Beta, Gamma, C1, C2):
        """
        :param Obsbinn: dataframe with the binned intensity data for one earthquake.
                        This dataframe should at least have have the following
                        columns : 'I', 'Depi', 'StdI', which are respectively
                        the binned intensity, the associated epicentral distance
                        and the associated standard deviation.
        :param depth: hypocenter's depth of the considered earthquake. In the case
                      of depth inversion, this value is the initial depth value
        :param mag: magnitude of the considered earthquake. In the case
                      of magnitude inversion, this value is the initial magnitude value
        :param Beta: the geometric attenuation coefficient of the Koveslighety equation
        :param Gamma: the intresic attenuation coefficient of the Koveslighety equation
        :param C1: first magnitude coefficient
        :param C2: second magnitude coefficient
        :type Obsbinn: pandas.DataFrame
        :type depth: float
        :type mag: float
        :type Beta: float
        :type Gamma: float
        :type C1: float
        :type C2: float 
        """
        self.beta = Beta
        self.gamma = Gamma
        self.Obsbin = ObsBinn
        self.depth = depth
        self.C1 = C1
        self.C2 = C2
        self.mag = mag
        
    def EMIPE_H(self, Depi, H):
        """
        Function used to inverse depth
        :param Depi: epicentral distances associated to the binned intensity data
        :param H: hypocenter's depth of the considered earthquake.
                      Should be greater than 0
        :type Depi: numpy.array
        :type H: float
        """
        I = self.C1 + self.C2*self.mag + self.beta*np.log10(np.sqrt(Depi**2+H**2))+self.gamma*np.sqrt(Depi**2+H**2)
        return I
    
    def EMIPE_M(self, Depi, M):
        """
        Function used to inverse magnitude
        :param Depi: epicentral distances associated to the binned intensity data
        :param M: magnitude of the considered earthquake.
        :type Depi: numpy.array
        :type M: float
        """
        I = self.C1 + self.C2*M+ self.beta*np.log10(np.sqrt(Depi**2+self.depth**2))+self.gamma*np.sqrt(Depi**2+self.depth**2)
        return I

    def EMIPE_JACdH(self, Depi, H):
        """
        Jacobian function used to inverse depth
        :param Depi: epicentral distances associated to the binned intensity data
        :param H: hypocenter's depth of the considered earthquake.
                      Should be greater than 0
        :type Depi: numpy.array
        :type H: float
        """
        Hypo = np.sqrt(Depi**2+H**2)
        tmpValue = H/Hypo
        g = (tmpValue)*((self.beta/(np.log(10)*Hypo))+self.gamma)
        return g.reshape(len(Depi),1)

    def EMIPE_JACdM(self, Depi, H):
        """
        Jacobian function to inverse magnitude
        :param Depi: epicentral distances associated to the binned intensity data
        :param M: magnitude of the considered earthquake.
        :type Depi: numpy.array
        :type M: float
        """        
        g = self.C2*np.ones(len(Depi))
        return g.reshape(len(Depi),1)

    def do_wlsic_depth(self, depth_inf, depth_sup):
        """
        Function used to launch depth inversion within limits.
        
        :param depth_inf: lower depth limit of inversion
        :param depth_sup: upper depth limit of inversion
        :type depth_inf: float
        :type depth_sup: float
        
        :return: [popt, pcov] list with popt a (1, 1) shape array containing
                the inverted depth. pcov is a 2-D array and 
                the estimated covariance of popt. The diagonals provide the
                variance of the parameter estimate. 
                To compute one standard deviation errors on the parameters use
                perr = np.sqrt(np.diag(pcov)).
        """
        Ibin = self.Obsbin['I'].values.astype(float)
        Depi = self.Obsbin['Depi'].values.astype(float)
        if self.depth<depth_inf:
            self.depth = depth_inf+0.1
        if self.depth>depth_sup:
            self.depth = depth_sup-0.1

        resH = curve_fit(self.EMIPE_H, Depi, Ibin, p0=self.depth,
                                 jac= self.EMIPE_JACdH, bounds=(depth_inf, depth_sup),
                                 sigma=self.Obsbin['StdI'].values.astype(float), absolute_sigma=True,
                                 ftol=5e-2)
        return resH
    
    def do_wlsic_depthM_std(self):
        """
        Function used to compute the covariance matrix associated to the 
        inverted depth based on the standard deviations associated
        to the intensity data.
        
        :return: [popt, pcov] list with popt a (1, 1) shape array containing
                the inverted depth. pcov is a 2-D array and 
                the estimated covariance of popt. The diagonals provide the
                variance of the parameter estimate. 
                To compute one standard deviation errors on the parameters use
                perr = np.sqrt(np.diag(pcov)).
        """
        Ibin = self.Obsbin['I'].values
        Depi = self.Obsbin['Depi'].values
        resH = curve_fit(self.EMIPE_HM, Depi, Ibin, p0=self.depth,
                                 jac= self.EMIPE_JACdHM, bounds=(self.depth-0.01, self.depth+0.01),
                                 sigma=self.Obsbin['StdI'].values, absolute_sigma=True,
                                 xtol=5e-2)
        return resH

    def do_wls_M(self):
        """
        Function used to launch magnitude inversion.
        :return: [popt, pcov] list with popt a (1, 1) shape array containing
                the inverted magnitude. pcov is a 2-D array and 
                the estimated covariance of popt. The diagonals provide the
                variance of the parameter estimate. 
                To compute one standard deviation errors on the parameters use
                perr = np.sqrt(np.diag(pcov)).

        """
        Ibin = self.Obsbin['I'].values
        Depi = self.Obsbin['Depi'].values
        resM = curve_fit(self.EMIPE_M, Depi, Ibin, p0=self.mag,
                                 jac= self.EMIPE_JACdM, 
                                 sigma=self.Obsbin['StdI'].values, absolute_sigma=True,
                                 xtol=1e-3)
        return resM
    
    def do_wls_M_std(self):
        """
        Function used to compute the covariance matrix associated to the 
        inverted magnitude based on the standard deviations associated
        to the intensity data.
        
        :return: [popt, pcov] list with popt a (1, 1) shape array containing
                the inverted magnitude. pcov is a 2-D array and 
                the estimated covariance of popt. The diagonals provide the
                variance of the parameter estimate. 
                To compute one standard deviation errors on the parameters use
                perr = np.sqrt(np.diag(pcov)).
        """
        Ibin = self.Obsbin['I'].values
        Depi = self.Obsbin['Depi'].values
        resM = curve_fit(self.EMIPE_M, Depi, Ibin, p0=self.mag, bounds=(self.mag-0.001, self.mag+0.001),
                                 jac= self.EMIPE_JACdM, 
                                 sigma=self.Obsbin['StdI'].values, absolute_sigma=True,
                                 xtol=1e-3)
        return resM


class WLS():
    """
    Set of functions that inverse the coefficient of the following equation
    from macroseismic data of a calibration dataset.
    The mathematical formulation is :
        
        I = C1 + C2.Mag + BETA.log10(Hypo) + GAMMA.Hypo
    
    where I is the intensity value, C1 and C2 the magnitude coefficients, M the
    magnitude, BETA the geometric attenuation coefficient, Hypo the hypocentral
    distance and GAMMA the intrisic attenuation coefficient.
    """
    def __init__(self, ObsBin_plus, C1, C2, Beta, Gamma):
        """
        :param ObsBin_plus: dataframe with the binned intensity data for all calibration earthquakes.
                        This dataframe should at least have have the following
                        columns : 'I', 'Depi', 'Depth', 'Mag', 'X', 'eqStd' and 'eqStdM'
                        which are respectively the binned intensity, the associated epicentral distance,
                        the associated depth, the associated magnitude, the X parameter used in C1/C2 inversion,
                        the associated inverse square root of the weights used in the inversion
                        of Gamma and the associated square root of the weights used in the inversion
                        of C1/C2, C1/C2/Beta, C1/C2/Beta/Gamma.
        :param Beta: the geometric attenuation coefficient
        :param Gamma: the intresic attenuation coefficient
        :param C1: first magnitude coefficient
        :param C2: second magnitude coefficient
        :type ObsBin_plus: pandas.DataFrame
        :type Beta: float
        :type Gamma: float
        :type C1: float
        :type C2: float
        """
        self.Obsbin_plus = ObsBin_plus
        self.C1 = C1
        self.C2 = C2
        self.Cb = -C1/C2
        self.Ca = 1/C2
        self.beta = Beta
        self.gamma = Gamma

    def EMIPE_C1C2BetaH(self, X, C1, C2, Beta, H1, H2, H3, H4, H5, H6, H7, H8,
                        H9, H10, H11, H12, H13, H14, H15, H16, H17, H18, H19, H20,
                        H21, H22, H23, H24, H25, H26, H27, H28, H29, H30, H31, H32):
        """
        Function used to inverse the magnitude coefficients and the geometric attenuation
        coefficient
        :param X: matrix that contains magnitude, epicentral distance and depth
                  intensity associated to the binned intensity
        :param C1: first magnitude coefficient
        :param C2: second magnitude coefficient
        :param Beta: geometric attenuation coefficient
        :type X: numpy.array
        :type Beta: float
        :type C1: float
        :type C2: float
        """
        mags, depi, id_evid = X
        liste_evid = np.unique(id_evid)
        #ah --> (n, len(Depi)) array avec n le nombre de EQ. Chaque ligne contient
        #des 1 et des 0 et correspond a un EQ. 1 est attribue aux indices de
        # obsbin_plus.EVID ==evid concerne.
        aH = np.array([])
        for compt, evid in enumerate(liste_evid):
            ind = (id_evid == evid)
            zeros = np.zeros(len(mags))
            zeros[ind] = 1
            try:
                aH = np.vstack((aH, zeros))
            except ValueError:
                 aH = np.concatenate((aH, zeros))
        if len(liste_evid)<32:
            len_noevt = 32 - len(liste_evid)
            for compt in range(len_noevt):
                zeros = np.zeros(len(self.Obsbin_plus.EVID))
                aH = np.vstack((aH, zeros))
        #ah = X[2:][0]
        #H --> (n, len(Depi)) array avec n le nombre de EQ, chaque ligne contient obsbin_plus.Depth
        H = np.vstack(([H1]*len(depi), [H2]*len(depi), [H3]*len(depi), [H4]*len(depi),
                       [H5]*len(depi), [H6]*len(depi), [H7]*len(depi), [H8]*len(depi),
                       [H9]*len(depi), [H10]*len(depi), [H11]*len(depi), [H12]*len(depi),
                       [H13]*len(depi), [H14]*len(depi), [H15]*len(depi), [H16]*len(depi),
                       [H17]*len(depi), [H18]*len(depi), [H19]*len(depi), [H20]*len(depi),
                       [H21]*len(depi), [H22]*len(depi), [H23]*len(depi), [H24]*len(depi),
                       [H25]*len(depi), [H26]*len(depi), [H27]*len(depi), [H28]*len(depi),
                       [H29]*len(depi), [H30]*len(depi), [H31]*len(depi), [H32]*len(depi),
                       ))
        hypos = np.sqrt(depi**2 + (H*aH).sum(axis=0)**2)
        I = C1 + C2*mags + Beta*np.log10(hypos)
        return I
    
    def EMIPE_C1C2BetaH_2regC1beta(self, X, C1a, C1b, C2, Betaa, Betab, H1, H2, H3, H4, H5, H6, H7, H8,
                                   H9, H10, H11, H12, H13, H14, H15, H16, H17, H18, H19, H20,
                                   H21, H22, H23, H24, H25, H26, H27, H28, H29, H30, H31, H32,
                                   H33, H34, H35, H36, H37, H38, H39, H40, H41, H42, H43,
                                   H44, H45, H46, H47, H48, H49, H50, H51, H52, H53, H54,
                                   H55, H56, H57, H58, H59, H60, H61, H62, H63, H64, H65):
        """
        Function used to inverse the magnitude coefficients and the geometric attenuation
        coefficient
        :param X: matrix that contains magnitude, epicentral distance and depth
                  intensity associated to the binned intensity
        :param C1: first magnitude coefficient
        :param C2: second magnitude coefficient
        :param Beta: geometric attenuation coefficient
        :type X: numpy.array
        :type Beta: float
        :type C1: float
        :type C2: float
        """
        mags, depi, id_evid, id_region = X
        liste_evid = np.unique(id_evid)
        liste_region = np.unique(id_region)
        #ah --> (n, len(Depi)) array avec n le nombre de EQ. Chaque ligne contient
        #des 1 et des 0 et correspond a un EQ. 1 est attribue aux indices de
        # obsbin_plus.EVID ==evid concerne.
        aH = np.array([])
        for compt, evid in enumerate(liste_evid):
            ind = (id_evid == evid)
            zeros = np.zeros(len(mags))
            zeros[ind] = 1
            try:
                aH = np.vstack((aH, zeros))
            except ValueError:
                 aH = np.concatenate((aH, zeros))
        if len(liste_evid)<65:
            len_noevt = 65 - len(liste_evid)
            for compt in range(len_noevt):
                zeros = np.zeros(len(self.Obsbin_plus.EVID))
                aH = np.vstack((aH, zeros))
                
        if len(liste_region)!=2:
            raise ValueError('Number of region should be 2')
        aregion = np.array([])
        for compt, region in enumerate(liste_region):
            ind = (id_region == region)
            zeros = np.zeros(len(mags))
            zeros[ind] = 1
            try:
                aregion = np.vstack((aregion, zeros))
            except ValueError:
                 aregion = np.concatenate((aregion, zeros))

        C1 = np.vstack(([C1a]*len(mags), [C1b]*len(mags)))
        Beta = np.vstack(([Betaa]*len(mags), [Betab]*len(mags)))       
        #ah = X[2:][0]
        #H --> (n, len(Depi)) array avec n le nombre de EQ, chaque ligne contient obsbin_plus.Depth
        H = np.vstack(([H1]*len(depi), [H2]*len(depi), [H3]*len(depi), [H4]*len(depi),
                       [H5]*len(depi), [H6]*len(depi), [H7]*len(depi), [H8]*len(depi),
                       [H9]*len(depi), [H10]*len(depi), [H11]*len(depi), [H12]*len(depi),
                       [H13]*len(depi), [H14]*len(depi), [H15]*len(depi), [H16]*len(depi),
                       [H17]*len(depi), [H18]*len(depi), [H19]*len(depi), [H20]*len(depi),
                       [H21]*len(depi), [H22]*len(depi), [H23]*len(depi), [H24]*len(depi),
                       [H25]*len(depi), [H26]*len(depi), [H27]*len(depi), [H28]*len(depi),
                       [H29]*len(depi), [H30]*len(depi), [H31]*len(depi), [H32]*len(depi),
                       [H33]*len(depi), [H34]*len(depi), [H35]*len(depi), [H36]*len(depi),
                       [H37]*len(depi), [H38]*len(depi), [H39]*len(depi), [H40]*len(depi),
                       [H41]*len(depi), [H42]*len(depi), [H43]*len(depi), [H44]*len(depi),
                       [H45]*len(depi), [H46]*len(depi), [H47]*len(depi), [H48]*len(depi),
                       [H49]*len(depi), [H50]*len(depi), [H51]*len(depi), [H52]*len(depi),
                       [H53]*len(depi), [H54]*len(depi), [H55]*len(depi), [H56]*len(depi),
                       [H57]*len(depi), [H58]*len(depi), [H59]*len(depi), [H60]*len(depi),
                       [H61]*len(depi), [H62]*len(depi), [H63]*len(depi), [H64]*len(depi),
                       [H65]*len(depi),
                       ))
        hypos = np.sqrt(depi**2 + (H*aH).sum(axis=0)**2)
        I = (C1*aregion).sum(axis=0) + C2*mags + ((Beta*aregion).sum(axis=0))*np.log10(hypos)
        return I
    
    
    def do_wls_C1C2BetaH_2regC1beta(self, 
                         ftol=5e-3, xtol=1e-8, max_nfev=200,
                         C1a=1, C1b=1, betaa=-3.0, betab=-3.0):
        """
        Function used to launch the inversion of all coefficients, except the intrinsic
        attenuation coefficient gamma.
        
         return: [popt, pcov] list with popt a (3, 1) shape array containing
                the inverted coefficient with popt[0] the C1 coefficient,
                popt[1] the C2 coefficient and popt[2] the beta coefficient. pcov is a 2-D array and 
                the estimated covariance of popt. The diagonals provide the
                variance of the parameter estimate. 
                To compute one standard deviation errors on the parameters use
                perr = np.sqrt(np.diag(pcov)). pcov values are not accurate with
                the eqStdM as sigma. A function has to be developped to compute
                accurate pcov
        """
        #print(self.Obsbin_plus.columns)
        liste_evid = np.unique(self.Obsbin_plus.EVID.values)
        depths = np.zeros(65)
        Hmin = np.zeros(65)
        Hmax = np.zeros(65)
        id_evid = np.zeros(len(self.Obsbin_plus.EVID))
        
        for compt, evid in enumerate(liste_evid):
            ind = (self.Obsbin_plus.EVID == evid)
            depth = self.Obsbin_plus[ind]['Depth'].values[0]
            hmin = self.Obsbin_plus[ind]['Hmin'].values[0]
            hmax = self.Obsbin_plus[ind]['Hmax'].values[0]
            depths[compt] = depth
            Hmin[compt] = hmin
            Hmax[compt] = hmax
            id_evid[ind] = compt
        Hmax[Hmax==0] = 0.01

        X = [self.Obsbin_plus.Mag.values.astype(float),
             self.Obsbin_plus.Depi.values.astype(float),
             id_evid,
             self.Obsbin_plus.RegID.values.astype(float)]
        
        Ibin = self.Obsbin_plus['I'].values.astype(float)
        #print(Ibin.dtype)
        
        sigma = self.Obsbin_plus['eqStdM'].values.astype(float)
        # if sigma.dtype is not np.dtype(np.float64):
        #     raise TypeError("sigma.dtype is not float")

        p0 = np.append(np.array([C1a, C1b, self.C2, betaa, betab]),
                       depths)
        bounds_inf = np.append(np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
                       Hmin)
        bounds_sup = np.append(np.array([np.inf, np.inf, np.inf, np.inf, np.inf]),
                       Hmax)
        C1C2BetaH = curve_fit(self.EMIPE_C1C2BetaH_2regC1beta, X, Ibin, 
                             p0=p0,
                             bounds=(bounds_inf, bounds_sup),
                             sigma=sigma,
                             absolute_sigma=False,
                             ftol=ftol,
                             xtol=xtol,
                             max_nfev=max_nfev)
        return C1C2BetaH

    def EMIPE_JACdC1C2BetaH(self, X, C1, C2, beta, H):
        """
        Jacobian function used to inverse depth
        :param Depi: epicentral distances associated to the binned intensity data
        :param H: hypocenter's depth of the considered earthquake.
                      Should be greater than 0
        :type Depi: numpy.array
        :type H: float
        """
        Depi, Mag = X[:2]
        aH = X[2:][0]
        Hypo = np.sqrt(Depi**2+(aH*H).sum(axis=0)**2)
        tmpValue = (aH*H).sum(axis=0)/Hypo
        gC1 = np.ones(len(Depi))
        gC2 = Mag
        gbeta = np.log10(Hypo)
        #gH = (tmpValue)*((self.beta/(np.log(10)*Hypo))+self.gamma)
        GH = np.array([])
        for ahh, hh in zip(aH, H):
            Hypo = np.sqrt(Depi**2+ahh*hh**2)
            tmpValue = ahh*hh/Hypo
            gH = (tmpValue)*((beta/(np.log(10)*Hypo)))
            gH = np.nan_to_num(gH)
            #GH = np.tile(gH, (len(Depi),1))
            try:
                GH = np.vstack((GH, gH))
            except ValueError:
                GH = np.concatenate((GH, gH))
        #g = np.array([gC1, gC2, gbeta, GH])
        g = np.vstack((gC1, gC2))
        g = np.vstack((g, gbeta))
        g = np.vstack((g, GH))
        return g.reshape(len(Depi),3+len(Depi))
    
    def do_wls_C1C2BetaH(self, sigma='none',
                         ftol=5e-3, xtol=1e-8, max_nfev=200):
        """
        Function used to launch the inversion of all coefficients, except the intrinsic
        attenuation coefficient gamma.
        
         return: [popt, pcov] list with popt a (3, 1) shape array containing
                the inverted coefficient with popt[0] the C1 coefficient,
                popt[1] the C2 coefficient and popt[2] the beta coefficient. pcov is a 2-D array and 
                the estimated covariance of popt. The diagonals provide the
                variance of the parameter estimate. 
                To compute one standard deviation errors on the parameters use
                perr = np.sqrt(np.diag(pcov)). pcov values are not accurate with
                the eqStdM as sigma. A function has to be developped to compute
                accurate pcov
        """
        #print(self.Obsbin_plus.columns)
        liste_evid = np.unique(self.Obsbin_plus.EVID.values)
        depths = np.zeros(32)
        Hmin = np.zeros(32)
        Hmax = np.zeros(32)
        id_evid = np.zeros(len(self.Obsbin_plus.EVID))
        for compt, evid in enumerate(liste_evid):
            ind = (self.Obsbin_plus.EVID == evid)
            depth = self.Obsbin_plus[ind]['Depth'].values[0]
            hmin = self.Obsbin_plus[ind]['Hmin'].values[0]
            hmax = self.Obsbin_plus[ind]['Hmax'].values[0]
            depths[compt] = depth
            Hmin[compt] = hmin
            Hmax[compt] = hmax
            id_evid[ind] = compt
        Hmax[Hmax==0] = 0.01

        X = [self.Obsbin_plus.Mag.values.astype(float),
             self.Obsbin_plus.Depi.values.astype(float),
             id_evid]
        
        Ibin = self.Obsbin_plus['I'].values.astype(float)
        print(Ibin.dtype)
        if sigma == 'none':
            sigma = self.Obsbin_plus['eqStdM'].values.astype(float)
        # if sigma.dtype is not np.dtype(np.float64):
        #     raise TypeError("sigma.dtype is not float")
        p0 = np.append(np.array([self.C1, self.C2, self.beta]),
                       depths)
        bounds_inf = np.append(np.array([-np.inf, -np.inf, -np.inf]),
                       Hmin)
        bounds_sup = np.append(np.array([np.inf, np.inf, np.inf]),
                       Hmax)
        C1C2BetaH = curve_fit(self.EMIPE_C1C2BetaH, X, Ibin, 
                             p0=p0,
                             bounds=(bounds_inf, bounds_sup),
                             sigma=sigma,
                             absolute_sigma=True,
                             ftol=ftol,
                             xtol=xtol,
                             max_nfev=max_nfev)
        return C1C2BetaH
    
    def EMIPE_C1C2BetaGammaH(self, X, C1, C2, Beta, Gamma, H1, H2, H3, H4, H5, H6, H7, H8,
                        H9, H10, H11, H12, H13, H14, H15, H16, H17, H18, H19, H20,
                        H21, H22, H23, H24, H25, H26, H27, H28, H29, H30, H31, H32):
        """
        Function used to inverse the magnitude coefficients and the attenuation
        coefficients
        :param X: matrix that contains magnitude, epicentral distance and depth
                  intensity associated to the binned intensity
        :param C1: first magnitude coefficient
        :param C2: second magnitude coefficient
        :param Beta: geometric attenuation coefficient
        :param Gamma: intresic attenuation coefficient
        :type X: numpy.array
        :type Beta: float
        :type C1: float
        :type C2: float
        """
        mags, depi, id_evid = X
        liste_evid = np.unique(id_evid)
        #ah --> (n, len(Depi)) array avec n le nombre de EQ. Chaque ligne contient
        #des 1 et des 0 et correspond a un EQ. 1 est attribue aux indices de
        # obsbin_plus.EVID ==evid concerne.
        aH = np.array([])
        for compt, evid in enumerate(liste_evid):
            ind = (id_evid == evid)
            zeros = np.zeros(len(mags))
            zeros[ind] = 1
            try:
                aH = np.vstack((aH, zeros))
            except ValueError:
                 aH = np.concatenate((aH, zeros))
        if len(liste_evid)<31:
            len_noevt = 31 - len(liste_evid)
            for compt in range(len_noevt):
                zeros = np.zeros(len(self.Obsbin_plus.EVID))
                aH = np.vstack((aH, zeros))
        #ah = X[2:][0]
        #H --> (n, len(Depi)) array avec n le nombre de EQ, chaque ligne contient obsbin_plus.Depth
        H = np.vstack(([H1]*len(depi), [H2]*len(depi), [H3]*len(depi), [H4]*len(depi),
                       [H5]*len(depi), [H6]*len(depi), [H7]*len(depi), [H8]*len(depi),
                       [H9]*len(depi), [H10]*len(depi), [H11]*len(depi), [H12]*len(depi),
                       [H13]*len(depi), [H14]*len(depi), [H15]*len(depi), [H16]*len(depi),
                       [H17]*len(depi), [H18]*len(depi), [H19]*len(depi), [H20]*len(depi),
                       [H21]*len(depi), [H22]*len(depi), [H23]*len(depi), [H24]*len(depi),
                       [H25]*len(depi), [H26]*len(depi), [H27]*len(depi), [H28]*len(depi),
                       [H29]*len(depi), [H30]*len(depi), [H31]*len(depi), [H32]*len(depi),
                       ))
        
        
        hypos = np.sqrt(depi**2 + (H*aH).sum(axis=0)**2)
        I = C1 + C2*mags + Beta*np.log10(hypos) + Gamma*hypos
        return I
    
    def do_wls_C1C2BetaGammaH(self, sigma='none',
                         ftol=5e-3, xtol=1e-8, max_nfev=200):
        """
        Function used to launch the inversion of all coefficients. Depth of 
        each earthquake is also inverted.
        
         return: [popt, pcov] list with popt a (3, 1) shape array containing
                the inverted coefficient with popt[0] the C1 coefficient,
                popt[1] the C2 coefficient and popt[2] the beta coefficient. pcov is a 2-D array and 
                the estimated covariance of popt. The diagonals provide the
                variance of the parameter estimate. 
                To compute one standard deviation errors on the parameters use
                perr = np.sqrt(np.diag(pcov)). pcov values are not accurate with
                the eqStdM as sigma. A function has to be developped to compute
                accurate pcov
        """
        #print(self.Obsbin_plus.columns)
        liste_evid = np.unique(self.Obsbin_plus.EVID.values)
        depths = np.zeros(31)
        Hmin = np.zeros(31)
        Hmax = np.zeros(31)
        id_evid = np.zeros(len(self.Obsbin_plus.EVID))
        for compt, evid in enumerate(liste_evid):
            ind = (self.Obsbin_plus.EVID == evid)
            depth = self.Obsbin_plus[ind]['Depth'].values[0]
            hmin = self.Obsbin_plus[ind]['Hmin'].values[0]
            hmax = self.Obsbin_plus[ind]['Hmax'].values[0]
            depths[compt] = depth
            Hmin[compt] = hmin
            Hmax[compt] = hmax
            id_evid[ind] = compt
        Hmax[Hmax==0] = 0.01

        X = [self.Obsbin_plus.Mag.values.astype(float),
             self.Obsbin_plus.Depi.values.astype(float),
             id_evid]
        
        Ibin = self.Obsbin_plus['I'].values.astype(float)
        print(Ibin.dtype)
        if sigma == 'none':
            sigma = self.Obsbin_plus['eqStdM'].values.astype(float)
        # if sigma.dtype is not np.dtype(np.float64):
        #     raise TypeError("sigma.dtype is not float")
        p0 = np.append(np.array([self.C1, self.C2, self.beta, self.gamma]),
                       depths)
        bounds_inf = np.append(np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
                       Hmin)
        bounds_sup = np.append(np.array([np.inf, np.inf, np.inf, 0]),
                       Hmax)
        C1C2BetaGammaH = curve_fit(self.EMIPE_C1C2BetaGammaH, X, Ibin, 
                             p0=p0,
                             bounds=(bounds_inf, bounds_sup),
                             sigma=sigma,
                             absolute_sigma=True,
                             ftol=ftol,
                             xtol=xtol,
                             max_nfev=max_nfev)
        return C1C2BetaGammaH

    def do_wls_C1C2BetaH_std(self, sigma):
        """
        Function used to compute the covariance matrix associated to the 
        inverted C1,, C2, Beta and depths based on the standard deviations associated
        to the intensity data.

        Parameters
        ----------
        sigma : TYPE
            DESCRIPTION.

        Returns
        -------
        Array
            Covariance matrix pour the inverted C1,, C2, Beta and depths, based
            on the standard deviations associated to the intensity data.
            The output is the same as pcov output of scipy.optimize.curve_fit,
            i.e. the estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
            To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov))

        """
        aH = np.array([])
        liste_evid = np.unique(self.Obsbin_plus.EVID.values)
        depths = np.zeros(31)
        Hmin = np.zeros(31)
        Hmax = np.zeros(31)
        for compt, evid in enumerate(liste_evid):
            ind = (self.Obsbin_plus.EVID == evid)
            depth = self.Obsbin_plus[ind]['Depth'].values[0]
            hmin = self.Obsbin_plus[ind]['Hmin'].values[0]
            hmax = self.Obsbin_plus[ind]['Hmax'].values[0]
            depths[compt] = depth
            Hmin[compt] = hmin
            Hmax[compt] = hmax
            zeros = np.zeros(len(self.Obsbin_plus.EVID))
            zeros[ind] = 1
            try:
                aH = np.vstack((aH, zeros))
            except ValueError:
                 aH = np.concatenate((aH, zeros))
    
        X = [self.Obsbin_plus.Mag.values,
             self.Obsbin_plus.Depi.values,
             aH]
        Ibin = self.Obsbin_plus['I'].values
        
        p0 = np.append(np.array([self.C1, self.C2, self.beta]),
                       depths)
        bounds_inf = np.append(np.array([-np.inf, -np.inf, -np.inf]),
                       Hmin)
        bounds_sup = np.append(np.array([np.inf, np.inf, np.inf]),
                       Hmax)
        C1C2BetaH = curve_fit(self.EMIPE_C1C2BetaH, X, Ibin, 
                             p0=p0,
                             bounds=(bounds_inf, bounds_sup),
                             sigma=sigma,
                             absolute_sigma=True,
                             xtol=1e-3)
        return C1C2BetaH[1]
    
    def compute_2Dsigma(self, eta, col='eqStdM'):
        """
        Compute a 2D sigma. The diagonal if equal to the column col input. The other
        elements of the matrix are filled with the eat input.
        
        Parameters
        ----------
        eta : float
            DESCRIPTION.
        col : str, optional
            Name of the column where are stored the standard deviation used for the
            diagonal of the 2D sigma matrix. The default is 'eqStdM'.
        Returns
        -------
        sigma : 2D array
            2D sigma corresponding to the sigma associated to the I column of the Obsbin_plus
            dataframe in the inversion process 
        """
        sigma = np.diag(self.Obsbin_plus[col].values)
        sigma[sigma==0] = eta
        return sigma
    
    
    def C1_4regionC2_EMIPE(self, X, C1a, C1b, C1c, C1d, C2):
        """
        Function used to inverse the C1 and C2 coefficients.The C1 coefficient
        is estimated for 4 different regions. 

        Parameters
        ----------
        X : array
            Array that contains the magnitudes of the calibration earthquakes and the corresponding 
            region ID of the earthquakes. The array should correspond to the columns of the Obsbin_plus
            dataframe.
        C1a : float
            C1 coefficient for the first region.
        C1b : float
            C1 coefficient for the second region.
        C1c : float
            C1 coefficient for the third region.
        C1d : float
            C1 coefficient for the fourth region.
        C2 : float
            C2 coefficient.

        Raises
        ------
        ValueError
            This error is raised if the number of regions given in the input
            (stored in X input) is not equal to 4.

        Returns
        -------
        IminusAtt : array
            The predicted value of the intenisty minus the attenuation term, equal to
            C1 + C2.mag (mag is the magnitude). 

        """
        mags, id_region = X
        liste_region = np.unique(id_region)
        if len(liste_region)!=4:
            raise ValueError('Number of region should be 4')
        aregion = np.array([])

        for compt, region in enumerate(liste_region):
            ind = (id_region == region)
            zeros = np.zeros(len(mags))
            zeros[ind] = 1
            try:
                aregion = np.vstack((aregion, zeros))
            except ValueError:
                 aregion = np.concatenate((aregion, zeros))

        C1 = np.vstack(([C1a]*len(mags), [C1b]*len(mags), [C1c]*len(mags), [C1d]*len(mags)
                       ))
        IminusAtt = (C1*aregion).sum(axis=0) + C2*mags 
        return IminusAtt
    
    def C1_3regionC2_EMIPE(self, X, C1a, C1b, C1c, C2):
        """
        Function used to inverse the C1 and C2 coefficients.The C1 coefficient
        is estimated for 3 different regions. 

        Parameters
        ----------
        X : array
            Array that contains the magnitudes of the calibration earthquakes and the corresponding 
            region ID of the earthquakes. The array should correspond to the columns of the Obsbin_plus
            dataframe.
        C1a : float
            C1 coefficient for the first region.
        C1b : float
            C1 coefficient for the second region.
        C1c : float
            C1 coefficient for the third region.

        C2 : float
            C2 coefficient.

        Raises
        ------
        ValueError
            This error is raised if the number of regions given in the input
            (stored in X input) is not equal to 3.

        Returns
        -------
        IminusAtt : array
            The predicted value of the intenisty minus the attenuation term, equal to
            C1 + C2.mag (mag is the magnitude). 
            
        """
        mags, id_region = X
        liste_region = np.unique(id_region)
        if len(liste_region)!=3:
            raise ValueError('Number of region should be 3')
        aregion = np.array([])

        for compt, region in enumerate(liste_region):
            ind = (id_region == region)
            zeros = np.zeros(len(mags))
            zeros[ind] = 1
            try:
                aregion = np.vstack((aregion, zeros))
            except ValueError:
                 aregion = np.concatenate((aregion, zeros))
       

        C1 = np.vstack(([C1a]*len(mags), [C1b]*len(mags), [C1c]*len(mags)
                       ))
        IminusAtt = (C1*aregion).sum(axis=0) + C2*mags 
        return IminusAtt
    
    def C1_2regionC2_EMIPE(self, X, C1a, C1b, C2):
        """
        Function used to inverse the C1 and C2 coefficients.The C1 coefficient
        is estimated for 2 different regions. 

        Parameters
        ----------
        X : array
            Array that contains the magnitudes of the calibration earthquakes and the corresponding 
            region ID of the earthquakes. The array should correspond to the columns of the Obsbin_plus
            dataframe.
        C1a : float
            C1 coefficient for the first region.
        C1b : float
            C1 coefficient for the second region.
        C2 : float
            C2 coefficient.

        Raises
        ------
        ValueError
            This error is raised if the number of regions given in the input
            (stored in X input) is not equal to 2.

        Returns
        -------
        IminusAtt : array
            The predicted value of the intenisty minus the attenuation term, equal to
            C1 + C2.mag (mag is the magnitude). 
            
        """
        mags, id_region = X
        liste_region = np.unique(id_region)
        if len(liste_region)!=2:
            raise ValueError('Number of region should be 2')
        aregion = np.array([])

        for compt, region in enumerate(liste_region):
            ind = (id_region == region)
            zeros = np.zeros(len(mags))
            zeros[ind] = 1
            try:
                aregion = np.vstack((aregion, zeros))
            except ValueError:
                 aregion = np.concatenate((aregion, zeros))
       

        C1 = np.vstack(([C1a]*len(mags), [C1b]*len(mags)
                       ))
        IminusAtt = (C1*aregion).sum(axis=0) + C2*mags 
        return IminusAtt
    
    def C1C2_EMIPE(self, X, C1, C2):
        """
        Function used to inverse the C1 and C2 coefficients.

        Parameters
        ----------
        X : array
            Array that contains the magnitudes of the calibration earthquakes and the corresponding 
            region ID of the earthquakes. The array should correspond to the columns of the Obsbin_plus
            dataframe.
        C1 : float
            C1 coefficient.
        C2 : float
            C2 coefficient.

        Raises
        ------
        ValueError
            This error is raised if the number of regions given in the input
            (stored in X input) is not equal to 1.

        Returns
        -------
        IminusAtt : array
            The predicted value of the intenisty minus the attenuation term, equal to
            C1 + C2.mag (mag is the magnitude). 
            
        """
        mags, id_region = X
        liste_region = np.unique(id_region)
        if len(liste_region)!=1:
            raise ValueError('Number of region should be 1')
        IminusAtt = C1 + C2*mags 
        return IminusAtt
    
    def do_linregressC1regC2(self, sigma='none',
                         ftol=5e-3, xtol=1e-8, max_nfev=200):
        """
        Function that inverts the C1 and C2 coefficients in the following equation:
            I = C1 + C2.Mag + beta.log10(hypo) + gamma.hypo
        C1 coefficient can be estimated by regions, with a maximal number of region
        equal to 4.

        Parameters
        ----------
        sigma : float or array or 2D array, optional
            sigma used to weight the data in the inversion process. The default is 'none'.
            None (default) is equivalent of 1-D sigma filled with ones
        ftol : TYPE, optional
           Tolerance for termination by the change of the cost function.
           The optimization process is stopped when dF < ftol * F, 
           and there was an adequate agreement between a local quadratic model 
           and the true model in the last step. This tolerance must be higher than machine epsilon.
           The default is 5e-3.
        xtol : TYPE, optional
            Tolerance for termination by the change of the independent variables
            The default is 1e-8. The condition is:
                Delta < xtol * norm(xs), where Delta is a trust-region radius
                and xs is the value of x scaled according to x_scale parameter
        max_nfev : TYPE, optional
            Maximum number of function evaluations before the termination. The default is 200.

        Raises
        ------
        ValueError
            This error is raised when the number of regions to regionalized C1 coefficient
            is greater than 4. Each region is associated to an ID.

        Returns
        -------
        C1regC2 : TYPE
            array with two array elements. First element is the popt output of scipy.optimize.curve_fit,
            i.e. the optimal values for the parameters so that the sum of the squared residuals
            of f(xdata, *popt) - ydata is minimized. The number of elements of the popt depends on the number of region
            defined by the user. The first elements are the C1 coefficients (one for each region defined) and the last one is the C2 coefficient.
            The order of the regions is the same as the order of appareance in the RegID of the Obsbin_plus dataframe.
            Second element is pcov output of scipy.optimize.curve_fit,
            i.e. the estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
            To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).

        """
        hypos = np.sqrt(self.Obsbin_plus['Depi'].values.astype(float)**2 + self.Obsbin_plus['Depth'].values.astype(float)**2)
        IminusAtt = self.Obsbin_plus['I'].values - self.Obsbin_plus['beta'].values*np.log10(hypos) -  self.Obsbin_plus['gamma'].values*hypos
        X = [self.Obsbin_plus.Mag.values.astype(float),
             self.Obsbin_plus.RegID.values.astype(float)]
        # print(IminusAtt)
        # print(self.Obsbin_plus['I'].values)
        # print(self.Obsbin_plus['beta'].values)
        # print(self.Obsbin_plus['gamma'].values)
        # print(hypos)
        #print(X)
        liste_region = np.unique(self.Obsbin_plus.RegID.values.astype(float))

        if len(liste_region) == 4:
            C1regC2 = curve_fit(self.C1_4regionC2_EMIPE, X, IminusAtt, 
                                 sigma=sigma,
                                 absolute_sigma=False,
                                 xtol=xtol, ftol=ftol)
        elif len(liste_region) == 3:
            C1regC2 = curve_fit(self.C1_3regionC2_EMIPE, X, IminusAtt, 
                                 sigma=sigma,
                                 absolute_sigma=False,
                                 xtol=xtol, ftol=ftol)
        elif len(liste_region) == 2:
            C1regC2 = curve_fit(self.C1_2regionC2_EMIPE, X, IminusAtt, 
                                 sigma=sigma,
                                 absolute_sigma=False,
                                 xtol=xtol, ftol=ftol)
        elif len(liste_region) == 1:
            C1regC2 = curve_fit(self.C1C2_EMIPE, X, IminusAtt, 
                                 sigma=sigma,
                                 absolute_sigma=False,
                                 xtol=xtol, ftol=ftol)
        else:
            raise ValueError('Number of region should be less than 4')
        return C1regC2
