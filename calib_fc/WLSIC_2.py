#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:31:20 2018

@author: PROVOLU
"""
import numpy as np
import pandas as pd
import sys

from scipy.optimize import curve_fit, minimize, Bounds
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
        return I

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
        Ibin = self.Obsbin['I'].values
        Depi = self.Obsbin['Depi'].values
        resH = curve_fit(self.EMIPE_H, Depi, Ibin, p0=self.depth,
                                 jac= self.EMIPE_JACdH, bounds=(depth_inf, depth_sup),
                                 sigma=self.Obsbin['StdI'].values, absolute_sigma=True,
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
        Ibin = self.Obsbin['I'].values
        Depi = self.Obsbin['Depi'].values
        resI0 = curve_fit(self.EMIPE_I0, Depi, Ibin, p0=self.I0,
                                  jac= self.EMIPE_JACdI0, bounds=(I0_inf, I0_sup),
                                  sigma=self.Obsbin['StdI'].values, absolute_sigma=True,
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
        return I
    
    def EMIPE_gamma(self, X, gamma):
        """
        Function used to inverse the intrisic attenuation coefficient
        :param X: matrix that contains epicentral distance, depth and epicentral
                  intensity associated to the binned intensity
        :param gamma: intrisic attenuation coefficient
        :type X: numpy.array
        :type gamma: float
        """
        Depi, depths, I0s = X
        try:
            logterm = np.sqrt(Depi**2+depths**2)/depths
        except AttributeError:
            tmp = (Depi**2+depths**2)**0.5/depths
            logterm = np.array([])
            for tt in tmp:
                logterm = np.append(logterm, tt)
            
        I = I0s + self.beta*np.log10(logterm) + gamma*(np.sqrt(Depi**2+depths**2)-depths)
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
        return I
        
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
        Ibin = self.ObsBin_plus['I'].values
        Depi = self.ObsBin_plus['Depi'].values
        depths = self.ObsBin_plus['Depth'].values
        I0s = self.ObsBin_plus['Io'].values
        X = [np.array(Depi), np.array(depths), np.array(I0s)]
        resBeta = curve_fit(self.EMIPE_beta, X, Ibin, p0=self.beta,
                                  sigma=self.ObsBin_plus['eqStd'].values, absolute_sigma=True,
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
        Ibin = self.ObsBin_plus['I'].values
        Depi = self.ObsBin_plus['Depi'].values
        depths = self.ObsBin_plus['Depth'].values
        I0s = self.ObsBin_plus['Io'].values
        X = [np.array(Depi), np.array(depths), np.array(I0s)]
        resBeta = curve_fit(self.EMIPE_beta, X, Ibin, p0=self.beta, bounds=(self.beta-0.0001, self.beta+0.0001),
                                  sigma=self.ObsBin_plus['StdI'].values, absolute_sigma=True,
                                  xtol=1e-3)
        return resBeta
    
    def do_wls_gamma_std(self):
        """
        Function used to compute the covariance matrix associated to the 
        intrinsic attenuation coefficient based on the standard deviations associated
        to the intensity data.
        :return: [popt, pcov] list with popt a (1, 1) shape array containing
                 the inverted gamma. pcov is a 2-D array and 
                 the estimated covariance of popt. The diagonals provide the
                 variance of the parameter estimate. 
                 To compute one standard deviation errors on the parameters use
                 perr = np.sqrt(np.diag(pcov)).
        """
        Ibin = self.ObsBin_plus['I'].values
        Depi = self.ObsBin_plus['Depi'].values
        depths = self.ObsBin_plus['Depth'].values
        I0s = self.ObsBin_plus['Io'].values
        X = [np.array(Depi), np.array(depths), np.array(I0s)]
        #
        resGamma = curve_fit(self.EMIPE_gamma, X, Ibin, p0=self.gamma, bounds=(self.gamma-1e+6, self.gamma+1e-6),
                                  sigma=self.ObsBin_plus['StdI'].values, absolute_sigma=True,
                                  xtol=1e-4)
        if resGamma[0] > 0:
            resGamma[0] = 0
        return resGamma
    
    def do_wls_gamma(self):
        """
        Function used to launch intrisic attenuation coefficient inversion
        :return: [popt, pcov] list with popt a (1, 1) shape array containing
                 the inverted gamma. pcov is a 2-D array and 
                 the estimated covariance of popt. The diagonals provide the
                 variance of the parameter estimate. 
                 To compute one standard deviation errors on the parameters use
                 perr = np.sqrt(np.diag(pcov)).
                 In the case of a weighting scheme different as 'Ponderation dI',
                 please use the do_wls_gamma_std() function to retrieve
                 the covariance matrix based on the intensity data standard
                 deviation just after inverting beta with the present function.
        """
        Ibin = self.ObsBin_plus['I'].values
        Depi = self.ObsBin_plus['Depi'].values
        depths = self.ObsBin_plus['Depth'].values
        I0s = self.ObsBin_plus['Io'].values
        X = [np.array(Depi), np.array(depths), np.array(I0s)]
        #
        resGamma = curve_fit(self.EMIPE_gamma, X, Ibin, p0=self.gamma, bounds=(-np.inf, 1e-6),
                                  sigma=self.ObsBin_plus['eqStd'].values, absolute_sigma=True,
                                  xtol=1e-4)
        if resGamma[0] > 0:
            resGamma[0] = 0
        return resGamma
    
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
        Ibin = self.ObsBin_plus['I'].values
        Depi = self.ObsBin_plus['Depi'].values
        depths = self.ObsBin_plus['Depth'].values
        I0s = self.ObsBin_plus['Io'].values
        X = [Depi, depths, I0s]
        resBetaGamma = curve_fit(self.EMIPE_beta_gamma, X, Ibin, p0=[self.beta, self.gamma],
                                  bounds=([-np.inf, -np.inf], [np.inf, 0]),
                                  sigma=self.ObsBin_plus['eqStd'].values, absolute_sigma=True,
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
        
        Ibin = self.ObsBin_plus['I'].values
        Depi = self.ObsBin_plus['Depi'].values
        depths = self.ObsBin_plus['Depth'].values
        I0s = self.ObsBin_plus['Io'].values
        X = [Depi, depths, I0s]
        resBetaGamma = curve_fit(self.EMIPE_beta_gamma, X, Ibin, p0=[self.beta, self.gamma],
                                  bounds=([self.beta-0.0001, self.gamma-1e-6], [self.beta+0.0001, self.gamma+1e-6]),
                                  sigma=self.ObsBin_plus['StdI'].values, absolute_sigma=True,
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
        Ibin = self.Obsbin['I'].values
        Depi = self.Obsbin['Depi'].values
        resH = curve_fit(self.EMIPE_H, Depi, Ibin, p0=self.depth,
                                 jac= self.EMIPE_JACdH, bounds=(depth_inf, depth_sup),
                                 sigma=self.Obsbin['StdI'].values, absolute_sigma=True,
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
        
    

    def EMIPE_JACdC1C2BetaH_old(self, X, C1, C2, beta, H1, H2, H3, H4, H5, H6, H7, H8,
                        H9, H10, H11, H12, H13, H14, H15, H16, H17, H18, H19, H20,
                        H21, H22, H23, H24, H25, H26, H27, H28, H29, H30, H31):
        """
        Jacobian function used to inverse depth
        :param Depi: epicentral distances associated to the binned intensity data
        :param H: hypocenter's depth of the considered earthquake.
                      Should be greater than 0
        :type Depi: numpy.array
        :type H: float
        """
        depi, Mag = X[:2]
        aH = X[2:][0]
        H = np.vstack(([H1]*len(depi), [H2]*len(depi), [H3]*len(depi), [H4]*len(depi),
                       [H5]*len(depi), [H6]*len(depi), [H7]*len(depi), [H8]*len(depi),
                       [H9]*len(depi), [H10]*len(depi), [H11]*len(depi), [H12]*len(depi),
                       [H13]*len(depi), [H14]*len(depi), [H15]*len(depi), [H16]*len(depi),
                       [H17]*len(depi), [H18]*len(depi), [H19]*len(depi), [H20]*len(depi),
                       [H21]*len(depi), [H22]*len(depi), [H23]*len(depi), [H24]*len(depi),
                       [H25]*len(depi), [H26]*len(depi), [H27]*len(depi), [H28]*len(depi),
                       [H29]*len(depi), [H30]*len(depi), [H31]*len(depi)
                       ))
        Hypo = np.sqrt(depi**2+(aH*H).sum(axis=0)**2)
        tmpValue = (aH*H).sum(axis=0)/Hypo
        gC1 = np.array([np.ones(len(depi))])
        gC2 = np.array([Mag])
        gbeta = np.array([np.log10(Hypo)])
        #gH = (tmpValue)*((self.beta/(np.log(10)*Hypo))+self.gamma)
        GH = np.array([])
        #np.hstack((a,b))
        for ahh, hh in zip(aH, H):
            Hypo = np.sqrt(depi**2+ahh*hh**2)
            tmpValue = ahh*hh/Hypo
            gH = (tmpValue)*((beta/(np.log(10)*Hypo)))
            gH = np.nan_to_num(gH)
            #GH = np.tile(gH, (len(Depi),1))
            try:
                GH = np.vstack((GH, np.array([gH])))
            except ValueError:
                GH = np.array([gH])
        #g = np.array([gC1, gC2, gbeta, GH])
        g = np.vstack((gC1, gC2))
        print(g.shape)
        g = np.vstack((g, gbeta))
        g = np.vstack((g, GH))
        print(g.shape)
        print(g.T.shape)
        print(g.T[0])
        print(g.T)
        return g.T
    
    def EMIPE_JACdH(self, X, C1, C2, beta, H1, H2, H3, H4, H5, H6, H7, H8,
                        H9, H10, H11, H12, H13, H14, H15, H16, H17, H18, H19, H20,
                        H21, H22, H23, H24, H25, H26, H27, H28, H29, H30, H31):
        """
        Jacobian function used to inverse depth
        :param Depi: epicentral distances associated to the binned intensity data
        :param H: hypocenter's depth of the considered earthquake.
                      Should be greater than 0
        :type Depi: numpy.array
        :type H: float
        """
        depi, Mag = X[:2]
        aH = X[2:][0]
        H = np.vstack(([H1]*len(depi), [H2]*len(depi), [H3]*len(depi), [H4]*len(depi),
                       [H5]*len(depi), [H6]*len(depi), [H7]*len(depi), [H8]*len(depi),
                       [H9]*len(depi), [H10]*len(depi), [H11]*len(depi), [H12]*len(depi),
                       [H13]*len(depi), [H14]*len(depi), [H15]*len(depi), [H16]*len(depi),
                       [H17]*len(depi), [H18]*len(depi), [H19]*len(depi), [H20]*len(depi),
                       [H21]*len(depi), [H22]*len(depi), [H23]*len(depi), [H24]*len(depi),
                       [H25]*len(depi), [H26]*len(depi), [H27]*len(depi), [H28]*len(depi),
                       [H29]*len(depi), [H30]*len(depi), [H31]*len(depi)
                       ))
        GH = np.array([])
        for ahh, hh in zip(aH, H):
            Hypo = np.sqrt(depi**2+ahh*hh**2)
            tmpValue = ahh*hh/Hypo
            gH = (tmpValue)*((beta/(np.log(10)*Hypo)))
            gH = np.nan_to_num(gH)
            try:
                GH = np.vstack((GH, np.array([gH])))
            except ValueError:
                GH = np.array([gH])
#        print(GH.shape)
#        print(len(GH[0]))
#        for gh in GH:
#            print(gh)
        return GH
    
    def dgh(self, X, C1, C2, beta, H1, H2, H3, H4, H5, H6, H7, H8,
            H9, H10, H11, H12, H13, H14, H15, H16, H17, H18, H19, H20,
            H21, H22, H23, H24, H25, H26, H27, H28, H29, H30, H31):
        depi, Mag = X[:2]
        aH = X[2:][0]
        H = np.vstack(([H1]*len(depi), [H2]*len(depi), [H3]*len(depi), [H4]*len(depi),
                       [H5]*len(depi), [H6]*len(depi), [H7]*len(depi), [H8]*len(depi),
                       [H9]*len(depi), [H10]*len(depi), [H11]*len(depi), [H12]*len(depi),
                       [H13]*len(depi), [H14]*len(depi), [H15]*len(depi), [H16]*len(depi),
                       [H17]*len(depi), [H18]*len(depi), [H19]*len(depi), [H20]*len(depi),
                       [H21]*len(depi), [H22]*len(depi), [H23]*len(depi), [H24]*len(depi),
                       [H25]*len(depi), [H26]*len(depi), [H27]*len(depi), [H28]*len(depi),
                       [H29]*len(depi), [H30]*len(depi), [H31]*len(depi)
                       ))
        dGH = np.array([])
        for ahh, hh in zip(aH, H):
            Hypo = np.sqrt(depi**2+ahh*hh**2)
            tmpValue = ahh*hh/Hypo
            bt = (beta/np.log(10))
            dgH = ((Hypo- hh*tmpValue)/(Hypo**2))*(bt/Hypo) + tmpValue*(bt/Hypo)
            dgH = np.nan_to_num(dgH)
            try:
                dGH = np.vstack((dGH, np.array([dgH])))
            except ValueError:
                dGH = np.array([dgH])
        return dGH
    
    def dhypodH(self, X, C1, C2, beta, H1, H2, H3, H4, H5, H6, H7, H8,
            H9, H10, H11, H12, H13, H14, H15, H16, H17, H18, H19, H20,
            H21, H22, H23, H24, H25, H26, H27, H28, H29, H30, H31):
        depi, Mag = X[:2]
        aH = X[2:][0]
        H = np.vstack(([H1]*len(depi), [H2]*len(depi), [H3]*len(depi), [H4]*len(depi),
                       [H5]*len(depi), [H6]*len(depi), [H7]*len(depi), [H8]*len(depi),
                       [H9]*len(depi), [H10]*len(depi), [H11]*len(depi), [H12]*len(depi),
                       [H13]*len(depi), [H14]*len(depi), [H15]*len(depi), [H16]*len(depi),
                       [H17]*len(depi), [H18]*len(depi), [H19]*len(depi), [H20]*len(depi),
                       [H21]*len(depi), [H22]*len(depi), [H23]*len(depi), [H24]*len(depi),
                       [H25]*len(depi), [H26]*len(depi), [H27]*len(depi), [H28]*len(depi),
                       [H29]*len(depi), [H30]*len(depi), [H31]*len(depi)
                       ))
        dHypo = np.array([])
        for ahh, hh in zip(aH, H):
            Hypo = np.sqrt(depi**2+ahh*hh**2)
            tmpValue = ahh*hh/Hypo
            dhyp = (tmpValue/Hypo)/np.log(10)
            try:
                dHypo = np.vstack((dHypo, np.array([dhyp])))
            except ValueError:
                dHypo = np.array([dhyp])
        return dHypo
    
    def EMIPE_C1C2BetaH(self, X, C1, C2, Beta, H1, H2, H3, H4, H5, H6, H7, H8,
                        H9, H10, H11, H12, H13, H14, H15, H16, H17, H18, H19, H20,
                        H21, H22, H23, H24, H25, H26, H27, H28, H29, H30, H31):
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
        mags, depi = X[:2]
        ah = X[2:][0]
        H = np.vstack(([H1]*len(depi), [H2]*len(depi), [H3]*len(depi), [H4]*len(depi),
                       [H5]*len(depi), [H6]*len(depi), [H7]*len(depi), [H8]*len(depi),
                       [H9]*len(depi), [H10]*len(depi), [H11]*len(depi), [H12]*len(depi),
                       [H13]*len(depi), [H14]*len(depi), [H15]*len(depi), [H16]*len(depi),
                       [H17]*len(depi), [H18]*len(depi), [H19]*len(depi), [H20]*len(depi),
                       [H21]*len(depi), [H22]*len(depi), [H23]*len(depi), [H24]*len(depi),
                       [H25]*len(depi), [H26]*len(depi), [H27]*len(depi), [H28]*len(depi),
                       [H29]*len(depi), [H30]*len(depi), [H31]*len(depi)
                       ))
#        print(H.shape)
#        print(ah.shape)
        #print((H*ah).sum(axis=0))
        #ah --> (n, len(Depi)) array avec n le nombre de EQ. Chaque ligne contient
        #des 1 et des 0 et correspond a un EQ. 1 est attribue aux indices de
        # obsbin_plus.EVID ==evid concerne.
        #H --> (n, len(Depi)) array avec n le nombre de EQ, chaque ligne contient obsbin_plus.Depth
        hypos = np.sqrt(depi**2 + (H*ah).sum(axis=0)**2)
        I = C1 + C2*mags + Beta*np.log10(hypos)
        return I
    
    def hypos(self, X, C1, C2, Beta, H1, H2, H3, H4, H5, H6, H7, H8,
                        H9, H10, H11, H12, H13, H14, H15, H16, H17, H18, H19, H20,
                        H21, H22, H23, H24, H25, H26, H27, H28, H29, H30, H31):
        mags, depi = X[:2]
        ah = X[2:][0]
        H = np.vstack(([H1]*len(depi), [H2]*len(depi), [H3]*len(depi), [H4]*len(depi),
                       [H5]*len(depi), [H6]*len(depi), [H7]*len(depi), [H8]*len(depi),
                       [H9]*len(depi), [H10]*len(depi), [H11]*len(depi), [H12]*len(depi),
                       [H13]*len(depi), [H14]*len(depi), [H15]*len(depi), [H16]*len(depi),
                       [H17]*len(depi), [H18]*len(depi), [H19]*len(depi), [H20]*len(depi),
                       [H21]*len(depi), [H22]*len(depi), [H23]*len(depi), [H24]*len(depi),
                       [H25]*len(depi), [H26]*len(depi), [H27]*len(depi), [H28]*len(depi),
                       [H29]*len(depi), [H30]*len(depi), [H31]*len(depi)
                       ))
        return np.sqrt(depi**2 + (H*ah).sum(axis=0)**2)
        
    
    def dC1_driver_func(self, x, xobs, I):
        ynew = self.EMIPE_C1C2BetaH(xobs, *x)
        return np.sum(2*(ynew - I))
    
    def dC2_driver_func(self, x, xobs, I):
        ynew = self.EMIPE_C1C2BetaH(xobs, *x)
        return np.sum(2*xobs[0]*(ynew - I))
    
    
    def dbeta_driver_func(self, x, xobs, I):
        ynew = self.EMIPE_C1C2BetaH(xobs, *x)
        hypo = self.hypos(xobs, *x)
        return np.sum(2*np.log10(hypo)*(ynew - I))
    
    def dH_driver_func(self, x, xobs, I):
        ynew = self.EMIPE_C1C2BetaH(xobs, *x)
        GH = self.EMIPE_JACdH(xobs, *x)
        val_jac = np.array([])
        for gh in GH:
            vv = np.sum(2*gh*(ynew - I))
            val_jac = np.append(val_jac, vv)
        return val_jac
    
    def ddC1_driver_func(self, x, xobs, I):
        return np.sum(2*np.ones(len(xobs[0])))
    
    def ddC2_driver_func(self, x, xobs, I):
        return np.sum(2*xobs[0]**2)
    
    
    def ddbeta_driver_func(self, x, xobs, I):
        hypo = self.hypos(xobs, *x)
        return np.sum(2*np.log10(hypo)**2)
    
    
    def ddH_driver_func(self, x, xobs, I):
        ynew = self.EMIPE_C1C2BetaH(xobs, *x)
        GH = self.EMIPE_JACdH(xobs, *x)
        dGH = self.dgh(xobs, *x)
        val_jac = np.array([])
        for gh in GH:
            vv = np.sum(2*(gh*gh + dGH*(ynew - I)))
            val_jac = np.append(val_jac, vv)
        return val_jac    
        
    def hess_C1C2BetaH(self, x, xobs, I):
        ddC1 = self.ddC1_driver_func(x, xobs, I)
        ddC2 = self.ddC2_driver_func(x, xobs, I)
        ddbeta = self.ddbeta_driver_func(x, xobs, I)
        ddH = self.ddH_driver_func(x, xobs, I)
        diag = np.append(ddC1, ddC2)
        diag = np.append(diag, ddbeta)
        diag = np.append(diag, ddH)
        hess = np.diag(diag) # que la diagonale, manque les autres  derivees secondes
        hess[0][1] = np.sum(2*xobs[0])
        hess[1][0] = np.sum(2*xobs[0])
        hypo = self.hypos(xobs, *x)
        hess[0][2] = np.sum(2*np.log10(hypo))
        hess[2][0] = np.sum(2*np.log10(hypo))
        hess[1][2] = np.sum(2*np.log10(hypo)*xobs[0])
        hess[2][1] = np.sum(2*np.log10(hypo)*xobs[0])
        # Reste a determiner les autres diagonales
        GH = self.EMIPE_JACdH(xobs, *x)
        ynew = self.EMIPE_C1C2BetaH(xobs, *x)
        dhypodH = self.dhypodH(xobs, *x)
        ind = 0
        for gh, dhyp in zip(GH, dhypodH):
            #C1
            hess[0][2+ (ind+1)] = np.sum(2*gh)
            hess[2+ (ind+1)][0] = np.sum(2*gh)
            #C2
            hess[1][2+ (ind+1)] = np.sum(2*gh*xobs[0])
            hess[2+ (ind+1)][1] = np.sum(2*gh*xobs[0])
            #beta
            hess[2][2+ (ind+1)] = np.sum(2*gh*xobs[0])
            hess[2+ (ind+1)][2] = np.sum(2*(gh*np.log10(hypo)+(ynew - I)*dhyp))
            ind += 1
        #print(hess)
        return hess
        
    def jac_C1C2BetaH(self, x, xobs, I):
        dC1 = self.dC1_driver_func(x, xobs, I)
        dC2 = self.dC2_driver_func(x, xobs, I)
        dbeta = self.dbeta_driver_func(x, xobs, I)
        dH = self.dH_driver_func(x, xobs, I)
        jacobian = np.append(dC1, dC2)
        jacobian = np.append(jacobian, dbeta)
        jacobian = np.append(jacobian, dH)
        return jacobian
    
    def driver_func(self, x, xobs, I):

        # Evaluate the fit function with the current parameter estimates
    
        ynew = self.EMIPE_C1C2BetaH(xobs, *x)
        yerr = np.sum((ynew - I) ** 2)
    
        return yerr
    
    def do_wls_C1C2BetaH2(self, sigma='none'):
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
        if sigma == 'none':
            sigma = self.Obsbin_plus['eqStdM'].values
        p0 = np.append(np.array([self.C1, self.C2, self.beta]),
                       depths)
        bounds_inf = np.append(np.array([-np.inf, -np.inf, -np.inf]),
                       Hmin)
        bounds_sup = np.append(np.array([np.inf, np.inf, np.inf]),
                       Hmax)
        #print(len(p0), len(bounds_inf))
        bounds = Bounds(bounds_inf, bounds_sup)
        C1C2BetaH = minimize(self.driver_func,
                             jac=self.jac_C1C2BetaH,
                             hess=self.hess_C1C2BetaH,
                             args=(X, Ibin),
                             x0=p0,
                             bounds=bounds,
                             method='trust-constr',
                             )
                             #
        return C1C2BetaH
    
    """
    Comment calculer la matrice de covariance apres optimisation avec minimize:
    J = res_lsq.jac
    cov = np.linalg.inv(J.T.dot(J)) * sum[(f(x)-y)^2]/(N-n)
    N --> nombre de donnees
    n --> nombre de variables à inverser
    Sinon, voir dans curve_fit comment pcov est calcule
    """
    def compute_2Dsigma(self, eta, col='eqStdM'):
        sigma = np.diag(self.Obsbin_plus[col].values)
        sigma[sigma==0] = eta
        return sigma
    
