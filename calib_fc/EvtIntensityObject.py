# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:41:31 2021

@author: PROVOST-LUD
"""
import numpy as np
import pandas as pd
import library_bin as libr
from tkinter import messagebox as tkm
from mpl_toolkits.basemap import pyproj

_GEOD = pyproj.Geod(ellps='WGS84')
def CalcDist(lon1, lat1, lon2, lat2):
    return _GEOD.inv(lon1, lat1, lon2, lat2)[2]/1000.


class Evt():
    def __init__(self, Ffp):
        """
        Initialize the Evt class with an evt file and a obs file 

        Parameters
        ----------
        Ffp : class
            class object with EvtFile and ObsFile attributes. EvtFile and ObsFile
            are pandas.dataframe objects. EvtFile contains the metadata of different
            earthquakes and ObsFile contains the corresponding macroseismic fields.
            Mandatory columns for EvtFile:
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
        self.FfP = Ffp
    
    def build(self, evid):
        """
        Build the evt basic object. The evt basic object contains information like the
        date of occurence, the epicentral intensity or the location of a given earthquake

        Parameters
        ----------
        evid : str or int
            ID of the earthquake of interest.

        Returns
        -------
        None.

        """
        # Standard deviation of epicentral intensity based on quality factors
        #Std ={'A':0.5,'B':0.5,'C':0.5,'E':0.750, 'K':0.5}
        Std ={'A':0.25,'B':0.375,'C':0.5,'E':0.750, 'K':0.5}
        
        self.evid = evid
        try:
            EvtFile = self.FfP.EvtFile
            ObsFile = self.FfP.ObsFile
        except AttributeError:
            tkm.showerror("Error", "Unexpected error, please reload the input data files by pushing Reset")
        
        # if self.evid < 0 or not isinstance(self.evid, int):
        #     tkm.showerror("Error", "Please enter an integer")
        #     return
        
        if np.size(EvtFile[EvtFile['EVID']==self.evid].values) == 0 :
            print("L'evid " + str(self.evid) + " n'existe pas")
            tkm.showerror("Error","Evid " + str(self.evid) + " doesn't exist")
            return
        
        self.day = EvtFile[EvtFile['EVID']==self.evid]['Day'].values[0]
        self.month = EvtFile[EvtFile['EVID']==self.evid]['Month'].values[0]
        self.year = EvtFile[EvtFile['EVID']==self.evid]['Year'].values[0]
        
        self.QI0name = EvtFile[EvtFile['EVID']==self.evid]['QI0'].values[0]
        if not isinstance(self.QI0name,str):
            tkm.showerror("Error","invalid QI0")
        if not ((self.QI0name >= 'A' and self.QI0name <= 'C') or self.QI0name == 'K' or self.QI0name == 'E' or self.QI0name == 'I'):
            tkm.showerror("","invalid QI0")
            return;
        self.QI0 = Std[(self.QI0name)]
        #print("QI0name:")
        #print(self.QI0name)
        self.QPos = EvtFile[EvtFile['EVID']==self.evid]['QPos'].values[0]
        if isinstance(self.QPos, bytes):
            self.QPos = self.QPos.decode('utf8')
        if not isinstance(self.QPos, str):
            tkm.showerror("Error","invalid QPos")
            print(self.QPos)
            return;
        if not ((self.QPos >= 'A' and self.QPos <= 'D') or self.QPos == 'I' or self.QPos == 'K' or self.QPos == 'E'):
            tkm.showerror("Error","invalid QPos")
            print(self.QPos)
            return;
        #print("QPos")
        #print(self.QPos)
        self.Io_ini = EvtFile[EvtFile['EVID']==self.evid]['I0'].values[0]
        self.I0 = self.Io_ini
        if self.Io_ini < 0 or self.Io_ini > 12:
            tkm.showerror("Error","invalid I0")
            return;
        #print("I0 catalogue")
        #print(self.Io_ini )
        #voir pur maj Iinf Isup LImitforsampling
        self.Io_inf = self.Io_ini - 2*self.QI0
        self.Io_sup = self.Io_ini + 2*self.QI0
        try:
            self.Ic = EvtFile[EvtFile['EVID']==self.evid]['Ic'].values[0]
            self.Dc = EvtFile[EvtFile['EVID']==self.evid]['Dc'].values[0]
        except KeyError:
            print('No Ic, Dc in evt file')
        if self.Ic == 12:
            self.Ic = 3
        try:
            self.Mag = EvtFile[EvtFile['EVID']==self.evid]['Mag'].values[0]
            self.StdM = EvtFile[EvtFile['EVID']==self.evid]['StdM'].values[0]
        except KeyError:
            print('No Mag, StdM in evt file')
            
        try:
            self.depth = EvtFile[EvtFile['EVID']==self.evid]['Depth'].values[0]
            self.Hmin = EvtFile[EvtFile['EVID']==self.evid]['Hmin'].values[0]
            self.Hmax = EvtFile[EvtFile['EVID']==self.evid]['Hmax'].values[0]
        except KeyError:
            print('No depth, Hmin, Hmax in evt file')
        """
        # Under development
        if hasattr(self.FfP, 'Parameterfile'):
            print("Use Parameterfile")
            self.Ic = ParameterFile[ParameterFile['EVID']==self.evid]['Ic'].values[0]
            if not (isinstance(self.Ic, float)):
                tkm.showerror("Error","Invalid Ic")
                return
            self.Nobs = ParameterFile[ParameterFile['EVID']==self.evid]['NObs'].values[0]
            self.Nfelt = ParameterFile[ParameterFile['EVID']==self.evid]['NFelt'].values[0]
        else:
            self.Nfelt = float(np.size(ObsFile[(ObsFile['EVID']==self.evid) & (ObsFile['Iobs'] == -1)].values)/6)
        """
        
        self.Nobs = float(np.size(ObsFile[(ObsFile['EVID']==self.evid) & (ObsFile['Iobs'] > 0)].values)/6)
        
        if not (isinstance(self.Nobs, float)) :
            tkm.showerror("Error", "Invalid Nobs ")
            return
        self.Lat_evt = float(EvtFile[EvtFile['EVID']==self.evid]['Lat'].values[0])
        self.Lon_evt = float(EvtFile[EvtFile['EVID']==self.evid]['Lon'].values[0])
        if not (isinstance(self.Lat_evt, float) and isinstance(self.Lon_evt,float)):
            tkm.showerror("Error", "invalid lat or lon")
            return
        ObsFile.loc[ObsFile['EVID']==self.evid, 'Depi'] = ObsFile.loc[ObsFile['EVID']==self.evid].apply(lambda row:CalcDist(row['Lon'],row['Lat'],self.Lon_evt,self.Lat_evt),axis=1)
        ObsFile.loc[ObsFile['EVID']==self.evid,'I0'] = self.Io_ini
        
        date = EvtFile[EvtFile['EVID']==self.evid]['Year'].values[0]
        ObsFile.loc[ObsFile['EVID']==self.evid,'Year'] = date
        
        self.Obsevid = ObsFile[ObsFile['EVID']==self.evid]
    
    def Binning_Obs(self, depth, Ic, method_bin='ROBS'):
        """
        Compute isoseismal radii with 6 different available methods from the Evt object.
        Evt object should have been built with the build function before computins isoseismal 
        radii. The 6 methods are described in Traversa et al (2017), Exploration tree approach
        to estimate historical earthquakes Mw and depth, test cases from the French past seismicity,
        Bulletin of Earthquake Engineering

        Parameters
        ----------
        depth : float
            depth of the earthquake.
        Ic : float
            intensity of completeness of the earthquake. The intensities smaller than Ic in a macroseismic field are considered as incomplete
            In the isoseismal radii based on intensity bins, intensities smaller than Ic are not taken into account
            to compute the isoseismal radii.
        method_bin : str, optional
            Name of the chosen method to compute isoseismal radii. The default is 'ROBS'.
            The other names are 'RAVG', 'RP50', 'RP84', 'RF50' and 'RF84'. All methods are based
            on intensity bins.
            - ROBS : width of the intensity bin : 0.1. 
                  Value of intensity for one bin : weighted mean of the IDP's
                  intensity within the intensity bin. With a width of 0.1 in intensity, 
                  the mean is equal to the intensity value of the intensity bin
                  Value of the epicentral distance for one bin: weighted mean of 
                  decimal logarithm of the IDP's distance within the intensity bin.
            - RAVG : width of the intensity bin : 1. 
                  Value of intensity for one bin : weighted mean of the IDP's
                  intensity within the intensity bin. 
                  Value of the epicentral distance for one bin: weighted mean of 
                  decimal logarithm of the IDP's distance within the intensity bin.
            - RP50 : width of the intensity bin : 0.1. 
                  Value of intensity for one bin : intensity value of the intensity bin. 
                  Value of the epicentral distance for one bin: weighted median of 
                  decimal logarithm of the IDP's distance within the intensity bin.
            - RP84 : width of the intensity bin : 0.1. 
                 Value of intensity for one bin : intensity value of the intensity bin. 
                 Value of the epicentral distance for one bin: weighted 84 percentile of 
                 decimal logarithm of the IDP's distance within the intensity bin.
            - RF50 : same as RP50, however only the epicentral intensity and the intensity
                  bin being representative of far field reliable information are kept
            - RF84 : same as RP84, however only the epicentral intensity and the intensity
                  bin being representative of far field reliable information are kept

        Returns
        -------
        None.

        """
        if method_bin == 'RAVG':
            self.ObsBinn = libr.RAVG(self.Obsevid, depth, Ic, self.I0, self.QI0)
        elif method_bin == 'ROBS':
            self.ObsBinn = libr.ROBS(self.Obsevid, depth, Ic, self.I0, self.QI0)
        elif method_bin == 'RP50':
            self.ObsBinn = libr.RP50(self.Obsevid, depth, Ic, self.I0, self.QI0)
        elif method_bin == 'RP84':
            self.ObsBinn = libr.RP84(self.Obsevid, depth, Ic, self.I0, self.QI0)
        elif method_bin == 'RF50':
            self.ObsBinn = libr.RF50(self.Obsevid, depth, Ic, self.I0, self.QI0)
        elif method_bin == 'RF84':
            self.ObsBinn = libr.RF84(self.Obsevid, depth, Ic, self.I0, self.QI0)
        
    def Binning_Obs_old(self, depth, Ic):
        Stdobs = {'A':0.5,'B':0.577,'C':0.710,'D':1.0,'I':1.5,'K':2.0}
        colonnes_binn = ['EVID','Hypo','I','Io','QIo','StdI','StdLogR','Ndata']
        Depi = []
        for epi in self.Obsevid['Depi'].values:
            Depi.append(epi)
        Depi = np.array(Depi)
        
        Hypo = libr.Distance_c(depth, Depi)
        #print("Hypo:")
        #print(Hypo)
        IOBS = []
        for iob in self.Obsevid['Iobs'].values:
            IOBS.append(iob)
        Iobs = np.array(IOBS)
#        print("Iobs:")
#        print(Iobs)
        QIOBS = []
        for qiob in self.Obsevid['QIobs'].values:
            QIOBS.append(Stdobs[qiob])
        QIobs=np.array(QIOBS)
#        print("QIobs:")
#        print(QIobs)
        
        if hasattr(self,'Ic') and Ic == -1:
            Ic = self.Ic
        
#        print("Ic:")
#        print(Ic)
        
        I0 = float(self.Io_ini)
        QI0 = float(self.QI0)
        depth = float(depth)
        evid = int(self.evid)
        Ic = float(Ic)
        SortieBinn = libr.RAVG_c(Iobs,Hypo,QIobs,I0,QI0,depth,evid, Ic,30)
        self.ObsBinn = pd.DataFrame(data=SortieBinn, columns=colonnes_binn)
        self.ObsBinn = self.ObsBinn[self.ObsBinn['EVID'] != 0]
        print(self.ObsBinn)