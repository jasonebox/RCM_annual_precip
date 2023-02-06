#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jeb and 
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
from glob import glob
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import pandas as pd
from datetime import datetime 
from numpy.polynomial.polynomial import polyfit
from scipy import stats

AD=0
path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
if AD:path='/'
os.chdir(path)

#---------------- global plot settings

ly='x'

th=1
font_size=18
plt.rcParams['axes.facecolor'] = 'k'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams["font.size"] = font_size
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = "grey"

#---------------- 

years=np.arange(1958,2021).astype('str')

n_years=len(years)
#---------------- 

skip=8*12
fn='/Users/jason/Dropbox/CARRA/CARRA_rain/RCM_annual_precip/NCDC monthly NAO 1950-2021.csv'
NAO=pd.read_csv(fn,skiprows=skip)
NAO.columns=['YYYYMM','NAO']
NAO["date"]=pd.to_datetime(NAO['YYYYMM'], format='%Y%m')

NAO['year'] = pd.DatetimeIndex(NAO['date']).year
NAO['month'] = pd.DatetimeIndex(NAO['date']).month
NAO['month']=NAO['month'].astype(int)
# print(len(NAO))
# print(NAO.columns)

NAO_annual=[]

for year in years:
    # v=NAO.year.astype(str)==year
    v=( ((NAO.month.any()==1)or(NAO.month.any()==12)or(NAO.month.any()==2)))
        # v=( ((NAO.month==1)or(NAO.month==12)or(NAO.month==2)) and (NAO.year.astype(str)==year))

    print(year,np.sum(v))
    NAO_annual.append(np.mean(NAO.NAO[v]))

NAO_annual=np.array(NAO_annual)
    #%%

#----------------------------------------------- MAR
fn='./RCM_annual_precip/MAR Precipitation GrIS andPISM.xls'
skip=8
MAR = pd.read_excel(fn,skiprows=skip)
MAR.columns = ['year','sn_6','rf_6','E_6','sn_10','rf_10','E_10','sn_15','rf_15','E_15','sn_20','rf_20','E_20']
print(len(MAR))
print(MAR.columns)
MAR['tp_6']=MAR.sn_6+MAR.rf_6
MAR['tp_10']=MAR.sn_10+MAR.rf_10
MAR['tp_15']=MAR.sn_15+MAR.rf_15
MAR['tp_20']=MAR.sn_20+MAR.rf_20

#----------------------------------------------- NHM
fn='./RCM_annual_precip/NHM-SMAP_v1.01_1980-2020_GrIS-SMB_in_Gt.csv'
NHMi = pd.read_csv(fn)
fn='./RCM_annual_precip/NHM-SMAP_v1.01_1980-2020_PIMs-SMB_in_Gt.csv'
NHMp = pd.read_csv(fn) 

NHM=pd.DataFrame()
 # year,P,E,DSS,Rainfall
NHM["year"]=NHMp[' year']
NHM["sf"]=(NHMi.P-NHMi.Rainfall)-(NHMp.P-NHMp.Rainfall)
NHM["rf"]=(NHMi.Rainfall)+(NHMp.Rainfall)



#----------------------------------------------- CARRA
fn='./output_annual/tabulate_annual_CARRA.csv'
CARRA=pd.read_csv(fn)
print(len(CARRA))
print(CARRA.columns)


# Dropping last 2 rows using drop
# MAR.drop(MAR.tail(2).index,inplace = True)

vars=['sn','rf','tp']
varnams=['snowfall','rainfall']

ress=[6,10,15,20]

fig, ax = plt.subplots(2, figsize = [14,22])

RCM_name='MAR 3.11.5'
# RCM_name='MAR'
colors=['k','b','r','grey']
for i,var in enumerate(vars):
    for j,res in enumerate(ress):
        var_res=var+"_"+str(res)
        if i<2:
            ax[i].plot(MAR.year,MAR[var_res],label=RCM_name+' '+var_res[3:5]+' km',color=colors[j])
        v=MAR.year>1978
        x=MAR.year[v]
        y=MAR[var_res][v]
        b, m = polyfit(x,y, 1)
        xx=[np.min(x),np.max(x)]
        if i<2:
            ax[i].plot(xx,[m*xx[0]+b,m*xx[1]+b],c=colors[j],ls='--',linewidth=th+2)
        if i==0:ax[i].xaxis.set_ticklabels([])
        print("MAR "+var_res,m*42,m*42/(xx[0]*m+b))
        # v=np.where(MAR.year>1978)
        # z=NAO_annual[v[0]]
        # coefs=stats.pearsonr(z,y)
        # print(coefs)



ax[0].plot(CARRA.year,CARRA.tp-CARRA.rf,label='CARRA, 2.5 km',color='g')
ax[1].plot(CARRA.year,CARRA.rf,label='CARRA, 2.5 km',color='g')

#----------------------------------------------- NHM
color='m'
vars=['sf','rf']
for i,var in enumerate(vars):
    ax[i].plot(NHM.year,NHM[var],label='NHM-SMAP, 5.5 km',color=color)
    v=NHM.year>1979
    x=NHM.year[v]
    y=NHM[var][v]
    b, m = polyfit(x,y, 1)
    xx=[np.min(x),np.max(x)]
    ax[i].plot(xx,[m*xx[0]+b,m*xx[1]+b],c=color,ls='--',linewidth=th+2)
        
ax[0].set_ylabel('Gt y$^{-1}$')
ax[1].set_ylabel('Gt y$^{-1}$')


ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5),title='snowfall')
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5),title='rainfall')

plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)

