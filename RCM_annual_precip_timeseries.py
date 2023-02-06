#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jeb and AD
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from netCDF4 import Dataset
AD=0
if AD:
    os.environ['PROJ_LIB'] = r'C:/Users/Armin/Anaconda3/pkgs/proj4-5.2.0-ha925a31_1/Library/share' #Armin needed to not get an error with import basemap: see https://stackoverflow.com/questions/52295117/basemap-import-error-in-pycharm-keyerror-proj-lib
import pandas as pd
from numpy.polynomial.polynomial import polyfit
from scipy import stats

path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
if AD:path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_rain/'
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

resampling=1  #CARRA and ERA5 are resampled only. 
no_resampling=0

t0=1958#1998
t1=2020


# selection=['all','peripheral']
# ss=1
#----------------------------------------------- MAR
# # MAR old
if no_resampling: 
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
    MAR_ress=[6,10,15,20]
    # # Dropping last 2 rows using drop
    # # MAR.drop(MAR.tail(2).index,inplace = True)

# MAR new resampled
if resampling: 
    fn=path+'RCM_annual_precip/MAR_1950to2020_yearly.csv'
    MAR=pd.read_csv(fn, skiprows=8) 
    MAR.columns = ['year','sn_6','rf_6','tp_6','sn_10','rf_10','tp_10','sn_15','rf_15','tp_15','sn_20','rf_20','tp_20']
    MAR_ress=[6]#,10,15,20]
    #for stats
    MAR_sf_stats=MAR.sn_6[MAR.year>=t0]
    MAR_rf_stats=MAR.rf_6[MAR.year>=t0]
    MAR_tp_stats=MAR.tp_6[MAR.year>=t0]


#----------------------------------------------- JRA-55
#JRA old
if no_resampling: 
    fn='./RCM_annual_precip/JRA-55_Greenland_precipitation.xlsx'
    JRA = pd.read_excel(fn,skiprows=1)
    JRA.columns = ['year','tp']
    print(len(JRA))
    print(MAR.columns)
    JRA['tp']=JRA.tp    #MAR.sn_6+MAR.rf_6

# JRA new resampled
if resampling: 
    fn='./RCM_annual_precip/JRA_1980to2020_yearly_tp.csv'
    JRA1 = pd.read_csv(fn)
    JRA1.columns = ['year','tp', 'sf']
    
    JRA=pd.DataFrame()
    JRA["year"]=JRA1.year
    JRA["sf"]=JRA1.sf
    JRA["rf"]=JRA1.tp-JRA1.sf
    JRA["tp"]= JRA1.tp
    
     #for stats
    JRA_sf_stats=JRA.sf[JRA.year>=t0]
    JRA_rf_stats=JRA.rf[JRA.year>=t0]
    JRA_tp_stats=JRA.tp[JRA.year>=t0]


#----------------------------------------------- NHM
#NHM old
if no_resampling: 
    fn='./RCM_annual_precip/NHM-SMAP_v1.01_1980-2020_GrIS-SMB_in_Gt.csv'
    NHMi = pd.read_csv(fn)
    fn='./RCM_annual_precip/NHM-SMAP_v1.01_1980-2020_PIMs-SMB_in_Gt.csv'
    NHMp = pd.read_csv(fn) 
    NHMp.columns
    NHM=pd.DataFrame()
      # year,P,E,DSS,Rainfall
    NHM["year"]=NHMp[' year']
    NHM["sf"]=(NHMi.P-NHMi.Rainfall)+(NHMp.P-NHMp.Rainfall)
    NHM["rf"]=(NHMi.Rainfall)+(NHMp.Rainfall)
    NHM["tp"]= NHMp.P+NHMi.P
    


# # NHM new resampled
if resampling: 
    fn='./RCM_annual_precip/NHM_1980to2020_yearly_tp.csv'
    NHM1 = pd.read_csv(fn)
    NHM1.columns = ['year','tp', 'sf']
    
    NHM=pd.DataFrame()
    NHM["year"]=NHM1.year
    NHM["sf"]=NHM1.sf
    NHM["rf"]=NHM1.tp-NHM1.sf
    NHM["tp"]= NHM1.tp
    
    #for stats
    NHM_sf_stats=NHM.sf[NHM.year>=t0]
    NHM_rf_stats=NHM.rf[NHM.year>=t0]
    NHM_tp_stats=NHM.tp[NHM.year>=t0]


#----------------------------------------------- CARRA
fn='./output_annual/tabulate_annual_CARRA.csv'
CARRA=pd.read_csv(fn)
CARRA["sf"]=CARRA.tp - CARRA.rf 
CARRA["sf"][CARRA["year"]==1995]=np.nan
CARRA["rf"][CARRA["year"]==1995]=np.nan
CARRA["tp"][CARRA["year"]==1995]=np.nan
print(len(CARRA))
print(CARRA.columns)
print(CARRA["rf"])
#%%
#for stats
CARRA_sf_stats=CARRA.sf[CARRA.year>=t0]
CARRA_rf_stats=CARRA.rf[CARRA.year>=t0]
CARRA_tp_stats=CARRA.tp[CARRA.year>=t0]

#----------------------------------------------- RACMO
#RACMO old
if no_resampling: 
    fn=path+'RCM_annual_precip/Box-components_RACMO2.3p2_ERA5_3h_1958-2020_1km_GrIS.txt'
    RACMOi1 = pd.read_csv(fn,delim_whitespace=(True))
    
    fn=path+'RCM_annual_precip/Box-components_RACMO2.3p2_ERA5_3h_1958-2020_1km_PIM.txt'
    RACMOp1 = pd.read_csv(fn,delim_whitespace=(True))
    
    fn=path+'RCM_annual_precip/Box-components_RACMO2.3p2_ERA5_3h_1958-2020_5km_GrIS.txt'
    RACMOi5 = pd.read_csv(fn,delim_whitespace=(True))
    
    fn=path+'RCM_annual_precip/Box-components_RACMO2.3p2_ERA5_3h_1958-2020_5km_PIM.txt'
    RACMOp5 = pd.read_csv(fn,delim_whitespace=(True))
    
    RACMO_ress=[1,5]
    
    # print(RACMOi1.columns)

    RACMOi1["sf_1"]=(RACMOi1.Precip-RACMOi1.Rainfall)+(RACMOp1.Precip-RACMOp1.Rainfall)
    RACMOi1["rf_1"]=(RACMOi1.Rainfall)+(RACMOp1.Rainfall)
    RACMOi1["tp_1"]=(RACMOi1.Precip)+(RACMOp1.Precip)

    RACMOi1["sf_5"]=(RACMOi5.Precip-RACMOi5.Rainfall)+(RACMOp5.Precip-RACMOp5.Rainfall)
    RACMOi1["rf_5"]=(RACMOi5.Rainfall)+(RACMOp5.Rainfall)
    RACMOi1["tp_5"]=(RACMOi5.Precip)+(RACMOp5.Precip)
    
    varnams=['snowfall','rainfall']
    
    RACMOi1.rename(columns={'Year':'year'}, inplace=True)

#RACMO new 
if resampling:
    #1km
    RACMO_ress=[5]#[1,5]
    fn='./RCM_annual_precip/RACMO1km_1958to2020_yearly.csv' #tp
    RACMOi1 = pd.read_csv(fn)
    RACMOi1.columns = ['year','tp', 'sf']
    RACMOi1["rf"]=RACMOi1.tp-RACMOi1.sf
    
    RACMOi1["sf_1"]=RACMOi1.sf
    RACMOi1["rf_1"]=RACMOi1.rf
    RACMOi1["tp_1"]=RACMOi1.tp
    
    #5km 
    fn='./RCM_annual_precip/RACMO5km_1958to2020_yearly.csv' #tp
    RACMOi2 = pd.read_csv(fn)
    RACMOi2.columns = ['year','tp', 'sf']
    RACMOi2["rf"]=RACMOi2.tp-RACMOi2.sf
    
    RACMOi1["sf_5"]=RACMOi2.sf
    RACMOi1["rf_5"]=RACMOi2.rf
    RACMOi1["tp_5"]=RACMOi2.tp
    
        #for stats
    RACMOi1_sf_stats=RACMOi1.sf_5[RACMOi1.year>=t0]
    RACMOi1_rf_stats=RACMOi1.rf_5[RACMOi1.year>=t0]
    RACMOi1_tp_stats=RACMOi1.tp_5[RACMOi1.year>=t0]
    
    
#----------------------------------------------- ERA5
#resampled only
fn='./RCM_annual_precip/ERA5_1958to2020_yearly_tp.csv' #tp
ERA5 = pd.read_csv(fn)
ERA5.columns = ['year','tp']

fn='./RCM_annual_precip/ERA5_1958to2020_yearly_sf.csv' #sf
dd = pd.read_csv(fn)
ERA5["sf"]=dd.tp #is labelled wrongly 'tp' in the .csv file -> but is snowfall
ERA5["rf"]=ERA5.tp - ERA5.sf    #rf

#for stats
ERA5_sf_stats=ERA5.sf[ERA5.year>=t0]
ERA5_rf_stats=ERA5.rf[ERA5.year>=t0]
ERA5_tp_stats=ERA5.tp[ERA5.year>=t0]

#DataFrame to summarize statistics of each model
statistics=pd.DataFrame(columns= ['model', 'variable', 'change in Gt', 'change in %', 'confidence (1-p)'])
kk=0

#----------------------------------------------- 
#----------------------------------------------- Figures
#----------------------------------------------- 

fig, ax = plt.subplots(3, figsize = [12,35])
trend_start=1980
Nx=4
slopes=np.zeros((3,Nx))
changes=np.zeros((3,Nx))
confidences=np.zeros((3,Nx))
ny=42

slopes_sf=[]
changes_sf=[]
confidences_sf=[]
slopes_rf=[]
changes_rf=[]
confidences_rf=[]
slopes_tp=[]
changes_tp=[]
confidences_tp=[]

vars2=['SF','RF','TP']

#----------------------------------------------- CARRA
C_trend=1
CARRA_name='CARRA, 2.5 km'
color='g'
vars=['sf','rf','tp']

if C_trend==1: 
    for i,var in enumerate(vars):
            var_res=var
            if var=='rf':
                CARRA[var_res]+=25
            v=((CARRA.year>=trend_start)&(~np.isnan(CARRA[var_res])))
            x=CARRA.year[v]
            y=CARRA[var_res][v]
                
            b, m = polyfit(x,y, 1)
            xx=[np.min(x),np.max(x)]
            coefs=stats.pearsonr(x,y)
            if i<3:
                sign='+'
                if m<0:sign=""
                lab=CARRA_name+' '+var_res[3:5]+\
                "\n\u0394"+vars2[i]+":"+sign+f'{(m*ny):.0f}'+" Gt y$^{-1}$, "+\
                sign+f'{((100*m*ny/(xx[0]*m+b))):.0f}'+"%, "+\
                "1-p:"+f'{(1-coefs[1]):.2f}'
                
                ax[i].plot(CARRA.year,CARRA[var],label=lab,color=color)
                ax[i].plot(xx,[m*xx[0]+b,m*xx[1]+b],c=color,ls='--',linewidth=th+2)
 
if C_trend==0:
    ax[0].plot(CARRA.year,CARRA.tp-CARRA.rf,label='CARRA, 2.5 km',color='g')
    ax[1].plot(CARRA.year,CARRA.rf,label='CARRA, 2.5 km',color='g')
    ax[2].plot(CARRA.year,CARRA.tp,label='CARRA, 2.5 km',color='g')
    

#----------------------------------------------- MAR
RCM_name='MAR 3.11.5'
RCM_name='MAR'
vars=['sn','rf','tp']
colors=['b','k','c','grey']
for i,var in enumerate(vars):
    for j,res in enumerate(MAR_ress):
        var_res=var+"_"+str(res)
        # print(var_res)
        v=MAR.year>=trend_start
        x=MAR.year[v]
        y=MAR[var_res][v]
        b, m = polyfit(x,y, 1)
        xx=[np.min(x),np.max(x)]
        coefs=stats.pearsonr(x,y)
        # if i==0:
        #     slopes_sf.append(m*ny)
        #     confidences_sf.append(1-coefs[1])
        #     changes_sf.append(m*ny/(xx[0]*m+b)*100)
        # if i==1:
        #     slopes_rf.append(m*ny)
        #     confidences_rf.append(1-coefs[1])
        #     changes_rf.append(m*ny/(xx[0]*m+b)*100)
        # if i==2:
        #     slopes_tp.append(m*ny)
        #     confidences_tp.append(1-coefs[1])
        #     changes_tp.append(m*ny/(xx[0]*m+b)*100)
        if i<3:
            sign='+'
            if m<0:sign=""
            lab=RCM_name+' '+var_res[3:5]+\
            ' km \n '+\
            "\u0394"+vars2[i]+":"+sign+f'{(m*ny):.0f}'+" Gt y$^{-1}$, "+\
            sign+f'{((100*m*ny/(xx[0]*m+b))):.0f}'+"%, "+\
            "1-p:"+f'{(1-coefs[1]):.2f}'
            ax[i].plot(MAR.year,MAR[var_res],label=lab,color=colors[j])
            ax[i].plot(xx,[m*xx[0]+b,m*xx[1]+b],c=colors[j],ls='--',linewidth=th+2)
        if i==0:
            ax[i].xaxis.set_ticklabels([])

        # z=NAO_annual[v[0]]
        # coefs=stats.pearsonr(z,y)
        # print(coefs)
        
        statistics.loc[kk]=pd.Series({'model':"MAR"+str(res), 'variable':var, 'change in Gt':f'{(m*ny):.0f}', 'change in %':f'{((100*m*ny/(xx[0]*m+b))):.1f}', 'confidence (1-p)':f'{(1-coefs[1]):.2f}'}); kk+=1


#----------------------------------------------- RACMO
RCM_name='RACMO'
colors=['orange','r']
vars=['sf','rf','tp']

for i,var in enumerate(vars):
    for j,res in enumerate(RACMO_ress):
        var_res=var+"_"+str(res)
        v=RACMOi1.year>=trend_start
        x=RACMOi1.year[v]
        y=RACMOi1[var_res][v]
        b, m = polyfit(x,y, 1)
        xx=[np.min(x),np.max(x)]
        coefs=stats.pearsonr(x,y)
        # if i==0:
        #     slopes_sf.append(m*ny)
        #     confidences_sf.append(1-coefs[1])
        #     changes_sf.append(m*ny/(xx[0]*m+b)*100)
        # if i==1:
        #     slopes_rf.append(m*ny)
        #     confidences_rf.append(1-coefs[1])
        #     changes_rf.append(m*ny/(xx[0]*m+b)*100)
        # if i==2:
        #     slopes_tp.append(m*ny)
        #     confidences_tp.append(1-coefs[1])
        #     changes_tp.append(m*ny/(xx[0]*m+b)*100)
        if i<3:
            sign='+'
            if m<0:sign=""
            lab=RCM_name+' '+var_res[3:5]+\
            ' km \n '+\
            "\u0394"+vars2[i]+":"+sign+f'{(m*ny):.0f}'+" Gt y$^{-1}$, "+\
            sign+f'{((100*m*ny/(xx[0]*m+b))):.0f}'+"%, "+\
            "1-p:"+f'{(1-coefs[1]):.2f}'
            ax[i].plot(RACMOi1.year,RACMOi1[var_res],label=lab,color=colors[j])
            ax[i].plot(xx,[m*xx[0]+b,m*xx[1]+b],c=colors[j],ls='--',linewidth=th+2)
        if i==0:ax[i].xaxis.set_ticklabels([])
        print("RACMO"+str(res)+'km',var,
        "change: "+f'{(m*ny):.0f}'+" Gt,"\
        "or "+f'{((m*ny/(xx[0]*m+b))):.1f}'+" %,"\
        "confidence (1-p): "+f'{(1-coefs[1]):.2f}'
        )        # v=np.where(RACMOi1.year>1978)
        # z=NAO_annual[v[0]]
        # coefs=stats.pearsonr(z,y)
        # print(coefs)
        statistics.loc[kk]=pd.Series({'model':"RACMO"+str(res), 'variable':var, 'change in Gt':f'{(m*ny):.0f}', 'change in %':f'{((m*ny/(xx[0]*m+b))):.1f}', 'confidence (1-p)':f'{(1-coefs[1]):.2f}'}); kk+=1

#----------------------------------------------- NHM
color='m'
vars=['sf','rf','tp']
for i,var in enumerate(vars):
    for j in range(1):
        v=NHM.year>=trend_start
        x=NHM.year[v]
        y=NHM[var][v]
        b, m = polyfit(x,y, 1)
        xx=[np.min(x),np.max(x)]
        coefs=stats.pearsonr(x,y)
        # if i==0:
        #     slopes_sf.append(m*ny)
        #     confidences_sf.append(1-coefs[1])
        #     changes_sf.append(m*ny/(xx[0]*m+b)*100)
        # if i==1:
        #     slopes_rf.append(m*ny)
        #     confidences_rf.append(1-coefs[1])
        #     changes_rf.append(m*ny/(xx[0]*m+b)*100)
        # if i==2:
        #     slopes_tp.append(m*ny)
        #     confidences_tp.append(1-coefs[1])
        #     changes_tp.append(m*ny/(xx[0]*m+b)*100)
        if i<3:
            sign='+'
            if m<0:sign=""
            lab='NHM-SMAP 5 km \n'+\
            "\u0394"+vars2[i]+":"+sign+f'{(m*ny):.0f}'+" Gt y$^{-1}$, "+\
            sign+f'{((100*m*ny/(xx[0]*m+b))):.0f}'+"%, "+\
            "1-p:"+f'{(1-coefs[1]):.2f}'
            ax[i].plot(NHM.year,NHM[var],label=lab,color=color)
            ax[i].plot(xx,[m*xx[0]+b,m*xx[1]+b],c=color,ls='--',linewidth=th+2)
        # slopes[i,j]=m*ny
        # confidence (1-p)s[i,j]=1-coefs[1]
        # changes[i,j]=m*ny/(xx[0]*m+b)*100
        # print("NHM-SMAP ",var,m*ny,m*ny/(xx[0]*m+b),1-coefs[1])
        print("NHM-SMAP ",var,
        "change: "+f'{(m*ny):.0f}'+" Gt,"\
        "or "+f'{((m*ny/(xx[0]*m+b))):.1f}'+" %,"\
        "confidence (1-p): "+f'{(1-coefs[1]):.2f}'
        )
        statistics.loc[kk]=pd.Series({'model':"NHM-SMAP", 'variable':var, 'change in Gt':f'{(m*ny):.0f}', 'change in %':f'{((m*ny/(xx[0]*m+b))):.1f}', 'confidence (1-p)':f'{(1-coefs[1]):.2f}'}); kk+=1

#----------------------------------------------- JRA
color='gray'
if resampling: vars=['sf','rf','tp']
if no_resampling: vars=['tp']
JRA[var]-=40
for i,var in enumerate(vars):
    for j in range(1):
        v=JRA.year>=trend_start
        x=JRA.year[v]
        y=JRA[var][v]
        b, m = polyfit(x,y, 1)
        xx=[np.min(x),np.max(x)]
        coefs=stats.pearsonr(x,y)
        if i==0:
            slopes_sf.append(m*ny)
            confidences_sf.append(1-coefs[1])
            changes_sf.append(m*ny/(xx[0]*m+b)*100)
        if i==1:
            slopes_rf.append(m*ny)
            confidences_rf.append(1-coefs[1])
            changes_rf.append(m*ny/(xx[0]*m+b)*100)
        if i==2:
            slopes_tp.append(m*ny)
            confidences_tp.append(1-coefs[1])
            changes_tp.append(m*ny/(xx[0]*m+b)*100)
        if i<3:
            sign='+'
            if m<0:sign=""
            lab='JRA-55 c.50 km \n'+\
            "\u0394"+vars2[i]+":"+sign+f'{(m*ny):.0f}'+" Gt y$^{-1}$, "+\
            sign+f'{((100*m*ny/(xx[0]*m+b))):.0f}'+"%, "+\
            "1-p:"+f'{(1-coefs[1]):.2f}'
            ax[i].plot(JRA.year,JRA[var],label=lab,color=color)
            ax[i].plot(xx,[m*xx[0]+b,m*xx[1]+b],c=color,ls='--',linewidth=th+2)
        # slopes[i,j]=m*ny
        # confidence (1-p)s[i,j]=1-coefs[1]
        # changes[i,j]=m*ny/(xx[0]*m+b)*100
        # print("JRA-SMAP ",var,m*ny,m*ny/(xx[0]*m+b),1-coefs[1])
        print("JRA ",var,
        "change: "+f'{(m*ny):.0f}'+" Gt,"\
        "or "+f'{((m*ny/(xx[0]*m+b))):.1f}'+" %,"\
        "confidence (1-p): "+f'{(1-coefs[1]):.2f}'
        )
        statistics.loc[kk]=pd.Series({'model':"JRA", 'variable':var, 'change in Gt':f'{(m*ny):.0f}', 'change in %':f'{((m*ny/(xx[0]*m+b))):.1f}', 'confidence (1-p)':f'{(1-coefs[1]):.2f}'}); kk+=1

#----------------------------------------------- ERA5
color='k'
vars=['sf','rf','tp']
ERA5[var]-=40
for i,var in enumerate(vars):
    for j in range(1):
        v=ERA5.year>=trend_start
        x=ERA5.year[v]
        y=ERA5[var][v]
        b, m = polyfit(x,y, 1)
        xx=[np.min(x),np.max(x)]
        coefs=stats.pearsonr(x,y)
        if i==0:
            slopes_sf.append(m*ny)
            confidences_sf.append(1-coefs[1])
            changes_sf.append(m*ny/(xx[0]*m+b)*100)
        if i==1:
            slopes_rf.append(m*ny)
            confidences_rf.append(1-coefs[1])
            changes_rf.append(m*ny/(xx[0]*m+b)*100)
        if i==2:
            slopes_tp.append(m*ny)
            confidences_tp.append(1-coefs[1])
            changes_tp.append(m*ny/(xx[0]*m+b)*100)
        if i<3:
            sign='+'
            if m<0:sign=""
            lab='ERA5 31 km \n '+\
            "\u0394"+vars2[i]+":"+sign+f'{(m*ny):.0f}'+" Gt y$^{-1}$,"+\
            sign+f'{((100*m*ny/(xx[0]*m+b))):.0f}'+"%, "+\
            "1-p:"+f'{(1-coefs[1]):.2f}'
            ax[i].plot(ERA5.year,ERA5[var],label=lab,color=color)
            ax[i].plot(xx,[m*xx[0]+b,m*xx[1]+b],c=color,ls='--',linewidth=th+2)

        # slopes[i,j]=m*ny
        # confidence (1-p)s[i,j]=1-coefs[1]
        # changes[i,j]=m*ny/(xx[0]*m+b)*100
        # print("JRA-SMAP ",var,m*ny,m*ny/(xx[0]*m+b),1-coefs[1])
        print("ERA5",var,
        "change: "+f'{(m*ny):.0f}'+" Gt,"\
        "or "+f'{((m*ny/(xx[0]*m+b))):.1f}'+" %,"\
        "confidence (1-p): "+f'{(1-coefs[1]):.2f}'
        )
        statistics.loc[kk]=pd.Series({'model':"ERA5", 'variable':var, 'change in Gt':f'{(m*ny):.0f}', 'change in %':f'{((m*ny/(xx[0]*m+b))):.1f}', 'confidence (1-p)':f'{(1-coefs[1]):.2f}'}); kk+=1


#-------------------------------------------- ensemble
#define ensemble table
#to change ensemble start date change t0!
color='r'
models=['MAR', 'NHM', 'RACMOi1', 'JRA', 'ERA5'] #, 'CARRA']  -> CARRA excluded
mean_dat=np.zeros((t1-t0+1,len(models)))*np.nan
ens_year=np.arange(t0,t1+1)
for i,var in enumerate(vars):  
    for m, mod in enumerate(models):
        coefs=stats.pearsonr(x,y)
        nam=mod+'_'+var+'_stats'
        mean_dat[(len(ens_year)-len(eval(nam))):,m]=np.array(eval(nam))
    ensemble=np.nanmean(mean_dat, axis=1)
    v=ens_year>=trend_start
    x=ens_year[v]
    y=ensemble[v]
    b, m = polyfit(x,y, 1)
    xx=[np.min(x),np.max(x)]
    coefs=stats.pearsonr(x,y)
        # b, m = polyfit(ens_year,ensemble, 1)
    # xx=[np.min(ens_year),np.max(ens_year)]
    # coefs=stats.pearsonr(ens_year,ensemble)
    sign='+'
    lab='ensemble mean \n'+\
    "\u0394"+vars2[i]+":"+sign+f'{(m*ny):.0f}'+" Gt y$^{-1}$,"+\
    sign+f'{((100*m*ny/(xx[0]*m+b))):.0f}'+"%, "+\
    "1-p:"+f'{(1-coefs[1]):.2f}'
    
    ax[i].plot(ens_year,ensemble,label=lab,color=color)
    ax[i].plot(xx,[m*xx[0]+b,m*xx[1]+b],c=color,ls='--',linewidth=th+2)


ax[0].set_ylabel('snowfall, Gt y$^{-1}$')
ax[1].set_ylabel('rainfall, Gt y$^{-1}$')
ax[2].set_ylabel('total precipitation, Gt y$^{-1}$')
for i in range(3):ax[i].set_xlim([1957,2021])

mult=0.8 ; yy0=1.02
ax[0].legend(loc='upper left', bbox_to_anchor=(1, yy0),fontsize=font_size*mult)
ax[1].legend(loc='upper left', bbox_to_anchor=(1, yy0),fontsize=font_size*mult)
ax[2].legend(loc='upper left', bbox_to_anchor=(1, yy0),fontsize=font_size*mult)

ax[1].set(xticklabels=[]) #remove x-axis labels

plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)

#write out statistics table
statistics.to_csv(path+'RCM_annual_precip/statistics_all_models.csv', index=False)


## annotate mean trends

varnams2=['snowfall','rainfall','total precipitation']

mult=0.9
xx0=0.01 ; yy0=0.95 ; dy=0.

for i in range(3):
    if i==0:
        msg=varnams2[i]+\
          " trend (ERA5 & JRA): "+f'{(np.mean(slopes_sf)):.0f}'+"±"+f'{(np.std(slopes_sf)):.0f}'+" Gt y$^{-1}$,"\
          " +"+f'{(np.mean(changes_sf)):.0f}'+"±"+f'{(np.std(changes_sf)):.0f}'+" %,"\
          " 1-p: "+f'{(np.mean(confidences_sf)):.2f}'+" ± "+f'{(np.std(confidences_sf)):.2f}'
          # +\
          #     ', N: '+f'{(len(confidences_sf)):.0f}'
    if i==1:
        msg=varnams2[i]+\
          " trend (ERA5 & JRA): "+f'{(np.mean(slopes_rf)):.0f}'+"±"+f'{(np.std(slopes_rf)):.0f}'+" Gt y$^{-1}$,"\
          " +"+f'{(np.mean(changes_rf)):.0f}'+"±"+f'{(np.std(changes_rf)):.0f}'+" %,"\
          " 1-p: "+f'{(np.mean(confidences_rf)):.2f}'+" ± "+f'{(np.std(confidences_rf)):.2f}'
          # +\
          #     ', N: '+f'{(len(confidences_sf)):.0f}'
    if i==2:
        msg=varnams2[i]+\
          " trend (ERA5 & JRA): "+f'{(np.mean(slopes_tp)):.0f}'+"±"+f'{(np.std(slopes_tp)):.0f}'+" Gt y$^{-1}$,"\
          " +"+f'{(np.mean(changes_tp)):.0f}'+"±"+f'{(np.std(changes_tp)):.0f}'+" %,"\
          " 1-p: "+f'{(np.mean(confidences_tp)):.2f}'+" ± "+f'{(np.std(confidences_tp)):.2f}'
          # +\
          #    ", N: "+f'{(len(confidences_sf)):.0f}'
              
    print(msg)
    plt.text(xx0, yy0+i*dy,msg, fontsize=font_size*mult,transform=ax[i].transAxes, color='k')
    
#%% statistical investigation


models=['MAR', 'NHM', 'RACMOi1', 'JRA', 'ERA5', 'CARRA']
# all_models=np.zeros((len(MAR_sf_stats)))

for var in vars:  
    for m in models:
        stat_nam=m+'_'+var+'_stats'
        stat_data=eval(stat_nam)
        # all_models = np.append(all_models, stat_data, axis=0)

        #conversion Gt into mm/yr
        area=1804032000000 #m2
        stat_data_conv=stat_data*1e12/area 
        
        print(m, var, 'average ', np.mean(stat_data_conv))
