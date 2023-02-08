# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:34:42 2021

@author: Armin
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
from glob import glob
import pickle
# from netCDF4 import Dataset
import pandas as pd
from datetime import datetime 
from scipy.spatial import cKDTree
from calendar import monthrange
import matplotlib as mpl 
# import metpy.calc
# from metpy.units import units
import xarray as xr


#%%
AD=1


path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
if AD:
    path='/home/rmean/Dokumente/Work/GEUS/CARRA_rain/'
    raw_path='/home/rmean/Dokumente/Work/GEUS/CARRA/'
    tool_path='/home/rmean/Dokumente/Work/GEUS/CARRA_tools/'
else:
    tool_path='/Users/jason/Dropbox/CARRA/CARRA_tools/'
    
os.chdir(path)



niC=1269 ; njC=1069

 # read ice mask CARRA
fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat_mat=lat.reshape(niC, njC)

fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon_mat=lon.reshape(niC, njC)

fn='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = xr.open_dataset(fn)
mask = np.array(nc2.z)
mask_iceland=1; mask_svalbard=1; mask_jan_mayen=1    
if mask_jan_mayen:
    mask[((lon_mat-360>-15)&(lat_mat>66.6)&(lat_mat<75))]=0
if mask_iceland:
    mask[((lon_mat-360>-30)&(lat_mat<66.6))]=0
if mask_svalbard:
    mask[0:300,800:]=0
# mask[mask==0]=np.nan
maskC=mask.copy()

#load data
# fn=raw_path+'./MARv3.11.5-6km.nc'
# fn=raw_path+'./MARv3.11.5-10km.nc'
# fn=raw_path+'./MARv3.11.5-15km.nc'
# fn=raw_path+'./MARv3.11.5-20km.nc'
fn=raw_path+'withBS/MARv3.13.0-withBS-15km-yearly-ERA5-1950-2020_v5.nc' 
# fn='/Users/jason/0_dat/MARv3.13.0/withBS/MARv3.13.0-15km-daily-ERA5-1960.nc'
ds=xr.open_dataset(fn)
sf=ds.sf[:,:,:].values # mmWE/yr
rf=ds.rf[:,:,:].values # mmWE/yr
time=ds.sf.time.values

print(ds.variables)
#%%
ni=np.shape(sf)[1]
nj=np.shape(sf)[2]

lon=ds.lon.values
lat=ds.lat.values
# lon_mesh, lat_mesh = np.meshgrid(lon, lat)

# ERA5 data into CARRA grid
# load ERA5 to CARRA grid resampling key
fn=tool_path+'resampling_key_MARv3.13.0-15km_BS_to_CARRA.pkl'
infile = open(fn,'rb')
df_res=pickle.load(infile) 
  
#%%
def MARtoCARRA(data, df, res_x, res_y):
    data_out= data[df.row_m, df.col_m]
    data_out=data_out.reshape(res_x, res_y)
    return data_out


df1 = pd.DataFrame()
df1['date']=time

sf_all=np.zeros((np.shape(sf)[0], niC, njC))
rf_all=np.zeros((np.shape(rf)[0], niC, njC))
for i in range(len(time)):  #resample each year
    sff=sf[i]
    sff=MARtoCARRA(sff, df_res, niC, njC)
    sff[maskC==0]=np.nan
    sf_all[i,:,:]=sff   
    rff=rf[i]
    rff=MARtoCARRA(rff, df_res, niC, njC)
    rff[maskC==0]=np.nan
    rf_all[i,:,:]=rff


#units conversion from mm/y to Gt/year
areax=2500**2
sf_mean=np.nansum(sf_all, axis=(1,2))
sf_mean2=sf_mean *areax /1e12
rf_mean=np.nansum(rf_all, axis=(1,2))
rf_mean2=rf_mean *areax /1e12
# plt.plot(sf_mean2)

#%% produce Dataframe
df_out=pd.DataFrame(time,columns= ['year'])
# df_out["sf_15km"]=np.sum(sf_mean2,axis=0)
# df_out["rf_15km"]=np.sum(rf_mean2,axis=0)
# df_out["tp_15km"]=np.sum(df_out.rf_15km,axis=0) + np.sum(df_out.sf_15km,axis=0)
df_out["sf_15km"]=sf_mean2
df_out["rf_15km"]=rf_mean2
df_out["tp_15km"]=df_out.rf_15km+ df_out.sf_15km

df_out.to_csv(path+'RCM_annual_precip/MARv3.13.0_1950to2020_yearly.csv', index=False)




