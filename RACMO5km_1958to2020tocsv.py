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
import metpy.calc
from metpy.units import units
import gzip
import xarray as xr
AD=1



path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
if AD:
    path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_rain/'
    raw_path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA/'
    tool_path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_tools/'
os.chdir(path)

ni=566; nj=438

niC=1269 ; njC=1069

years=np.arange(1958,2021).astype(str)
vars=['prec', 'snow']

# read ice mask CARRA
fn='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = xr.open_dataset(fn)
mask = np.array(nc2.z)

fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat_mat=lat.reshape(niC, njC)

fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon_mat=lon.reshape(niC, njC)

mask_iceland=1; mask_svalbard=1 ; mask_jan_mayen=1
       
if mask_jan_mayen:
    mask[((lon_mat-360>-15)&(lat_mat>66.6)&(lat_mat<75))]=0
if mask_iceland:
    mask[((lon_mat-360>-30)&(lat_mat<66.6))]=0
if mask_svalbard:
    mask[0:300,800:]=0
# mask[mask==0]=np.nan
maskC=mask.copy()

#load RACMO 5km to CARRA grid resampling key
fn=tool_path+'resampling_key_RACMO5km_to_CARRA.pkl'
infile = open(fn,'rb')
df_res=pickle.load(infile) #upsampling

def RACMOtoCARRA(tpp, df_res, niC, njC):
    tp_res= tpp[df_res.row_r, df_res.col_r]
    tp_res=tp_res.reshape(niC, njC)
    return tp_res

#load data

fn=raw_path+'precip.1958-2020.FGRN055_BN_RACMO2.3p2_ERA5_3h_FGRN055.YY.nc.gz'
with gzip.open(fn) as gz:
    ds=xr.open_dataset(gz,decode_times=False)
    tp_R=np.array(ds.precip[:,0,:,:]) #total precip in mmWE

fn=raw_path+'snowfall.1958-2020.FGRN055_BN_RACMO2.3p2_ERA5_3h_FGRN055.YY.nc.gz'
with gzip.open(fn) as gz:
    ds=xr.open_dataset(gz,decode_times=False)
    sf_R=np.array(ds.snowfall[:,0,:,:]) #snowfall in mmWE


tp_5820_C=np.zeros((len(years),niC, njC))
sf_5820_C=np.zeros((len(years),niC, njC))

#resample into CARRA
for j, year in enumerate(years):
    tpp=tp_R[j]
    tpp=RACMOtoCARRA(tpp, df_res, niC, njC)
    tpp[maskC==0]=np.nan
    tp_5820_C[j,:,:]=tpp
        
    sff=sf_R[j]
    sff=RACMOtoCARRA(sff, df_res, niC, njC)
    sff[maskC==0]=np.nan
    sf_5820_C[j,:,:]=sff

#units conversion into Gt
areax=2500**2
tp_mean=np.nansum(tp_5820_C, axis=(1,2))
tp_mean2=tp_mean *areax /1e12 

sf_mean=np.nansum(sf_5820_C, axis=(1,2))
sf_mean2=sf_mean *areax /1e12 

plt.plot(sf_mean2)
# plt.imshow(tp_5820_C[12])

#%% Difference between resamplings

# diff=sss[0,:,:]-ssss[0,:,:]
# print(np.nanmax(diff))
# print(np.nanmin(diff))
# sss.any()==ssss.any()


#%% produce Dataframe

df_out=pd.DataFrame(years,columns= ['year'])
df_out["tp"]=tp_mean2
df_out["sf"]=sf_mean2

df_out.to_csv(path+'RCM_annual_precip/RACMO5km_1958to2020_yearly.csv', index=False)


#%% produce ncfile
# from netCDF4 import Dataset,num2date
# outpath='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA/'
# ofile=outpath+'ERA5_tp_annual_1958to2020v10.nc'
# data=tp_all
# n_years=len(time_years)
# print("start making .nc file")
# # os.system("/bin/rm "+ofile)
# ncfile = Dataset(ofile,mode='w',format='NETCDF4_CLASSIC')
# lat_dim = ncfile.createDimension('lat', nj)     # latitude axis
# lon_dim = ncfile.createDimension('lon', ni)    # longitude axis
# time_dim = ncfile.createDimension('time', n_years) # unlimited axis (can be appended to)

# ncfile.subtitle="subtitle"


# latitude = ncfile.createVariable('latitude', np.float32, ('lon','lat'))
# latitude.units = 'degrees_north'
# latitude.long_name = 'latitude'
# longitude = ncfile.createVariable('longitude', np.float32, ('lon','lat'))
# longitude.units = 'degrees_east'
# longitude.long_name = 'longitude'
# time = ncfile.createVariable('time', np.float64, ('time',))
# # time.units = 'days since '+year+'-01-01'
# time.units = 'year'
# time.long_name = 'time'
# # Define a 3D variable to hold the data
# print("compressing")
# temp = ncfile.createVariable('tp',np.float32,('time','lon','lat'),zlib=True,least_significant_digit=3) # note: unlimited dimension is leftmost
# temp.units = 'my-1' # degrees Kelvin
# temp.standard_name = 'total precipitation' # this is a CF standard name

# nlats = len(lat_dim); nlons = len(lon_dim); ntimes = 3

# latitude[:,:]=lat_mesh
# longitude[:,:]=lon_mesh
# time[:]=time_years
# temp[:,:,:] = data  # Appends data along unlimited dimension


# print("-- Wrote data, temp.shape is now ", temp.shape)
# print(ofile)
   
# ncfile.close(); print('Dataset is closed!')

#%%

# fn=raw_path+'./ERA5_tp_annual_1958to2020v10.nc'
# dss=xr.open_dataset(fn)