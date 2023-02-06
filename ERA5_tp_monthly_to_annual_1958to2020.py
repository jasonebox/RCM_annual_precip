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
import xarray as xr
AD=1



path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
if AD:
    path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_rain/'
    raw_path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA/'
    tool_path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_tools/'
os.chdir(path)

ni=721
nj=1440

niC=1269 ; njC=1069

#mask
fn='./ancil/ERA5_regional_masks_raster_1440x721.npy'
mask = np.fromfile(fn, dtype='int16')
mask=mask.reshape(ni, nj)
mask[((mask>1))]=0 #set 0 to non-greenland values, Greenland has a mask value of 1
 # read ice mask CARRA
fn='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = xr.open_dataset(fn)
maskC = np.array(nc2.z)

#load data
fn=raw_path+'./ERA5_tp_monthly_1958to1978.grib'
ds=xr.open_dataset(fn,engine='cfgrib')
tp78=ds.tp[:,:,:].values
time78=ds.time.values

lon=ds.longitude.values
lat=ds.latitude.values
lon_mesh, lat_mesh = np.meshgrid(ds.longitude.values, ds.latitude.values)

fn=raw_path+'./ERA5_tp_monthly_1979to2020.grib'
ds=xr.open_dataset(fn,engine='cfgrib')
tp20=ds.tp[:,:,:].values
time20=ds.time.values

# ERA5 data into CARRA grid
# load ERA5 to CARRA grid resampling key
fn=tool_path+'resampling_key_ERA5_to_CARRA.pkl'
infile = open(fn,'rb')
df_res=pickle.load(infile) 
  
def ERA5toCARRA(tpp, df_res, niC, njC):
    tp_res= tpp[df_res.col_e, df_res.row_e]
    tp_res=tp_res.reshape(niC, njC)
    return tp_res


df1 = pd.DataFrame()
df1['date']=pd.to_datetime(time78)
df1['year'] = pd.DatetimeIndex(df1['date']).year

df2 = pd.DataFrame()
df2['date']=pd.to_datetime(time20)
df2['year'] = pd.DatetimeIndex(df2['date']).year

#stack two datasets
df_all = pd.concat([df1,df2],ignore_index=True)
tp_all_monthly=np.vstack((tp78, tp20))  #in m -> m of water per day

# units conversion m/day to m/month
df_all['month'] = pd.DatetimeIndex(df_all['date']).month
m_range=np.zeros(len(df_all))
tp_all_monthly2=tp_all_monthly.copy()
m_range[-1]=31
for i in range(0,len(df_all)): 
    if i<755:
        m_range=(df_all.date[i+1]-df_all.date[i]).days
        tp_all_monthly2[i,:,:]*=m_range  #in m of water per month

tp_all = np.zeros((63, niC, njC))  
time_years = df_all.year.unique()[1:]
years=np.arange(1958, 2021)



i=0
for year in years: 
    tp_year=tp_all_monthly2[df_all.year==year]
    tpp=np.sum(tp_year, axis=0) #sum months
    tpp[mask==0]=np.nan #mask
    tpp=ERA5toCARRA(tpp, df_res, niC, njC)
    tpp[maskC==0]=np.nan
    tp_all[i,:,:]=tpp   #in m of water per year
    i+=1

#units conversion from m/y to Gt/year
areax=2500**2
tp_mean=np.nansum(tp_all, axis=(1,2))
tp_mean2=tp_mean *areax /1e9
plt.plot(tp_mean2)


#%% produce Dataframe
df_out=pd.DataFrame(years,columns= ['year'])
df_out["tp"]=tp_mean2

df_out.to_csv(path+'RCM_annual_precip/ERA5_1958to2020_yearly_tp.csv', index=False)


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