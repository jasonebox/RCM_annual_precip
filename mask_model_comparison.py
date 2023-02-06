# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 09:56:48 2021

@author: Armin
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from pyproj import Proj, transform
import pandas as pd
import xarray as xr
import pickle
import gzip
AD=1
if AD:
    os.environ['PROJ_LIB'] = r'C:/Users/Armin/Anaconda3/pkgs/proj4-5.2.0-ha925a31_1/Library/share' #Armin needed to not get an error with import basemap: see https://stackoverflow.com/questions/52295117/basemap-import-error-in-pycharm-keyerror-proj-lib
from mpl_toolkits.basemap import Basemap


if AD:
    path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_rain/'
    raw_path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA/'
    tool_path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_tools/'
os.chdir(path)


#Basemap 
# ni=1269 ; nj=1069
# fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
# lat=np.fromfile(fn, dtype=np.float32)
# lat_mat=lat.reshape(ni, nj)
# fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
# lon=np.fromfile(fn, dtype=np.float32)
# lon_mat=lon.reshape(ni, nj)
# LLlat=lat_mat[0,0]
# LLlon=lon_mat[0,0]-360
# lon0=lon_mat[int(round(ni/2)),int(round(nj/2))]-360
# lat0=lat_mat[int(round(ni/2)),int(round(nj/2))]         
# URlat=lat_mat[ni-1,nj-1]
# URlon=lon_mat[ni-1,nj-1]
# m = Basemap(llcrnrlon=LLlon, llcrnrlat=LLlat, urcrnrlon=URlon, urcrnrlat=URlat, lat_0=lat0, lon_0=lon0, resolution='h', projection='lcc')


#CARRA mask
fn='./ancil/CARRA_W_domain_ice_mask.nc'
nc2 = xr.open_dataset(fn)
maskC = np.array(nc2.z)
mask_iceland=1; mask_svalbard=1; mask_jan_mayen=1
       
if mask_jan_mayen:
    maskC[((lon_mat-360>-15)&(lat_mat>66.6)&(lat_mat<75))]=0
if mask_iceland:
    maskC[((lon_mat-360>-30)&(lat_mat<66.6))]=0
if mask_svalbard:
    maskC[0:300,800:]=0
maskC[maskC==0]=np.nan

fn='./ancil/mask_peri_glaciers_ice_caps_1269x1069.npy'
maskC_peri = np.load(fn)
maskC_peri=np.rot90(maskC_peri.T)


#MAR mask  --> here 1km mask but we use 6km MAR at the best
fn=raw_path+'common_ice_mask.nc4'
ds=xr.open_dataset(fn)
msk2=ds.MSK2 #with peri
msk3=ds.MSK3 #w/o peri
maskM=np.array(msk3)
# mask[mask==0]=np.nan
plt.imshow(maskM)
fn=raw_path+'GRD-1km.nc4'
ds2=xr.open_dataset(fn)
ga_MAR=ds2.AREA.values

maskM_peri=np.array(msk3.copy())
maskM_peri[np.array(msk2)!=0]=0


#RACMO 5km
fn=raw_path+'FGRN055_Masks_5.5km.nc.gz'
with gzip.open(fn) as gz:
    ds=xr.open_dataset(gz,decode_times=False)
    ga_RACMO=ds.Area.values
    maskR=ds.Icemask_GR.values
    maskR_peri=ds.Promicemask.values
maskR_peri[maskR_peri==4]=0; maskR_peri[maskR_peri==1]=0; maskR_peri[maskR_peri==3]=0 #remove non peri values
maskR_peri[maskR_peri!=0]=1 #set all mask values to 1

#ERA 5
# fn=path+'ERA5_mask/land_mask_GrIS_ERA5.npy'
# maskE=np.load(fn)
niE=1440 ; njE=721
fn=path+'./ancil/ERA5_regional_masks_raster_1440x721.npy'
maskE = np.fromfile(fn, dtype='int16')
maskE=maskE.reshape(njE, niE)
maskE[((maskE>1))]=0 #set 0 to non-greenland values, Greenland has a mask value of 1




#%%calculate area
models=['CARRA', 'MAR', 'RACMO', 'ERA5']
all_ice=np.zeros((len(models)))
peri=np.zeros((len(models)))

def calc_area(grid_area, mask):
    grid_nr=np.nansum(mask) #empty values in mask need to be nans
    area=grid_nr*grid_area
    return area

#CARRA 2.5km
areaC=calc_area(2.5**2, maskC)
areaC_peri=calc_area(2.5**2, maskC_peri)
all_ice[0]='%.0f'% areaC
peri[0]='%.0f'% areaC_peri
print(areaC)
# print(areaC_peri)

#MAR  1km
# areaM=calc_area(1e3**2, maskM)
areaM=np.nansum(ga_MAR*maskM)
areaM_peri=np.nansum(ga_MAR*maskM_peri)
all_ice[1]='%.0f'% areaM
peri[1]='%.0f'% areaM_peri
print(areaM)
# print(areaM_peri)

#NHM -> no mask available

#RACMO 5km
areaR=np.nansum(ga_RACMO*maskR)
areaR_peri=np.nansum(ga_RACMO*maskR_peri)
all_ice[2]='%.0f'% areaR
peri[2]='%.0f'% areaR_peri
print(areaR)
# print(areaR_peri)


# #ERA5 
#load ERA5 to CARRA grid resampling key
fn=tool_path+'resampling_key_ERA5_to_CARRA.pkl' #resampling, since not every grid has the same area?!
infile = open(fn,'rb')
df_res=pickle.load(infile)
maskE1= maskE[df_res.col_e, df_res.row_e]
maskE2=maskE1.reshape(ni, nj)
areaE=calc_area(2.5**2, maskE2) 
all_ice[3]='%.0f'% areaE
print(areaE)
#%% Create table output

df_masks = pd.DataFrame(models)
df_masks['all_ice']= all_ice
df_masks['peri']= peri
df_masks['all']= peri+all_ice
# df_masks.to_csv(path+'RCM_annual_precip/model_masks_area.csv')
