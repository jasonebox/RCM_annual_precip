# -*- coding: utf-8 -*-
"""
@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)
   
"""

import rasterio
import numpy as np
import pandas as pd
import os
from pyproj import Proj, transform
import xarray as xr
from datetime import datetime, timedelta
import glob
from scipy.spatial import cKDTree

# %% paths

AW = 1

base_path = '/Users/jason/Dropbox/CARRA/CARRA_rain/'

if AW:
    base_path = 'C:/Users/Pascal/Desktop/GEUS_2019/SICE_AW_JEB/SICE_AW_JEB/'\
    + 'CARRA_rain/'

os.chdir(base_path)

# %% CARRA coordinates

def lon360_to_lon180(lon360):

    # reduce the angle  
    lon180 =  lon360 % 360 
    
    # force it to be the positive remainder, so that 0 <= angle < 360  
    lon180 = (lon180 + 360) % 360;  
    
    # force into the minimum absolute value residue class, so that -180 < angle <= 180  
    lon180[lon180 > 180] -= 360
    
    return lon180

# CARRA West grid dims
ni = 1269 ; nj = 1069

# read lat lon arrays
fn = './ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat = np.fromfile(fn, dtype=np.float32)
clat_mat = lat.reshape(ni, nj)

fn = './ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon = np.fromfile(fn, dtype=np.float32)
lon_pn = lon360_to_lon180(lon)
clon_mat = lon_pn.reshape(ni, nj) 

fn='./ancil/CARRA_W_domain_ice_mask.nc'
ds=xr.open_dataset(fn)
mask = np.array(ds.z)

# %% reproject 4326 CARRA coordinates to 3413 

inProj = Proj(init='epsg:4326')
outProj = Proj(init='epsg:3413')

x1, y1 = clon_mat.flatten(), clat_mat.flatten()
cx, cy = transform(inProj, outProj, x1, y1)
cx_mat = cx.reshape(ni, nj) 
cy_mat = cy.reshape(ni, nj)

cols, rows = np.meshgrid(np.arange(np.shape(clat_mat)[1]), 
                         np.arange(np.shape(clat_mat)[0]))

CARRA_positions = pd.DataFrame({'rowc': rows.ravel(),
                                'colc': cols.ravel(),
                                'xc': cx_mat.ravel(),
                                'yc': cy_mat.ravel(),
                                'maskc': mask.flatten()})


# %% load RCM coordinates

RCM_reader = rasterio.open('H:/RCM/RACMO/2019/2019_202.tif')
RCM_data = RCM_reader.read(1)

RCM_reader.xy(np.shape(RCM_data)[0], np.shape(RCM_data)[1])

cols2, rows2 = np.meshgrid(np.arange(np.shape(RCM_data)[1]), 
                           np.arange(np.shape(RCM_data)[0]))

x_m, y_m = RCM_reader.xy(rows2.flatten(), cols2.flatten())


# %% nearest neighbours

nA = np.column_stack((cx_mat.ravel(), cy_mat.ravel()))
    
nB = np.column_stack((x_m, y_m))

btree = cKDTree(nA)
dist, idx = btree.query(nB, k=1)

CARRA_cells_for_RCM = CARRA_positions.iloc[idx]
CARRA_cells_for_RCM['xm'] = x_m
CARRA_cells_for_RCM['ym'] = y_m
CARRA_cells_for_RCM['rowm'] = rows2.flatten()
CARRA_cells_for_RCM['colm'] = cols2.flatten()

CARRA_cells_for_RCM.to_csv('C:/Users/Pascal/Desktop/GEUS_2019/SICE_AW_JEB/' 
                                     + 'SICE_AW_JEB/CARRA_cells_for_RCM.csv')

CARRA_cells_for_RCM = pd.read_csv('C:/Users/Pascal/Desktop/GEUS_2019/SICE_AW_JEB/' 
                                    + 'SICE_AW_JEB/CARRA_cells_for_RCM.csv')

# %% CARRA RCM colocation

CARRA_files = glob.glob('H:/CARRA/rf_*.nc')
CARRA_file = 'H:/CARRA/rf_2012.nc'

results = pd.DataFrame()

day = datetime(2017, 9, 14, 0, 0)
i = 256

nb_ROS_events = []

for CARRA_file in CARRA_files:
    
    ds = xr.open_dataset(CARRA_file)
    
    year = int(CARRA_file.split(os.sep)[-1].split('.')[0].split('_')[-1])
    
    time = np.arange(datetime(year, 1, 1), datetime(year + 1, 1, 1), 
                         timedelta(days=1)).astype(datetime)
    
    RCM_files = glob.glob('H:/RCM/RACMO/' + str(year) + '/*.tif')
    
    annual_results = pd.DataFrame()
    
    for i, day in enumerate(time):
        
        # ------ CARRA 
        
        print(day)
        
        daily_rf = np.array(ds.rf[i, :, :])
        
        # just for information
        lon_rain = clon_mat[daily_rf > 0]
        lat_rain = clat_mat[daily_rf > 0]
        
        cx_rain = cx_mat[daily_rf > 0]
        cy_rain = cy_mat[daily_rf > 0]
        
        # try also with doy - 1 to not have the effect of rain on albedo
        doy = day.strftime("%j")
        
        # ------ RCM
        
        RCM_files_mask = [doy in file for file in RCM_files]
        
        # if np.sum(RCM_files_mask) == 0:
        #     continue
        
        RCM_file_matching_CARRA_date = np.array(RCM_files)[RCM_files_mask]
        
        RCM_reader = rasterio.open(RCM_file_matching_CARRA_date[0])
        RCM_data = RCM_reader.read(1)
        
        
        # get RCM pixels where there is rain from CARRA and bare ice in RCM
        RCM_rain_bareice = (CARRA_cells_for_RCM.xc.isin(cx_rain) 
                                            *  CARRA_cells_for_RCM.yc.isin(cy_rain))\
                                           & (RCM_data.flatten() <= 0.565)\
                                               & (CARRA_cells_for_RCM.maskc == 1)
    
        if np.sum(RCM_rain_bareice) == 0:
             
            nb_ROS_events.append(0)
            continue
        
        else:
            
            specs_rain_bareice = CARRA_cells_for_RCM[RCM_rain_bareice]
            RCM_albedo_rain_bareice = RCM_data.flatten()[RCM_rain_bareice]
        
            nb_ROS_events.append(np.sum(RCM_rain_bareice))
        
        
        
    #     annual_results = pd.DataFrame({'CARRA_lon': lon_rain[av_mask], 
    #                                    'CARRA_lat': lat_rain[av_mask],
    #                                    'RCM_albedo': RCM_rain,
    #                                    'bare_ice': bare_ice_mask})
        
    # results = results.append(annual_results)
        
    

# %% visual verification

from rasterio.plot import show

plt.figure()
ax1=plt.subplot(111)
show(RCM_reader.read(1), transform=RCM_reader.transform, ax=ax1)
ax1.plot(cx_mat.flatten(), cy_mat.flatten(), alpha=0.5)
ax1.plot(CARRA_cells_for_RCM.xc, CARRA_cells_for_RCM.yc, alpha=0.5)
# ax1.scatter(CARRA_cells_for_RCM.iloc[90000].xc, CARRA_cells_for_RCM.iloc[90000].yc, s=30)
# ax1.scatter(CARRA_cells_for_RCM.iloc[90000].xm, CARRA_cells_for_RCM.iloc[90000].ym, s=30)
test = CARRA_cells_for_RCM[(CARRA_cells_for_RCM.xc.isin(cx_rain) 
                                            *  CARRA_cells_for_RCM.yc.isin(cy_rain))\
                                           & (RCM_data.flatten() <= 0.565)\
                                               & (CARRA_cells_for_RCM.maskc == 1)]
                             
ax1.scatter(test.xm, test.ym, s=30)

# %% more visual verification

plt.figure()
plt.imshow(RCM_data)
plt.scatter(RCM_rain.colm, RCM_rain.rowm)

# %% and some more

plt.figure()
plt.imshow(mask==1, origin='lower left')
plt.scatter(specs_rain_bareice.colc, specs_rain_bareice.rowc)
