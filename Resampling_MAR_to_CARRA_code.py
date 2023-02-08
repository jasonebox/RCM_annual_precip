# -*- coding: utf-8 -*-
"""
@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)
Few modifications needed to be run with other datasets
-> modified by Armin Dachauer
"""

import numpy as np
import pandas as pd
import os
from pyproj import Proj, transform
import xarray as xr
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

AW = 0
AD=1

# if AW:
#     base_path = 'C:/Users/Pascal/Desktop/GEUS_2019/SICE_AW_JEB/SICE_AW_JEB/'\
#     + 'CARRA_rain/'
if AD:
    base_path='/home/rmean/Dokumente/Work/GEUS/CARRA_rain/'
    raw_path='/home/rmean/Dokumente/Work/GEUS/CARRA/'
    path_tools='/home/rmean/Dokumente/Work/GEUS/CARRA_tools/'
else:
    base_path = '/Users/jason/Dropbox/CARRA/CARRA_rain/'
    path_tools='/Users/jason/Dropbox/CARRA/CARRA_tools/'
    raw_path='/Users/jason/0_dat/CARRA/output/annual/'

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

# %% reproject 4326 (lat/lon) CARRA coordinates to 3413 (orth polar projection in meters)

#from lat/lon to meters
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

#import CARRA datset
# ds = xr.open_dataset(raw_path+'tp_2012.nc')
# CARRA_data = np.array(ds.tp[0, :, :]).flatten()

# %% load MAR coordinates 

#MAR data
# fn=raw_path+'MARv3.11.5-6km.nc'
fn=raw_path+'withBS/MARv3.13.0-15km-daily-ERA5-1960.nc'
# fn='/Users/jason/0_dat/MARv3.13.0/withBS/MARv3.13.0-15km-daily-ERA5-1960.nc'
# fn=raw_path+'MARv3.11.5-20km.nc'
ds=xr.open_dataset(fn)

#from lat/lon to meters
inProj = Proj(init='epsg:4326')
outProj = Proj(init='epsg:3413')

niM=np.shape(ds.LON.values)[1]
njM=np.shape(ds.LAT.values)[0]

# lat_mesh, lon_mesh = np.meshgrid(ds.LAT.values, ds.LON.values)
x1, y1 = lon360_to_lon180(ds.LON.values.flatten()), ds.LAT.values.flatten()
mx, my = transform(inProj, outProj, x1, y1)
mx_mat = mx.reshape(niM, njM) 
my_mat = my.reshape(niM, njM)

cols2, rows2 = np.meshgrid(np.arange(niM), np.arange(njM))
# lat_m, lon_m = np.meshgrid(ds.LAT.values, ds.LON.values)
MAR_positions = pd.DataFrame({
"row_m": rows2.ravel(),
"col_m": cols2.ravel(),
"lon_m": mx_mat.ravel(),
"lat_m": my_mat.ravel(),})

#load MAR dataset
# MAR_data = np.array(ds.SF2[0, :, :])
MAR_data = np.array(ds.SH[:, :])


# %% nearest neighbours

#dataset to be upscaled -> MAR
nA = np.column_stack((mx_mat.ravel(), my_mat.ravel()) ) 
#dataset to provide the desired grid -> CARRA
nB = np.column_stack((cx_mat.ravel(), cy_mat.ravel()))

btree = cKDTree(nA)  #train dataset
dist, idx = btree.query(nB, k=1)  #apply on grid

#collocate matching cells
CARRA_cells_for_MAR = MAR_positions.iloc[idx]

#%% Output resampling key ERA5 data in CARRA grid

# CARRA_cells_for_MAR.to_pickle(path_tools+'resampling_key_MAR20km_to_CARRA.pkl')
CARRA_cells_for_MAR.to_pickle(path_tools+'resampling_key_MARv3.13.0-15km_BS_to_CARRA.pkl')

#%%  visualisation

new_grid= MAR_data[CARRA_cells_for_MAR.row_m, CARRA_cells_for_MAR.col_m]
new_grid = new_grid.reshape(ni, nj)
plt.imshow(new_grid)
