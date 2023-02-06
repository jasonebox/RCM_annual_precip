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
import gzip
import netCDF4

AW = 0
AD=1
base_path = '/Users/jason/Dropbox/CARRA/CARRA_rain/'

if AW:
    base_path = 'C:/Users/Pascal/Desktop/GEUS_2019/SICE_AW_JEB/SICE_AW_JEB/'\
    + 'CARRA_rain/'
if AD:
    base_path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_rain/'
    raw_path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA/'
    path_tools='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_tools/'

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
ds = xr.open_dataset(raw_path+'tp_2012.nc')
CARRA_data = np.array(ds.tp[0, :, :]).flatten()

# %% load RACMO coordinates 

#RACMO data
fn=raw_path+'snowfall.1958-2020.BN_RACMO2.3p2_ERA5_3h_FGRN055.1km.YY.nc.gz'
with gzip.open(fn) as gz:
    ds=xr.open_dataset(gz,decode_times=False)

lon=ds.x.values
lat=ds.y.values

# #from lat/lon to meters
# inProj = Proj(init='epsg:4326')
# outProj = Proj(init='epsg:3413')

niR=np.shape(ds.x.values)[0]
njR=np.shape(ds.y.values)[0]

rx_mat, ry_mat = np.meshgrid(ds.x.values, ds.y.values)
# x1, y1 = lon360_to_lon180(ds.x.values.flatten()), ds.y.values.flatten()
# rx, ry = transform(inProj, outProj, lon_mesh, lat_mesh)
# rx_mat = rx.reshape(niR, njR) 
# ry_mat = ry.reshape(niR, njR)

cols2, rows2 = np.meshgrid(np.arange(niR), np.arange(njR))
# lat_m, lon_m = np.meshgrid(ds.LAT.values, ds.LON.values)
RACMO_positions = pd.DataFrame({
"row_r": rows2.ravel(),
"col_r": cols2.ravel(),
"lon_r": rx_mat.ravel(),
"lat_r": ry_mat.ravel(),})

#load RACMO dataset
RACMO_data = np.array(ds.snowfallcorr[0, :, :])


# %% nearest neighbours

#dataset to be upscaled -> RACMO
nA = np.column_stack((rx_mat.ravel(), ry_mat.ravel()) ) 
#dataset to provide the desired grid -> CARRA
nB = np.column_stack((cx_mat.ravel(), cy_mat.ravel()))

btree = cKDTree(nA)  #train dataset
dist, idx = btree.query(nB, k=1)  #apply on grid

#collocate matching cells
CARRA_cells_for_RACMO = RACMO_positions.iloc[idx]

#%% Output resampling key ERA5 data in CARRA grid

CARRA_cells_for_RACMO.to_pickle(path_tools+'resampling_key_RACMO_to_CARRA.pkl')

#%%  visualisation

new_grid= RACMO_data[CARRA_cells_for_RACMO.row_r, CARRA_cells_for_RACMO.col_r]
new_grid = new_grid.reshape(ni, nj)
plt.imshow(new_grid)
