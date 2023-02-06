# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)

Few modifications needed to be run with other datasets

"""

# import rasterio
import numpy as np
import pandas as pd
import os
from pyproj import Proj, transform
import xarray as xr
from datetime import datetime, timedelta
import glob
from scipy.spatial import cKDTree
import random
import matplotlib.pyplot as plt
import gzip

# %% paths

AW = 0
AD=1
base_path = '/Users/jason/Dropbox/CARRA/CARRA_rain/'

if AW:
    base_path = 'C:/Users/Pascal/Desktop/GEUS_2019/SICE_AW_JEB/SICE_AW_JEB/'\
    + 'CARRA_rain/'
    raw_path='H:/CARRA/'
if AD:
    base_path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_rain/'
    raw_path='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA/'
    path_tools='C:/Users/Armin/Documents/Work/GEUS/Github/CARRA_tools/'

os.chdir(base_path)

# AW: initially to 'tp' but I don't have this file with me atm
var = 'tp'

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


ds = xr.open_dataset(raw_path+ var + '_2012.nc') 
CARRA_data = np.array(ds[var][0, :, :]).flatten()

# %% load MODIS coordinates (just replace by ERA5 and load the same way than CARRA)

# MODIS_reader = rasterio.open('H:/MOD10A1_albedo/2011/2011_071.tif')
# MODIS_data = MODIS_reader.read(1)

# MODIS_reader.xy(np.shape(MODIS_data)[0], np.shape(MODIS_data)[1])

# cols2, rows2 = np.meshgrid(np.arange(np.shape(MODIS_data)[1]), 
#                            np.arange(np.shape(MODIS_data)[0]))

# x_m, y_m = MODIS_reader.xy(rows2.flatten(), cols2.flatten())


#RACMO data
fn=raw_path+'snowfall.1958-2020.BN_RACMO2.3p2_ERA5_3h_FGRN055.1km.YY.nc.gz'
with gzip.open(fn) as gz:
    ds=xr.open_dataset(gz,decode_times=False)
RACMO_data = np.array(ds.snowfallcorr[0, :, :])
cols2, rows2 = np.meshgrid(np.arange(np.shape(RACMO_data)[1]), 
                           np.arange(np.shape(RACMO_data)[0]))
lon_r, lat_r = np.meshgrid(ds.x.values, ds.y.values)
x_r=lon_r.flatten()
y_r=lat_r.flatten()

RACMO_positions = pd.DataFrame({'row_r': rows2.ravel(),
                                'col_r': cols2.ravel(),
                                'xr': x_r,
                                'yr': y_r})

# %% get the resolution of the other dataset with a pseudo-random sampling of distances

dataset = np.vstack((x_r, y_r)).T
 
tree = cKDTree(dataset)

distances = []

for i in range(1000):
    
    rand_index = random.randint(0, len(x_r))
    
    rand_point = [x_r[rand_index], y_r[rand_index]]
    
    # 2 because 1 is the point itself 
    dist, ind = tree.query(rand_point, k=2)
    
    distances.append(dist[1])
    
    print(i)

# As we discussed on Friday, it's a binary result, either std is 0 or not
print('The estimated resolution of your dataset is %.3f +- %.3f' % (np.nanmean(distances),
                                                                    np.nanstd(distances)))

# %% nearest neighbours

# not only 2500m because 45° angle
dist_threshold = np.sqrt(2) * 2500

# ratio between the two resolutions just to get an idea about the number of neighbors
# to ask for. Two times this number to stay on the safe side, and 4 neighbors
opt_neigh_nb = 4 * (2 * int(2500 / 1000))

nA = np.column_stack((cx_mat.ravel(), cy_mat.ravel()))
    
nB = np.column_stack((x_r, y_r))

# btree = cKDTree(nA)
# dist, idx = btree.query(nB, k=opt_neigh_nb)

# resampling RACMO is independent of the other dataset (find k neighbours
# in the same data set)
btree_resampling = cKDTree(nB)
dist_resampling, idx_resampling = btree_resampling.query(nB, k=opt_neigh_nb)

arr_idx_resampling = np.array(idx_resampling)
arr_dist_resampling = np.array(dist_resampling)


# compute the mean value in the kernel of size dist_threshold -> new value
# for each RACMO pixel, which is a mean within the kernel (independent of the second dataset).
raw_values = RACMO_data.flatten()[arr_idx_resampling]
raw_values[arr_dist_resampling > dist_threshold] = np.nan
upsampled_values = np.nanmean(raw_values, axis=1)

#run the normal direct matching and replace the RACMO value of the direct matching with the upsampled one
btree = cKDTree(nB)
dist, idx = btree.query(nA)

RACMO_cells_for_CARRA = RACMO_positions.iloc[idx]
RACMO_cells_for_CARRA['upsampled_value'] = upsampled_values[idx]
RACMO_cells_for_CARRA['direct_match_value'] = RACMO_data[RACMO_cells_for_CARRA.row_r, 
                                                         RACMO_cells_for_CARRA.col_r]


# %% check deviations between the two methods

plt.figure()
plt.plot(RACMO_cells_for_CARRA.direct_match_value)
plt.plot(RACMO_cells_for_CARRA.upsampled_value)

mean_dev = np.nanmean(RACMO_cells_for_CARRA.direct_match_value 
                      - RACMO_cells_for_CARRA.upsampled_value)

plt.figure()
plt.plot(RACMO_cells_for_CARRA.direct_match_value 
                      - RACMO_cells_for_CARRA.upsampled_value)



#%% 
CARRA_cells_for_RACMO.to_pickle(path_tools+'resampling_key_RACMO1km_to_CARRA.pkl')