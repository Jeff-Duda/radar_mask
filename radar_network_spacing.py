import numpy as np
from numpy.random import default_rng
#from matplotlib import use, colormaps
#use('agg')
#import matplotlib.pyplot as plt
from geopy import distance
from geopy.point import Point
from timeit import default_timer
import pandas as pd
from math import cos, sin, tan, pi, degrees, radians, acos, asin, atan2, sqrt, isinf
from sys import exit, path
path.append('/work/noaa/wrfruc/jdduda/radar_mask')
import radar_math_mod as rm

#SETTINGS
out_dir = "/work/noaa/wrfruc/jdduda/radar_mask"

maj_axis = distance.ELLIPSOIDS['WGS-84'][1]
min_axis = distance.ELLIPSOIDS['WGS-84'][0]
Re = sqrt(maj_axis*min_axis)*1000.

def gc1(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    a = cos(lat1) * cos(lat2) * cos(dlon) + sin(lat1)*sin(lat2)
    if a > 0:
       a = np.minimum(a,1.0)
    else:
       a = np.maximum(a,-1.0)
    dist = Re*acos(a)
    return dist

# Grab radar site information
US_radar_file = "/work/noaa/wrfruc/jdduda/radar_mask/nexrad-stations-valid-2026.txt"
df = pd.read_table(US_radar_file,sep="\s+",header=0,skiprows=[1],index_col='ICAO')
radar_sites_dict = df.to_dict(orient='index')
Canadian_radar_file = "/work/noaa/wrfruc/jdduda/radar_mask/Canadian_radar_list.txt"
df = pd.read_table(Canadian_radar_file,sep = "\s+",header=0,skiprows=[1],index_col="STATION_ID")
df.index.name = "ICAO"
can_dict = df.to_dict(orient='index')
radar_sites_dict.update(can_dict)
#remove_list = []
for name in radar_sites_dict.keys():
#   if radar_sites_dict[name]['LAT'] > 53.138378: # This value is 0.5 degree larger than the max latitude in the NR grid
#      remove_list.append(name)
#      continue
#   radar_sites_dict[name]['beam_width'] = 1.0
#   radar_sites_dict[name]['VCP'] = VCP12
#   if name in radars_clear:
#      radar_sites_dict[name]['VCP'] = VCP31
   if name[0] == "K": # Only US radars
      radar_sites_dict[name]['ELEV'] = radar_sites_dict[name]['ELEV']/3.821 # convert from [ft] to [m]
#for i in range(len(remove_list)):
#   del radar_sites_dict[remove_list[i]]
n_radars = len(radar_sites_dict)
back_to_df = pd.DataFrame(radar_sites_dict)
#print(back_to_df.loc['LON'])
#print(back_to_df.loc['LAT'])
radar_lons = back_to_df.loc['LON'].to_numpy(dtype=float)
radar_lats = back_to_df.loc['LAT'].to_numpy(dtype=float)

rad_codes = list(radar_sites_dict.keys())
rad_names = [ radar_sites_dict[rad]['NAME'] for rad in radar_sites_dict.keys() ]
network_distances = np.full((len(radar_lons),len(radar_lons)),np.nan,dtype=float)
radar_pairs = data = [['' for j in range(len(radar_lons))] for i in range(len(radar_lats))]
for a in range(len(radar_lons)):
 for b in range(0,len(radar_lons)):
  radar_pairs[a][b] = f"{rad_codes[a]} / {rad_codes[b]}"
  network_distances[a,b] = gc1(radar_lons[a],radar_lats[a],radar_lons[b],radar_lats[b])

#Flatten and remove same-radar distances (0s)
sort_idx = np.argsort(network_distances.flatten())
fl = open("radar_network_distances.txt",'w')
for i in range(0,len(sort_idx),2):
  if not np.isnan(network_distances.flatten()[sort_idx[i]]):
   ix,iy = np.unravel_index(sort_idx[i],network_distances.shape)
   rad_name_test = radar_pairs[ix][iy].split(' / ')
   if np.isclose(network_distances[ix,iy],0.0,atol=1.0):
    if rad_name_test[0] != rad_name_test[1]:
     print(f"lol wut...{rad_name_test[0]} {rad_name_test[1]}")
   else:
    fl.writelines(f"{network_distances[ix,iy]/1e3:6.1f} km from {rad_codes[ix]:5s}/{rad_names[ix]} to {rad_codes[iy]:5s}/{rad_names[iy]}\n")
#    fl.writelines(f"{i:3d} {sort_idx[i]:6d} {ix:3d} {iy:3d} {network_distances[ix,iy]/1e3:6.1f} km {radar_pairs[ix][iy]}\n")
fl.close()

dist_sort_2d = np.sort(network_distances,axis=1)
radar_nearest_neighbors = dist_sort_2d[:,1]
f1 = open("radar_nearest_neighbor_alphabetical.txt",'w')
f2 = open("radar_nearest_neighbor_ordered.txt",'w')
for i in range(len(radar_lons)):
   f1.writelines(f"{rad_codes[i]:5s} {rad_names[i]:30s}: {radar_nearest_neighbors[i]/1e3:5.1f} km\n")
f1.close()
sort_idx = np.argsort(radar_nearest_neighbors)
for i in range(len(radar_lons)):
   f2.writelines(f"{rad_codes[sort_idx[i]]:.5s} {rad_names[sort_idx[i]]:30s}: {radar_nearest_neighbors[sort_idx[i]]/1e3:5.1f} km\n")
f2.close()

print(f"The average spacing between nearest radars is {np.mean(radar_nearest_neighbors)/1e3:.1f} km")
