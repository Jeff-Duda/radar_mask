import numpy as np
from numpy.linalg import norm as mag
from numpy.random import default_rng
from matplotlib import use, colormaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
use('agg')
import matplotlib.pyplot as plt
from geopy import distance
from geopy.point import Point
from timeit import default_timer
import grib2io
import pandas as pd
from netCDF4 import Dataset
from math import cos, sin, tan, pi, degrees, radians, acos, asin, atan2, sqrt, isinf
from sys import exit
from os.path import isfile, getsize

#SETTINGS
tag = "_MRMS" # for describing different tests (use leading underscore if non-empty)
out_dir = "/work/noaa/wrfruc/jdduda/radar_mask"
radar_dH = 30. # [m]
cmap = colormaps['tab20']
colors = cmap(np.linspace(0,1,21))

maj_axis = distance.ELLIPSOIDS['WGS-84'][1]
min_axis = distance.ELLIPSOIDS['WGS-84'][0]
Re = sqrt(maj_axis*min_axis)*1000.
VCP12 = [0.5,0.9,1.3,1.8,2.4,3.1,4.0,5.1,6.4,8.0,10.0,12.5,15.6,19.5]
VCP215 = [0.5,0.9,1.3,1.8,2.4,3.1,4.0,5.1,6.4,8.0,10.0,12.0,14.0,16.7,19.5]
VCP31 = [0.5,1.5,2.5,3.5,4.5]
VCP100 = [0.25,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.5,13.0,14.5,16.0,18.0,20.0,22.0]
MRMS_heights = [0.50,0.75,1.0,1.25,1.50,1.75,2.0,2.25,2.50,2.75,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,10.,11.,12.,13.,14.,15.,16.,17.,18.,19] # [km]
max_height = 1e3*np.max(MRMS_heights)
max_radar_ranges = {'S':460.0,'C':250.0,'X':120.0} # maximum allowable range of data by band letter
prj = ccrs.PlateCarree()
rng = default_rng()

extent = [-127.5,-65,24,51]
fig_x = 12
AR = (extent[3]-extent[2])/(extent[1]-extent[0])
fig_y = AR*fig_x

def gc2d(lon1, lat1, lon2, lat2):

    # Input arguments must be 2D arrays with the same dimensions.
    # Returns a 2D array of distances.

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    a = np.cos(lat1) * np.cos(lat2) * np.cos(dlon) + np.sin(lat1)*np.sin(lat2)
    dist = Re*np.arccos(a)
    return dist

def gc3d(lon1, lat1, lon2, lat2, z_in, z_flat):

    # lon1, lat1, lon2, lat2 must be 2D arrays with the same dimensions.
    # z_in should be the height to compare to; z_flat should be the height of the flat level (where the 2D distances are considered)
    # Returns a 2D array of distances.

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    a = np.cos(lat1) * np.cos(lat2) * np.cos(dlon) + np.sin(lat1)*np.sin(lat2)
    dist = Re*np.arccos(a)
    dist3d = np.sqrt(dist**2 + (z_in-z_flat)**2)
    return dist3d

def calc_slant_range_2D(H,Z,xr):
   """Compute slant range of a radar beam.
   Required inputs:
   H - radar site height (ASL)
   Z - elevation ASL
   xr - horizontal distance along curved surface (surface of Earth)"""
   # all arrays must be in 2D, including radar height

   a_e = 4./3. * Re
   # Incorporate vertical gradient of refractive index using "four-thirds Earth radius" model (replace "Re" with "a_e")
   s = np.sqrt((a_e+H)**2 + (a_e+Z)**2 - 2*(a_e+H)*(a_e+Z)*np.cos(xr/a_e))
   return s

def beam_height_direct(H,angle,xr):
   # angle must be in radians
   # Distances should be in [m]

   a_e = 4./3. * Re
   term1 = cos(angle)/cos((xr/a_e)+angle)
   z = term1*(a_e+H)-a_e
   return z

def beam_height_2d(H,angle,dist):
   # Input arguments (in this case, dist) must be 2D arrays with the same dimensions.
   # Returns a 2D array of heights

   a_e = 4./3. * Re
   z2d = np.cos(angle)/np.cos((dist/a_e)+angle) * (a_e+H) - a_e
   return z2d

def alpha_to_num(input):
   # Convert alphabetic radar wavelength band name to an integer
   if input == "S":
      output = 1
   elif input == "C":
      output = 2
   elif input == "X":
      output = 3
   else:
      raise Exception(f"band name {input} not recognized. You made a mistake somewhere")
   return output

# Decide on a grid to make the mask on
if False:
 # Obtain nature run grid (from UPP files since there are some minor discrepancies between the grid from the geo_em.d01.nc file and the UPP files)
 grib2_grid_file = "/work/noaa/wrfruc/jdduda/radar_mask/nature_run_one_record.grib2"
 grbf = grib2io.open(grib2_grid_file,'r')
 rec = grbf[0]
 grid_lats,grid_lons = rec.latlons()
 grid_lats_r = np.deg2rad(grid_lats)
 grid_lons_r = np.deg2rad(grid_lons)
 grid_nx = rec.nx
 grid_ny = rec.ny
 grbf.close()
 # Also need the terrain grid
 terrain_file = "/work2/noaa/wrfruc/murdzek/geogrid_no_smooth/single_file/geo_em.d01.nc"
 nc = Dataset(terrain_file,'r')
 terrain_2D = nc.variables['HGT_M'][:][0,:,:]
 nc.close()
else:

#(base) hercules-login-2[87] jduda$ wgrib2 -V MRMS_MergedReflectivityQC_05.50_20201128-011040.grib2
#1:0:vt=2020112801:5500 m above mean sea level:anl:ConusMergedReflectivityQC WSR-88D 3D Reflectivty Mosaic - 33 CAPPIS (500-19000m) [dBZ]:
#    ndata=24500000:undef=0:mean=-525.491:min=-999:max=51
#        lat-lon grid:(7000 x 3500) units 1e-06 input WE:NS output WE:SN res 48
#        lat 54.995000 to 20.005000 by 0.010000
#        lon 230.004999 to 299.994999 by 0.010000 #points=24500000

 MRMS_lat_1 = 20.005
 MRMS_lon_1 = -129.995 # equivalent to 230.005 - 360
 MRMS_lat_2 = 54.995
 MRMS_lon_2 = -60.005 # equivalent to 299.995 - 360
 grid_nx = 7000
 grid_ny = 3500
 lat1d = np.linspace(MRMS_lat_1,MRMS_lat_2,grid_ny)
 lon1d = np.linspace(MRMS_lon_1,MRMS_lon_2,grid_nx)
 grid_lons, grid_lats = np.meshgrid(lon1d,lat1d)
 grid_lats_r = np.deg2rad(grid_lats)
 grid_lons_r = np.deg2rad(grid_lons)
 # Also need the terrain grid
 terrain_file = "/work/noaa/wrfruc/jdduda/NR_terrain_MRMS_grid.nc"
 nc = Dataset(terrain_file,'r')
 terrain_2D = nc.variables['HGT_M'][:][0,:,:]
 nc.close()

grid_lats_3d = np.broadcast_to(grid_lats,(len(MRMS_heights),grid_lats.shape[0],grid_lats.shape[1]))
grid_lons_3d = np.broadcast_to(grid_lons,(len(MRMS_heights),grid_lons.shape[0],grid_lons.shape[1]))

# Grab radar site information
US_radar_file = "/work/noaa/wrfruc/jdduda/radar_mask/nexrad-stations-valid-2026.txt"
df = pd.read_table(US_radar_file,sep="\s+",header=0,skiprows=[1],index_col='ICAO')
radar_sites_dict = df.to_dict(orient='index')
Canadian_radar_file = "/work/noaa/wrfruc/jdduda/radar_mask/Canadian_radar_list.txt"
df = pd.read_table(Canadian_radar_file,sep = "\s+",header=0,skiprows=[1],index_col="STATION_ID")
df.index.name = "ICAO"
can_dict = df.to_dict(orient='index')
radar_sites_dict.update(can_dict)
remove_list = []
for name in radar_sites_dict.keys():
   if radar_sites_dict[name]['LAT'] > 53.138378: # This value is 0.5 degree larger than the max latitude in the NR grid
      remove_list.append(name)
      continue
   radar_sites_dict[name]['beam_width'] = 1.0
   radar_sites_dict[name]['VCP'] = VCP12
   if name[0] == "K": # Only US radars
      radar_sites_dict[name]['ELEV'] = radar_sites_dict[name]['ELEV']/3.821 # convert from [ft] to [m]
for i in range(len(remove_list)):
   del radar_sites_dict[remove_list[i]]
n_radars = len(radar_sites_dict)
print(f"Final analysis will include {n_radars} radars in total. Here they are:")
for k in radar_sites_dict.keys():
   print(k,radar_sites_dict[k]['LAT'],radar_sites_dict[k]['LON'],radar_sites_dict[k]['ELEV'])

# Setup mask arrays
# THIS ITERATION OF THE MASKING LOGIC PRIORITIZES THE BEAM HEIGHT OVER DISTANCE FROM ANY RADAR SITE
# In other words, if there is a grid point that has two radars within {max_radar_distance}, the below
# arrays will take the data from the radar that has the lowest beam at this point, provided it is above ground
# See below note for starting to pull back on this ethos
#min_beam_height_grid = np.full((grid_ny,grid_nx),1e6,dtype=float)
#max_beam_height_grid = np.full((grid_ny,grid_nx),-1e6,dtype=float)
beam_angle_at_height = np.zeros((len(MRMS_heights),grid_ny,grid_nx),dtype=float)
closest_radar_slant_range = np.full((len(MRMS_heights),grid_ny,grid_nx),1e7,dtype=float)
#radar_site_height_grid = np.zeros_like(min_beam_height_grid,dtype=float)
radar_site_height_grid = np.zeros_like(closest_radar_slant_range,dtype=float)
closest_radar_dist_grid = np.full((grid_ny,grid_nx),1e7,dtype=float)
closest_radar_dir_grid = np.zeros((grid_ny,grid_nx),dtype=float)
#closest_radar_id_grid = np.zeros((grid_ny,grid_nx),dtype=int)
distance_3D = np.full((len(MRMS_heights),grid_ny,grid_nx),1e6,dtype=float)
wavelength_band = np.full((grid_ny,grid_nx),-1,dtype=int)
final_radar_mask = np.full((len(MRMS_heights),grid_ny,grid_nx),-1,dtype=int)
n_radars_used = 0
# For allowing for the fact that any given 3D grid point may be "illuminated" by multiple radars and how to handle subsequent processing
MRMS_heights_3D = np.zeros((len(MRMS_heights),grid_ny,grid_nx),dtype=float)
for i,z in enumerate(MRMS_heights):
   MRMS_heights_3D[i,:,:] = np.full((grid_ny,grid_nx),1e3*z)
illuminated_min_dist = np.full((len(MRMS_heights),grid_ny,grid_nx),1e7,dtype=float)
illuminated_radar = np.zeros((len(MRMS_heights),grid_ny,grid_nx),dtype=int)
### End of multi-dimensional array allocation ############

# Determine which radars have already been processed
partial_data_file = f"radar_mask{tag}_{list(radar_sites_dict.keys())[0]}.nc"
start_rn = -1
for rn,rad in enumerate(radar_sites_dict.keys()):
 if rad == "CASHR":
  continue
 old_partial_data_file = partial_data_file
 if not isfile(old_partial_data_file):
    print("Not even the first expected file exists! Oh, well then maybe this is the first time you're running this code. Okay, we\'ll start at the beginning")
    break
 partial_data_file = f"radar_mask{tag}_{rad}.nc"
 if isfile(partial_data_file):
  n_radars_used += 1
  continue
 else:
  print(f"Reading data from {old_partial_data_file}")
  nco = Dataset(old_partial_data_file,'r')
  in_dir = nco.variables['direction_to_closest_radar'][:]
  in_elevation_angle = nco.variables['elevation_angle'][:]
  in_slant_range = nco.variables['slant_range'][:]
  in_mask = nco.variables['mask'][:]
#  in_ID = nco.variables['closest_radar_ID'][:]
  in_band = nco.variables['wavelength_band_num'][:]
  in_illum_rad = nco.variables['illuminated_radar_ID'][:]
  in_illum_dist_min = nco.variables['illuminated_min_distance'][:]
 # in_illum_dist = nco.variables['illuminated_dist_this_radar'][:]
  nco.close()
  start_rn = rn-1
  closest_radar_slant_range = in_slant_range
  closest_radar_dir_grid = in_dir
  beam_angle_at_height = in_elevation_angle
  final_radar_mask = in_mask
 # min_beam_height_grid = in_min_height
 # max_beam_height_grid = in_max_height
#  closest_radar_id_grid = in_ID
  wavelength_band = in_band
  illuminated_min_dist = in_illum_dist_min
  illuminated_dist = in_illum_dist
 # illuminated_radar = in_illum_rad
  break
# End loop
if rn == n_radars-1:
 print(f"All partial radar mask files have been created, so reading from the last of those: {partial_data_file}")
 start_rn = n_radars-1
 nco = Dataset(partial_data_file,'r')
 in_dir = nco.variables['direction_to_closest_radar'][:]
 in_elevation_angle = nco.variables['elevation_angle'][:]
 in_slant_range = nco.variables['slant_range'][:]
 in_mask = nco.variables['mask'][:]
# in_ID = nco.variables['closest_radar_ID'][:]
 in_band = ncf.variables['wavelength_band_num'][:]
 in_illum_rad = nco.variables['illuminated_radar_ID'][:]
 in_illum_dist = nco.variables['illuminated_min_distance'][:]
 nco.close()
 closest_radar_slant_range = in_slant_range
# closest_radar_dir_grid = in_dir
 beam_angle_at_height = in_elevation_angle
 final_radar_mask = in_mask
# min_beam_height_grid = in_min_height
# max_beam_height_grid = in_max_height
 closest_radar_id_grid = in_ID
 wavelength_band = in_band
 illuminated_min_dist = in_illum_dist
 illuminated_radar = in_illum_rad

# Process all remaining radars that have not yet been processed
used_radars = []
elapsed_time = 0.
rc = 0
for rn,rad in enumerate(radar_sites_dict.keys()):

 time0 = default_timer()
 used_radars.append(rad)
 if start_rn == n_radars-1:
  print("All data computed...skipping big loop")
  break
 if rn <= start_rn:
  print(f"Data from radar {rad} has already been computed. skipping loop and moving to next radar")
  continue

 rc += 1
 print("******************************************************")
 print(f"*** Working on radar site {rad} ({rn+1:3d} of {n_radars} radars) *** (location: {radar_sites_dict[rad]['NAME']})")
 print("******************************************************")
 rdr_lon = radar_sites_dict[rad]['LON']
 rdr_lat = radar_sites_dict[rad]['LAT']
 rdr_lon_r = radians(rdr_lon)
 rdr_lat_r = radians(rdr_lat)
 cos_rdr_lat = cos(rdr_lat_r)
 cos_rdr_lon = cos(rdr_lon_r)
 sin_rdr_lat = sin(rdr_lat_r)
 sin_rdr_lon = sin(rdr_lon_r)
 # NEXRAD radar: The overall tower height can vary from 5 to 30 meters in 5 meter increments.
 # In many cases the stated height of the radar site from the lookup file actually falls below the terrain. Therefore, it will have to be assumed
 VCP_angles = radar_sites_dict[rad]['VCP']
 beam_width = radar_sites_dict[rad]['beam_width']
 xt = radians(np.max(VCP_angles))
 nt = radians(np.min(VCP_angles))
 max_def_dist = max_radar_ranges[radar_sites_dict[rad]['BAND']]*1e3 # convert from [km] in the dictionary to [m]
 illuminated_dist = np.full((len(MRMS_heights),grid_ny,grid_nx),1e7,dtype=float)

 # Obtain NR-gridpoint closest to radar site
 timea = default_timer()
 dumln = np.full_like(grid_lons,rdr_lon)
 dumlt = np.full_like(grid_lons,rdr_lat)
 dumln3d = np.broadcast_to(dumln,(len(MRMS_heights),grid_lats.shape[0],grid_lats.shape[1]))
 dumlt3d = np.broadcast_to(dumlt,(len(MRMS_heights),grid_lats.shape[0],grid_lats.shape[1]))
 distances = gc2d(grid_lons,grid_lats,dumln,dumlt)
 radar_j,radar_i = np.unravel_index(np.argmin(distances),grid_lons.shape)
# print(grid_lats[radar_j,radar_i])
# print(grid_lons[radar_j,radar_i])
# print(rdr_lon,rdr_lat)
 H = terrain_2D[radar_j,radar_i] + radar_dH # Radar site height ASL in the NR grid
 print( (f" The nominal height of this radar is {radar_sites_dict[rad]['ELEV']:.0f} m ASL, which puts it at {radar_sites_dict[rad]['ELEV']-terrain_2D[radar_j,radar_i]:.0f} m above ground."
         f" Regardless, the height in this code has been reset to {H:.0f} m ASL, which is set as {radar_dH:.0f} m above the terrain ({terrain_2D[radar_j,radar_i]:.0f} m) at the location of the radar"))
# time_ = default_timer()
# SHOULD NOT NEED THIS CODE ANYMORE, BUT KEEP AROUND JUST IN CASE
# for h in range(len(MRMS_heights)):
#    distance_at_height = gc3d(grid_lons,grid_lats,dumln,dumlt,H,1e3*MRMS_heights[h])
#    # Either this needs to change
#    distance_3D[h,:,:] = np.minimum(distance_3D[h,:,:],distance_at_height)
#    # change lin eabove?
# time__ = default_timer()
# print(f"Time to calculate 3D distances for the whole grid was {time__-time_:.2f} s")
 sorted_distances = np.sort(distances.flatten())
 number_within = len(sorted_distances[sorted_distances <= max_def_dist])
 print(f" This radar is {radar_sites_dict[rad]['BAND']}-band, so the max distance is set to {max_def_dist/1e3:.0f} km. There are {number_within} ({100*number_within/(grid_nx*grid_ny):.3f} % of the total domain) points within this range of the radar site.")
 if number_within < 10:
   print(" There are fewer than 10 domain points within range of this radar site. We will dis-include this radar site.")
   used_radars.remove(rad)
   continue
 n_radars_used += 1 # This line should stay immediately after the if statement in the line above
 sorted_idxs = np.unravel_index(np.argsort(distances.flatten()),distances.shape)
 this_radar_mask = distances <= max_def_dist
 ij_within_range = np.argwhere(this_radar_mask)
 rows = ij_within_range[:,0]
 columns = ij_within_range[:,1]
 radar_site_height_grid[:,:,:] = H
 this_radar_min_height = np.full((grid_ny,grid_nx),1e9,dtype=float)
 this_radar_max_height = np.full((grid_ny,grid_nx),-1e9,dtype=float)
 timeb = default_timer()
 #print(f" Time to sort the entire distance array: {timeb-timea:.3f} s")

 timea = default_timer()
 vector_np = np.array([0,0,Re])
 vector_rad = np.zeros((grid_nx*grid_ny,3),dtype=float)
 vector_rad[:,:] = [cos_rdr_lat*cos_rdr_lon,cos_rdr_lat*sin_rdr_lon,sin_rdr_lat]
 norm_gc = np.cross(vector_rad,vector_np)
 ax1 = Re*np.cos(grid_lats_r.flatten()[:,np.newaxis])*np.cos(grid_lons_r.flatten()[:,np.newaxis])
 ax2 = Re*np.cos(grid_lats_r.flatten()[:,np.newaxis])*np.sin(grid_lons_r.flatten()[:,np.newaxis])
 ax3 = Re*np.sin(grid_lats_r.flatten()[:,np.newaxis])
 vector_grid = np.hstack((ax1,ax2,ax3))
 norm_vectors = np.cross(vector_rad,vector_grid)
 dot_products = np.vecdot(norm_vectors,norm_gc,axis=1)
 norm_vectors_len = mag(norm_vectors,axis=1)
 angles = np.arccos(np.vecdot(norm_gc,norm_vectors,axis=1)/(mag(norm_gc,axis=1)*norm_vectors_len))
 angles2d = angles.reshape(grid_lats_r.shape)
 bearing = np.where(grid_lons > rdr_lon,angles2d,2*pi-angles2d)
 timeb = default_timer()
# print(f" It took {timeb-timea:.1f} s to compute bearings")

 # Now calculate the main array values and account for beam blockages
 blocked_points = np.empty((0,2),dtype=int)
 rbw = radians(beam_width)
 for az in np.arange(0,2*pi,rbw):
   if az >= 0.5*rbw and az <= 2*pi-0.5*rbw:
      idx_j,idx_i = np.where((np.abs(bearing-az) < 0.5*rbw) & (distances < max_def_dist))
   else:
      idx_j,idx_i = np.where(((bearing < 0.5*rbw) | (bearing > 2*pi-0.5*rbw)) & (distances < max_def_dist))
      # the where command returns arrays that are almost correctly sorted by distance. But it's not 100% accurate, and that is important.
      # So I'll perform a sort myself
   blocked = np.full((len(VCP_angles),len(idx_j)),False,dtype=bool)
   beam_distances = distances[idx_j,idx_i]
 #  print(f"The fraction of grid points in this beam (az = {degrees(az):.2f} deg.) and distance is {len(idx_j)}, ({100*len(idx_j)/number_within:.3f} %)")
   # Sort
   sorted_idxs = np.argsort(beam_distances)
   sorted_j,sorted_i = idx_j[sorted_idxs],idx_i[sorted_idxs]
   sorted_distances = distances[sorted_j,sorted_i]
   for n in range(len(sorted_idxs)):
    j = sorted_j[n]
    i = sorted_i[n]
  #  print(f"Working on point no. {n:04d}, i,j = {i},{j}, distance {beam_distances[n]:.1f} m, bearing {degrees(bearing[j,i]):.3f} deg")
    for v in range(len(VCP_angles)):
      vr = radians(VCP_angles[v])
      beam_height = beam_height_direct(H,vr,sorted_distances[n])
      if beam_height <= terrain_2D[j,i]:
         # Beam is blocked at this range and azimuth...as are all points further out along this radial
    #     print(f"{len(sorted_idxs)}, {n:04d}, beam: {beam_height:.1f} m, (ground: {terrain_2D[j,i]:.1f} m), range: {sorted_distances[n]:.1f} m, {len(blocked[v,n:])}, {len(blocked[v,:n])}")
         blocked[v,n:] = True
         # Try next elevation angle
         continue
      else:
         if not np.any(blocked[v,:n]):
          if beam_height < this_radar_min_height[j,i]:
            this_radar_min_height[j,i] = beam_height
         #   closest_radar_dir_grid[j,i] = bearing[j,i]
         #   closest_radar_id_grid[j,i] = rn+1
            wavelength_band[j,i] = alpha_to_num(radar_sites_dict[rad]['BAND'])
            break
         else:
          continue
    if blocked[0,n]:
     if blocked_points.shape[0] == 0:
        blocked_points = np.array([j,i])
     else:
        blocked_points = np.vstack((blocked_points,[j,i]))
    # Repeat above for max height of radar beam
    radar_z_max = beam_height_direct(H,xt,sorted_distances[n])
    if radar_z_max > this_radar_max_height[j,i]:
       this_radar_max_height[j,i] = radar_z_max
 n_blocked_points = len(np.unique(blocked_points,axis=0))
 if n_blocked_points > 0:
    print(f" {n_blocked_points} grid points ({100*n_blocked_points/float(number_within):.2f} % of all points within {max_def_dist/1e3:.0f} km of the radar site) were blocked at the lowest scan angle, but saying nothing about what happened at higher angles")

 timea = default_timer()
 illuminated_mask = (MRMS_heights_3D > this_radar_min_height) & (MRMS_heights_3D < this_radar_max_height)
 mask_idx = np.argwhere(illuminated_mask)
 mask_z = mask_idx[:,0]
 mask_y = mask_idx[:,1]
 mask_x = mask_idx[:,2]
 time__ = default_timer()
 # calculate the 3D distance to this radar site ONLY for points illuminated by THIS radar
 illuminated_dist[mask_z,mask_y,mask_x] = gc3d(grid_lons[mask_y,mask_x],grid_lats[mask_y,mask_x],dumln[mask_y,mask_x],dumlt[mask_y,mask_x],radar_site_height_grid[mask_z,mask_y,mask_x],MRMS_heights_3D[mask_z,mask_y,mask_x])
 time13 = default_timer()
 print(f"Calculating 3D distances just for this radar took {time13-time__:.3f} s")
 closer_mask = illuminated_dist < illuminated_min_dist
 print(f"This radar illuminated this many 3D points: {np.count_nonzero(closer_mask)} ({100*np.count_nonzero(closer_mask)/(grid_nx*grid_ny*len(MRMS_heights)):.4f}% of the total 3D grid size)")
 illuminated_min_dist = np.where(closer_mask,illuminated_dist,illuminated_min_dist)
 closer_idx = np.argwhere(closer_mask)
 closest_radar_dir_grid[closer_idx[:,1],closer_idx[:,2]] = bearing[closer_idx[:,1],closer_idx[:,2]]
 illuminated_radar[closer_mask] = rn+1
 final_radar_mask[mask_z,mask_y,mask_x] = 1
 # The point is to assign whatever slant range value occurs in (closer_mask) to "closest_radar_slant_range"
 timea = default_timer()
 for i,z in enumerate(MRMS_heights):
    height_2d = np.full_like(rdr_lat,1e3*z,dtype=float)
#    slant_range2d = calc_slant_range_2D(radar_site_height_grid,height_2d,closest_radar_dist_grid)
    slant_range2d = calc_slant_range_2D(radar_site_height_grid[i,:,:],height_2d,illuminated_min_dist[i,:,:])
    closest_radar_slant_range[i,:,:] = np.minimum(closest_radar_slant_range[i,:,:],slant_range2d)
 beam_angle_at_height = np.maximum(beam_angle_at_height,np.rad2deg(np.arctan((MRMS_heights_3D-H)/illuminated_min_dist)))
 timeb = default_timer()
 print(f"Calculating slant range and beam angle took {timeb-timea:.2f} s")

 timea = default_timer()
 ncf = Dataset(f"{out_dir}/radar_mask{tag}_{rad}.nc",'w',format='NETCDF4')
 dimy = ncf.createDimension('latitude',grid_ny)
 dimx = ncf.createDimension('longitude',grid_nx)
 dimz = ncf.createDimension('height',len(MRMS_heights))
 varz = ncf.createVariable('MRMS_heights','i2',dimensions=(dimz))
 vardir = ncf.createVariable('direction_to_closest_radar','f',dimensions=(dimy,dimx))
# varid = ncf.createVariable('closest_radar_ID','i2',dimensions=(dimy,dimx))
 varmask = ncf.createVariable('mask','i2',dimensions=(dimz,dimy,dimx))
# varmin = ncf.createVariable('min_beam_height','f',dimensions=(dimy,dimx))
# varmax = ncf.createVariable('max_beam_height','f',dimensions=(dimy,dimx))
 varangle = ncf.createVariable('elevation_angle','f',dimensions=(dimz,dimy,dimx))
 varslant = ncf.createVariable('slant_range','f',dimensions=(dimz,dimy,dimx))
 varband = ncf.createVariable('wavelength_band_num','i2',dimensions=(dimy,dimx))
 varid = ncf.createVariable('illuminated_radar_ID','i2',dimensions=(dimz,dimy,dimx))
 var2 = ncf.createVariable('illuminated_min_distance','f',dimensions=(dimz,dimy,dimx))
 var3 = ncf.createVariable('illuminated_dist_this_radar','f',dimensions=(dimz,dimy,dimx))
 varz[:] = MRMS_heights
 varmask[:] = final_radar_mask
# varmin[:] = min_beam_height_grid
# varmax[:] = max_beam_height_grid
 varangle[:] = beam_angle_at_height
 varslant[:] = closest_radar_slant_range
 vardir[:] = closest_radar_dir_grid
# varid[:] = closest_radar_id_grid
 varband[:] = wavelength_band
 varid[:] = illuminated_radar
 var2[:] = illuminated_min_dist
 var3[:] = illuminated_dist
 ncf.single_radar_site_name = rad
 ncf.values_key = "1 - within mask; 0 - outside of mask (but point was checked); -1 - gridpoint not checked, but assumed outside mask"
 ncf.grid_projection = "MRMS grid (0.01 x 0.01 deg.)"
 ncf.close()
 timeb = default_timer()
 print(f"Time to write netcdf file: {timeb-timea:.2f} s")

 time00 = default_timer()
 elapsed_time += (time00-time0)
 print(f" Processing this radar took {time00-time0:.3f} s. Averaging {elapsed_time/rc:.1f} s per radar site")
# End radar loop

ncf = Dataset(f"{out_dir}/radar_mask{tag}.nc",'w',format='NETCDF4')
dimy = ncf.createDimension('latitude',grid_ny)
dimx = ncf.createDimension('longitude',grid_nx)
dimz = ncf.createDimension('height',len(MRMS_heights))
varz = ncf.createVariable('MRMS_heights','f4',dimensions=(dimz))
#varid = ncf.createVariable('closest_radar_ID','i2',dimensions=(dimy,dimx))
#vardist = ncf.createVariable('distance_to_closest_radar_3D','f',dimensions=(dimz,dimy,dimx))
vardir = ncf.createVariable('azimuth_to_closest_radar','f',dimensions=(dimy,dimx))
varmask = ncf.createVariable('mask','i2',dimensions=(dimz,dimy,dimx))
varangle = ncf.createVariable('elevation_angle','f',dimensions=(dimz,dimy,dimx))
varslant = ncf.createVariable('slant_range','f',dimensions=(dimz,dimy,dimx))
varband = ncf.createVariable('wavelength_band_num','i2',dimensions=(dimy,dimx))
var1 = ncf.createVariable('illuminated_radar_ID','i2',dimensions=(dimz,dimy,dimx))
var2 = ncf.createVariable('illuminated_min_distance','f',dimensions=(dimz,dimy,dimx))
varz[:] = MRMS_heights
varmask[:] = final_radar_mask
#varid[:] = closest_radar_id_grid
vardir[:] = closest_radar_dir_grid
#vardist[:] = distance_3D
varangle[:] = beam_angle_at_height
varslant[:] = closest_radar_slant_range
var1[:] = illuminated_radar
var2[:] = illuminated_min_dist
varband[:] = wavelength_band
varband.note = "1 = S-band; 2 = C-band; 3 = X-band"
ncf.radars_used = used_radars
ncf.radar_count = n_radars_used
ncf.radar_IDs = "closest_radar_ID is an integer corresponding to the list 'radars_used'"
ncf.values_key = "1 - within mask; 0 - outside of mask (but point was checked); -1 - gridpoint not checked, but assumed outside mask"
ncf.grid_projection = "MRMS grid (0.01 x 0.01 deg.)"
ncf.close()
print("3D radar mask created")

if True:
 height_levs = np.concatenate((np.arange(1000.,1000.,100.),np.arange(1000.,10000,500.),np.arange(10000,20000.1,1000.)))
 distance_levs = np.arange(50.,750.,50.)
 angle_levs = np.arange(0,360.1,15.)
# plt.figure(figsize=(fig_x,fig_y))
# ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=prj)
# ax.set_extent(extent,crs=prj)
# ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=1)
# cf = ax.contourf(grid_lons,grid_lats,min_beam_height_grid,levels = height_levs,cmap=colormaps['CMRmap_r'],norm='linear',vmin=0,vmax=20000.,extend='both',transform=prj)
# cb = plt.colorbar(mappable=cf,orientation='horizontal',fraction=0.1,pad=0.01,shrink=0.8,aspect=50)
# cb.ax.tick_params(labelsize=6)
# cb.set_label("Minimum height of radar coverage [m ASL]",fontsize=8)
# plt.savefig(f"{out_dir}/min_beam_height{tag}.png",dpi=150)
# plt.close()

# plt.figure(figsize=(fig_x,fig_y))
# ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=prj)
# ax.set_extent(extent,crs=prj)
# ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=1)
# cf = ax.contourf(grid_lons,grid_lats,max_beam_height_grid,levels = np.arange(2500.,25000.1,2500.),cmap=colormaps['CMRmap_r'],norm='linear',vmin=0,vmax=25000.,extend='both',transform=prj)
# cb = plt.colorbar(mappable=cf,orientation='horizontal',fraction=0.1,pad=0.01,shrink=0.8,aspect=50)
# cb.ax.tick_params(labelsize=6)
# cb.set_label("Maximum height of radar coverage [m ASL]",fontsize=8)
# plt.savefig(f"{out_dir}/max_beam_height{tag}.png",dpi=150)
# plt.close()

 for i,z in enumerate(MRMS_heights):
  plt.figure(figsize=(fig_x,fig_y))
  ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=prj)
  ax.set_extent(extent,crs=prj)
  ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=1)
  cf = ax.contourf(grid_lons,grid_lats,closest_radar_slant_range[i,:,:]/1e3,levels = distance_levs,cmap=colormaps['CMRmap_r'],norm='linear',vmin=0.,vmax=550.,extend='both',transform=prj)
  cb = plt.colorbar(mappable=cf,orientation='horizontal',fraction=0.1,pad=0.01,shrink=0.8,aspect=50)
  cb.ax.tick_params(labelsize=6)
  cb.set_label("Slant range to closest radar [m ASL]",fontsize=8)
  plt.savefig(f"{out_dir}/nearest_radar_slant_range_{z}km{tag}.png",dpi=150)
  plt.close()

  plt.figure(figsize=(fig_x,fig_y))
  ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=prj)
  ax.set_extent(extent,crs=prj)
  ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=1)
  cf = ax.contourf(grid_lons,grid_lats,illuminated_min_dist[i,:,:]/1e3,levels = distance_levs,cmap=colormaps['CMRmap_r'],norm='linear',vmin=0.,vmax=500.,extend='both',transform=prj)
  cb = plt.colorbar(mappable=cf,orientation='horizontal',fraction=0.1,pad=0.01,shrink=0.8,aspect=50)
  cb.ax.tick_params(labelsize=6)
  cb.set_label("Distance to radar providing minimum height coverage [km]",fontsize=8)
  plt.savefig(f"{out_dir}/nearest_radar_dist_{z}km{tag}.png",dpi=150)
  plt.close()

 plt.figure(figsize=(fig_x,fig_y))
 ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=prj)
 ax.set_extent(extent,crs=prj)
 ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=1)
 cf = ax.contourf(grid_lons,grid_lats,np.rad2deg(closest_radar_dir_grid),levels = angle_levs,cmap = colormaps['twilight'],norm = 'linear',vmin=0.,vmax=360.,transform=prj)
 cb = plt.colorbar(mappable=cf,orientation='horizontal',fraction=0.1,pad=0.01,shrink=0.8,aspect=50)
 cb.ax.tick_params(labelsize=6)
 cb.set_label(r"Angle to radar providing minimum height coveage [$\degree$]",fontsize=8)
 plt.savefig(f"{out_dir}/nearest_radar_angle{tag}.png",dpi=150)
 plt.close()
print("script done")
