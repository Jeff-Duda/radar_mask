# Copied from create_radar_mask.py on 16 July 2026 at 0334 UTC
# The change here is to modify the setting of radar mask from
# -Selecting all space between the lowest and highest beam heights
# to
# -Creating a 3D mask matrix and setting points within that matrix to "ON" if they are illuminated by a beam in the loop starting on line 249
#  By default, all points in the 3d mask matrix are initialized to "OFF"
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
#from scipy.spatial import cKDTree
from sys import exit

#SETTINGS
tag = "_GJX_clear" # for describing different tests
out_dir = "/work/noaa/wrfruc/jdduda/radar_mask"
radar_dH = 30. # [m]
cmap = colormaps['tab20']
colors = cmap(np.linspace(0,1,21))
radars_clear = ['KFTG','KPUX','KCYS','KGJX','KMTX','KICX','KABX']

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
   Z - some total height level (generally terrain_height + height_above_ground
   xr - horizontal distance along curved surface (surface of Earth)"""
   # all arrays must be in 2D, including radar height

   s = np.sqrt((Re+H)**2 + (Re+Z)**2 - 2*(Re+H)*(Re+Z)*np.cos(xr/Re))
   return s

def beam_height_direct(H,angle,xr):
   # angle must be in radians
   # Distances should be in [m]

   term1 = cos(angle)/cos((xr/Re)+angle)
   z = term1*(Re+H)-Re
   return z

def beam_height_2d(H,angle,dist):
   # Input arguments (in this case, dist) must be 2D arrays with the same dimensions.
   # Returns a 2D array of heights

   z2d = np.cos(angle)/np.cos((dist/Re)+angle) * (Re+H) - Re
   return z2d

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
 MRMS_lat_1 = 20.005
 MRMS_lon_1 = -129.995
 MRMS_lat_2 = 54.995
 MRMS_lon_2 = -60.995
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

# Grab radar site information
US_radar_file = "/work/noaa/wrfruc/jdduda/radar_mask/nexrad-stations.txt"
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
   if name in radars_clear:
      radar_sites_dict[name]['VCP'] = VCP31
   if name[0] == "K": # Only US radars
      radar_sites_dict[name]['ELEV'] = radar_sites_dict[name]['ELEV']/3.821 # convert from [ft] to [m]
for i in range(len(remove_list)):
   del radar_sites_dict[remove_list[i]]
n_radars = len(radar_sites_dict)
print(f"Final analysis will include {n_radars} radars in total. Here they are:")
for k in radar_sites_dict.keys():
   print(k,radar_sites_dict[k]['LAT'],radar_sites_dict[k]['LON'],radar_sites_dict[k]['ELEV'])
#print(radar_sites_dict.keys())
#print(radar_sites_dict.values())

# Setup mask arrays
# THIS ITERATION OF THE MASKING LOGIC PRIORITIZES THE BEAM HEIGHT OVER DISTANCE FROM ANY RADAR SITE
# In other words, if there is a grid point that has two radars within {max_radar_distance}, the below
# arrays will take the data from the radar that has the lowest beam at this point, provided it is above ground
min_beam_height_grid = np.full((grid_ny,grid_nx),1e6,dtype=float)
max_beam_height_grid = np.full((grid_ny,grid_nx),-1e6,dtype=float)
#closest_radar_slant_range = np.zeros((grid_ny,grid_nx),dtype=float)
beam_angle_at_height = np.zeros((len(MRMS_heights),grid_ny,grid_nx),dtype=float)
closest_radar_slant_range = np.full((len(MRMS_heights),grid_ny,grid_nx),1e6,dtype=float)
radar_site_height_grid = np.zeros_like(min_beam_height_grid,dtype=float)
closest_radar_dist_grid = np.zeros((grid_ny,grid_nx),dtype=float)
closest_radar_dir_grid = np.zeros((grid_ny,grid_nx),dtype=float)
closest_radar_id_grid = np.zeros((grid_ny,grid_nx),dtype=int)
distance_3D = np.full((len(MRMS_heights),grid_ny,grid_nx),1e6,dtype=float)
final_radar_mask = np.full((len(MRMS_heights),grid_ny,grid_nx),-1,dtype=int)

used_radars = []
for i,rad in enumerate(radar_sites_dict.keys()):

 time0 = default_timer()
 used_radars.append(rad)
# if i > 5:
#  break

 print("******************************************************")
 print(f"*** Working on radar site {rad} ({i+1:3d} of {n_radars} radars) *** (location: {radar_sites_dict[rad]['NAME']})")
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
# H = radar_sites_dict[rad]['ELEV']
 VCP_angles = radar_sites_dict[rad]['VCP']
 beam_width = radar_sites_dict[rad]['beam_width']
 xt = radians(np.max(VCP_angles))
 nt = radians(np.min(VCP_angles))
 max_def_dist = max_radar_ranges[radar_sites_dict[rad]['BAND']]*1e3 # convert from [km] in the dictionary to [m]

 this_radar_beam_heights = np.zeros((len(VCP_angles),grid_ny,grid_nx),dtype=float)

 # Obtain NR-gridpoint closest to radar site
 if False:
  lat1d = grid_lats.flatten()
  lon1d = grid_lons.flatten()
  latlon = np.vstack((lon1d,lat1d)).T
  timea = default_timer()
  tree = cKDTree(latlon)
  d,i=tree.query([rdr_lon,rdr_lat],k=1)
  radar_j,radar_i = np.unravel_index(i,grid_lats.shape)
  print(radar_j,radar_i)
  print(grid_lats[radar_j,radar_i])
  print(grid_lons[radar_j,radar_i])
  timeb = default_timer()
  print(f"  -All that nearest-neighbor finding crap took {timeb-timea:.3f} s")

 timea = default_timer()
 dumln = np.full_like(grid_lons,rdr_lon)
 dumlt = np.full_like(grid_lons,rdr_lat)
 distances = gc2d(grid_lons,grid_lats,dumln,dumlt)
 radar_j,radar_i = np.unravel_index(np.argmin(distances),grid_lons.shape)
 H = terrain_2D[radar_j,radar_i] + radar_dH # Radar site height ASL in the NR grid
 print( (f" The nominal height of this radar is {radar_sites_dict[rad]['ELEV']:.0f} m ASL, which puts it at {radar_sites_dict[rad]['ELEV']-terrain_2D[radar_j,radar_i]:.0f} m above ground."
         f" Regardless, the height in this code has been reset to {H:.0f} m ASL, which is set as {radar_dH:.0f} m above the terrain ({terrain_2D[radar_j,radar_i]:.0f} m) at the location of the radar"))
 for h in range(len(MRMS_heights)):
    distance_at_height = gc3d(grid_lons,grid_lats,dumln,dumlt,H,1e3*MRMS_heights[h])
    distance_3D[h,:,:] = np.minimum(distance_3D[h,:,:],distance_at_height)
 sorted_distances = np.sort(distances.flatten())
 sorted_idxs = np.unravel_index(np.argsort(distances.flatten()),distances.shape)
 ij_within_range = np.argwhere(distances <= max_def_dist)
 rows = ij_within_range[:,0]
 columns = ij_within_range[:,1]
 radar_site_height_grid[rows,columns] = H
 print(np.max(ij_within_range,axis=0))
 print(np.min(ij_within_range,axis=0))

 distances_within_range = distances[distances <= max_def_dist]
 terrain_within_range = terrain_2D[distances <= max_def_dist]
 #print(grid_lats[radar_j,radar_i])
 #print(grid_lons[radar_j,radar_i])
 sorted_j = sorted_idxs[0]
 sorted_i = sorted_idxs[1]
 number_within = len(sorted_distances[sorted_distances <= max_def_dist])
 print(f" This radar is {radar_sites_dict[rad]['BAND']}-band, so the max distance is set to {max_def_dist/1e3:.0f} km. There are {number_within} ({100*number_within/(grid_nx*grid_ny):.3f} % of the total domain) points within this range of the radar site.")
 if number_within < 10:
   print(" There are fewer than 10 domain points within range of this radar site. We will dis-include this radar site.")
   used_radars.remove(rad)
   continue
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

 for v in range(len(VCP_angles)):
   this_radar_beam_heights[v,:,:] = beam_height_2d(H,radians(VCP_angles[v],distances)

 # Now calculate the main array values and account for beam blockages
 blocked_points = np.empty((0,2),dtype=int)
 xc = np.zeros((grid_ny,grid_nx,len(MRMS_heights),dtype=float)
 yc = np.zeros((grid_ny,grid_nx,len(MRMS_heights),dtype=float)
 zc = np.zeros((grid_ny,grid_nx,len(MRMS_heights),dtype=float)
 rbw = radians(beam_width)
 # Where is the beam center going to be relative to the grid?
 for v in range(len(VCP_angles)):
  vr = radians(VCP_angles[v])
  for az in np.arange(0,2*pi,rbw):
   for s in np.arange(0.,max_def_dist,250.):
      xc,yc,zc = distance(meters=s).destination((rdr_lon,rdr_lat),bearing=degrees(az))
      zc = beam_height_direct(H,vr,s)
#      distance_from_beam_center_at_s = np.sqrt((grid_lons
#   if az >= 0.5*rbw and az <= 2*pi-0.5*rbw:
#      idx_j,idx_i = np.where((np.abs(bearing-az) < 0.5*rbw) & (distances < max_def_dist))
#   else:
#      idx_j,idx_i = np.where(((bearing < 0.5*rbw) | (bearing > 2*pi-0.5*rbw)) & (distances < max_def_dist))
      # the where command returns arrays that are almost correctly sorted by distance. But it's not 100% accurate, and that is important.
      # So I'll perform a sort myself
   blocked = np.full((len(VCP_angles),len(idx_j)),False,dtype=bool)
   beam_distances = distances[idx_j,idx_i]
#   print(f"The fraction of grid points in this beam (az = {degrees(az):.2f} deg.) and distance is {len(idx_j)}, ({100*len(idx_j)/number_within:.3f} %)")
   # Sort
   sorted_idxs = np.argsort(beam_distances)
   sorted_j,sorted_i = idx_j[sorted_idxs],idx_i[sorted_idxs]
   sorted_distances = distances[sorted_j,sorted_i]
   for n in range(len(sorted_idxs)):
    j = sorted_j[n]
    i = sorted_i[n]
 #   print(f"Working on point no. {n:04d}, i,j = {i},{j}, distance {beam_distances[n]:.1f} m, bearing {degrees(bearing[j,i]):.3f} deg")
    radar_site_height_grid[j,i] = H
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
          if beam_height < min_beam_height_grid[j,i]:
            closest_radar_dist_grid[j,i] = distances[j,i]
            closest_radar_dir_grid[j,i] = (bearing[j,i] + pi) % (2*pi)
            min_beam_height_grid[j,i] = beam_height
            closest_radar_id_grid[j,i] = i+1
            break
         else:
          continue
    if blocked[0,n]:
     if blocked_points.shape[0] == 0:
        blocked_points = np.array([j,i])
     else:
        blocked_points = np.vstack((blocked_points,[j,i]))
    # Repeat above for max height of radar beam
    radar_z_max = beam_height_direct(H,xt,beam_distances[n])
    if radar_z_max <= terrain_2D[j,i]:
       min_beam_height_grid[j,i] = np.nan
       max_beam_height_grid[j,i] = np.nan
       closest_radar_dir_grid[j,i] = np.nan
       closest_radar_dist_grid[j,i] = np.nan
    else:
       if radar_z_max > max_beam_height_grid[j,i]:
          max_beam_height_grid[j,i] = radar_z_max
#   choice = rng.integers(low=0,high=100,size=1)
#   dist1d = np.zeros(len(sorted_idxs))
#   for n in range(len(sorted_idxs)):
#      dist1d[n] = min_beam_height_grid[sorted_j[n],sorted_i[n]]
#   if choice == 1:
#    plt.plot(sorted_distances/1e3,dist1d,'.',color=colors[cc,:],ms=1,label=r"$\theta$" + f" = {degrees(az):.1f}$\degree$")
#    cc += 1
 n_blocked_points = len(np.unique(blocked_points,axis=0))
 print(f" {n_blocked_points} grid points ({100*n_blocked_points/float(number_within):.2f} % of all points within {max_def_dist/1e3:.0f} km of the radar site) were blocked at the lowest scan angle, but saying nothing about what happened at higher angles")
#   time2 = default_timer()
#   print(f"This gridpoint took {time2-time1:.6f} s")
# print(f"bearing RMSD between simple tangent and spherical geometry: {sqrt(np.mean(diffs**2)):.3f} deg.")
# plt.grid(linestyle="--",color="0.8",linewidth=0.5)
# plt.tick_params(axis='both',labelsize=6)
# plt.xlabel('Distance from radar [km]',fontsize=8)
# plt.ylabel('Height of lowest beam [m]',fontsize=8)
# plt.xlim(0,max_def_dist/1e3)
# plt.legend(loc=0,fontsize=8)
# plt.savefig(f"{out_dir}/radar_{rad}_values.png",dpi=150)
# plt.close()
 for i,z in enumerate(MRMS_heights):
    closest_radar_slant_range[i,:,:] = np.minimum(closest_radar_slant_range[i,:,:],calc_slant_range_2D(radar_site_height_grid,np.full_like(radar_site_height_grid,1e3*z),closest_radar_dist_grid))

 time00 = default_timer()
 print(f" Processing this radar took {time00-time0:.3f} s")
# End radar loop

#closest_radar_slant_range = calc_slant_range_2D(radar_site_height_grid,min_beam_height_grid,closest_radar_dist_grid)
#print(np.nanmin(closest_radar_slant_range),np.nanmax(closest_radar_slant_range),np.nanmean(closest_radar_slant_range),np.nanmedian(closest_radar_slant_range),np.nanstd(closest_radar_slant_range))

time0 = default_timer()
for rz in range(len(MRMS_heights)):
 condition1_grid = ~np.isneginf(max_beam_height_grid) * ~np.isinf(min_beam_height_grid)
 condition2_grid = (MRMS_heights[rz]*1e3 >= min_beam_height_grid) * (1e3*MRMS_heights[rz] <= max_beam_height_grid)
 final_radar_mask[rz,:,:] = np.where(condition1_grid,np.where(condition2_grid,1,0),-1)
time1 = default_timer()
print(f"Time to set final mask is {time1-time0:.1f} s")

ncf = Dataset(f"{out_dir}/radar_mask{tag}.nc",'w',format='NETCDF4')
dimy = ncf.createDimension('latitude',grid_ny)
dimx = ncf.createDimension('longitude',grid_nx)
dimz = ncf.createDimension('height',len(MRMS_heights))
varz = ncf.createVariable('MRMS_heights','i2',dimensions=(dimz))
varid = ncf.createVariable('closest_radar_ID','i1',dimensions=(dimy,dimx))
vardist = ncf.createVariable('distance_to_closest_radar_3D',dimension=(dimz,dimy,dimx))
varmask = ncf.createVariable('mask','i1',dimensions=(dimz,dimy,dimx))
varangle = ncf.createVariable('elevation_angle',dimensions=(dimz,dimy,dimx))
varslant = ncf.createVariable('slant_range',dimensions=(dimz,dimy,dimx))
varz[:] = MRMS_heights
varmask[:] = final_radar_mask
varid[:] = closest_radar_id_grid
vardist[:] = distance_3D
varangle[:] = beam_angle_at_height[:]
varslant[:] = closest_radar_slant_range[:]
ncf.radars_used = used_radars
ncf.radar_IDs = "closest_radar_ID is an integer corresponding to the list 'radars_used'"
ncf.values_key = "1 - within mask; 0 - outside of mask (but point was checked); -1 - gridpoint not checked, but assumed outside mask"
ncf.grid_projection = "MRMS grid (0.01 x 0.01 deg.)"
ncf.close()

for i,z in enumerate(MRMS_heights):
   ncf = Dataset(f"{out_dir}/radar_mask{tag}_{1e3*z:05.0f}m.nc",'w',format='NETCDF4')
   dimy = ncf.createDimension('latitude',grid_ny)
   dimx = ncf.createDimension('longitude',grid_nx)
   varmask = ncf.createVariable('mask','i1',dimensions=(dimy,dimx))
   vardist = ncf.createVariable('distance_to_closest_radar',dimension=(dimy,dimx))
   varmask[:] = final_radar_mask[i,:,:]
   vardist[:] = distance_3D[i,:,:]
   ncf.radars_used = used_radars
   ncf.height = f"{1e3*z:.0f} m ASL"
   ncf.values_key = "1 - within mask; 0 - outside of mask (but point was checked); -1 - gridpoint not checked, but assumed outside mask"
   ncf.grid_projection = "MRMS grid (0.01 x 0.01 deg.)"
   ncf.close()

if True:
 extent = [-127.5,-65,24,51]
 fig_x = 12
 AR = (extent[3]-extent[2])/(extent[1]-extent[0])
 fig_y = AR*fig_x
 height_levs = np.concatenate((np.arange(100.,1000.,100.),np.arange(1000.,10000,500.),np.arange(10000,20000.1,1000.)))
 distance_levs = np.arange(50.,750.,50.)
 angle_levs = np.arange(0,360.1,15.)
 plt.figure(figsize=(fig_x,fig_y))
 ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=prj)
 ax.set_extent(extent,crs=prj)
 ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=1)
 cf = ax.contourf(grid_lons,grid_lats,min_beam_height_grid,levels = height_levs,cmap=colormaps['CMRmap_r'],norm='linear',vmin=100,vmax=20000.,extend='both',transform=prj)
 cb = plt.colorbar(mappable=cf,orientation='horizontal',fraction=0.075,pad=0.01,shrink=0.8,aspect=40)
 cb.ax.tick_params(labelsize=6)
 cb.set_label("Minimum height of radar coverage [m ASL]",fontsize=8)
 plt.savefig(f"{out_dir}/min_beam_height{tag}.png",dpi=150)
 plt.close()

 plt.figure(figsize=(fig_x,fig_y))
 ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=prj)
 ax.set_extent(extent,crs=prj)
 ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=1)
 cf = ax.contourf(grid_lons,grid_lats,max_beam_height_grid,levels = np.arange(10000,25000.,1000.),cmap=colormaps['CMRmap_r'],norm='linear',vmin=10000,vmax=25000.,extend='both',transform=prj)
 cb = plt.colorbar(mappable=cf,orientation='horizontal',fraction=0.075,pad=0.01,shrink=0.8,aspect=40)
 cb.ax.tick_params(labelsize=6)
 cb.set_label("Maximum height of radar coverage [m ASL]",fontsize=8)
 plt.savefig(f"{out_dir}/max_beam_height{tag}.png",dpi=150)
 plt.close()

 plt.figure(figsize=(fig_x,fig_y))
 ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=prj)
 ax.set_extent(extent,crs=prj)
 ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=1)
 cf = ax.contourf(grid_lons,grid_lats,closest_radar_slant_range/1e3,levels = distance_levs,cmap=colormaps['CMRmap_r'],norm='linear',vmin=25.,vmax=750.,extend='both',transform=prj)
 cb = plt.colorbar(mappable=cf,orientation='horizontal',fraction=0.075,pad=0.01,shrink=0.8,aspect=40)
 cb.ax.tick_params(labelsize=6)
 cb.set_label("Maximum height of radar coverage [m ASL]",fontsize=8)
 plt.savefig(f"{out_dir}/nearest_radar_slant_range{tag}.png",dpi=150)
 plt.close()

 plt.figure(figsize=(fig_x,fig_y))
 ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=prj)
 ax.set_extent(extent,crs=prj)
 ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=1)
 cf = ax.contourf(grid_lons,grid_lats,closest_radar_dist_grid/1e3,levels = distance_levs,cmap=colormaps['CMRmap_r'],norm='linear',vmin=25.,vmax=500.,extend='both',transform=prj)
 cb = plt.colorbar(mappable=cf,orientation='horizontal',fraction=0.075,pad=0.01,shrink=0.8,aspect=40)
 cb.ax.tick_params(labelsize=6)
 cb.set_label("Distance to radar providing minimum height coverage [km]",fontsize=8)
 plt.savefig(f"{out_dir}/nearest_radar_dist{tag}.png",dpi=150)
 plt.close()

 plt.figure(figsize=(fig_x,fig_y))
 ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=prj)
 ax.set_extent(extent,crs=prj)
 ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=1)
 cf = ax.contourf(grid_lons,grid_lats,np.rad2deg(closest_radar_dir_grid),levels = angle_levs,cmap = colormaps['twilight'],norm = 'linear',vmin=0.,vmax=360.,transform=prj)
 cb = plt.colorbar(mappable=cf,orientation='horizontal',fraction=0.075,pad=0.01,shrink=0.8,aspect=40)
 cb.ax.tick_params(labelsize=6)
 cb.set_label(r"Angle to radar providing minimum height coveage [$\degree$]",fontsize=8)
 plt.savefig(f"{out_dir}/nearest_radar_angle{tag}.png",dpi=150)
 plt.close()
