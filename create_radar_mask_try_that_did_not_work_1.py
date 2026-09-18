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
from scipy.spatial import cKDTree
from sys import exit

prj = ccrs.PlateCarree()

def gc2d(lon1, lat1, lon2, lat2):

    # Input arguments must be 2D arrays with the same dimensions.
    # Returns a 2D array of distances.

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    a = np.cos(lat1) * np.cos(lat2) * np.cos(dlon) + np.sin(lat1)*np.sin(lat2)
    dist = Re*np.arccos(a)
    return dist

def gc_sclar(lon1,lat1,lon2,lat2):

    # Input arguments must be scalars. Returns a scalar.

    import math as m

    lon1r = m.radians(lon1)
    lon2r = m.radians(lon2)
    lat1r = m.radians(lat1)
    lat2r = m.radians(lat2)
    dlon = lon1r - lon2r
    a = m.cos(lat1r) * m.cos(lat2r) * m.cos(dlon) + m.sin(lat1r)*m.sin(lat2r)
    dist = Re*m.acos(a)
    return dist

def calc_beam_height_spherical(H,s,ang):
   """Compute a radar beam height above sea level over a sphere.
   Required input arguments:
   H - radar site height (ASL)
   s - slant range
   ang - elevation angle of radar beam (radians)"""

   height = sqrt((Re+H)**2 + s**2 + 2*s*(Re+H)*sin(ang)) - Re
   return height

def calc_slant_range(H,Z,xr):
   """Compute slant range of a radar beam.
   Required inputs:
   H - radar site height (ASL)
   Z - some total height level (generally terrain_height + height_above_ground
   xr - horizontal distance along curved surface (surface of Earth)"""

   s = sqrt((Re+H)**2 + (Re+Z)**2 - 2*(Re+H)*(Re+Z)*cos(xr/Re))
   return s

def distance_from_slant(H,Z,S):
   """Compute distance along Earth's surface (circle arc length)
   Required arguments:
   H - radar site height (ASL)
   Z - beam height above surface
   S - radar beam slant range"""

   box = ((Re+H)**2 + (Re + Z)**2 - s**2) / (2*(Re+H)*(Re+Z))
   xr = Re*acos(box)
   return xr

def beam_height_direct(H,angle,xr):
   # angle must be in radians
   # Distances should be in [m]

   term1 = cos(angle)/cos((xr/Re)+angle)
   z = term1*(Re+H)-Re
   return z

def beam_height_2d(H,angle,dist):
   # Input arguments (in this case, xr) must be 2D arrays with the same dimensions.
   # Returns a 2D array of heights

   z2d = np.cos(angle)/np.cos((xr/Re)+angle) * (Re+H) - Re
   return z2d

def beam_height_iter(dist,max_dist,H,ang):
   # ang must be in radians
   # dist is point distance from radar
   # max_dist should always be this_radar_max_leash
   # H is radar height

   # output: beam_height in [m ASL]

   # Calculate candidate heights using the array form of S
   S = np.arange(0.,max_dist + 0.001e3,250.0)
   zzz = np.sqrt((Re+H)**2 + S**2 + 2*S*(Re+H)*sin(ang)) - Re
   # Now compute the error from the intended real distance
   term0 = ((Re+H)**2 + (Re+zzz)**2 - S**2) / (2*(Re+H)*(Re+zzz))
   equals = cos(dist/Re)
   errors = np.abs(term0 - equals)
   # Find the lowest error and use that as the solution
   idx = np.argmin(errors)
   beam_height = zzz[idx]
   return beam_height

rng = default_rng()

out_dir = "/work/noaa/wrfruc/jdduda/radar_mask"
colors = ["tab:blue","tab:orange","tab:red","tab:green","tab:purple","tab:brown","tab:olive",'black']
max_radar_distance = 750e3
make_alt_by_range_plots = False

maj_axis = distance.ELLIPSOIDS['WGS-84'][1]
min_axis = distance.ELLIPSOIDS['WGS-84'][0]
Re = sqrt(maj_axis*min_axis)*1000.
VCP12 = [0.5,0.9,1.3,1.8,2.4,3.1,4.0,5.1,6.4,8.0,10.0,12.5,15.6,19.5]
VCP215 = [0.5,0.9,1.3,1.8,2.4,3.1,4.0,5.1,6.4,8.0,10.0,12.0,14.0,16.7,19.5]
VCP31 = [0.5,1.5,2.5,3.5,4.5]
VCP100 = [0.25,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.5,13.0,14.5,16.0,18.0,20.0,22.0]

MRMS_heights = [0.50,0.75,1.0,1.25,1.50,1.75,2.0,2.25,2.50,2.75,3.0,3.25,3.5,3.75,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,10.,11.,12.,13.,14.,15.,16.,17.,18.,19] # [km]
max_height = 1e3*np.max(MRMS_heights)

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
#print(radar_sites_dict.keys())
#print(radar_sites_dict.values())
#print(radar_sites_dict['KFTG'])

# Setup mask arrays
# THIS ITERATION OF THE MASKING LOGIC PRIORITIZES THE BEAM HEIGHT OVER DISTANCE FROM ANY RADAR SITE
# In other words, if there is a grid point that has two radars within {max_radar_distance}, the below
# arrays will take the data from the radar that has the lowest beam at this point, provided it is above ground
min_beam_height_grid = np.full((grid_ny,grid_nx),1e10,dtype=float)
max_beam_height_grid = np.full((grid_ny,grid_nx),-1e10,dtype=float)
closest_radar_dist_grid = np.zeros((grid_ny,grid_nx),dtype=float)
closest_radar_dir_grid = np.zeros((grid_ny,grid_nx),dtype=float)
final_radar_mask = np.full((len(MRMS_heights),grid_ny,grid_nx),-1,dtype=int)

# Determine minimum angle and thus max distance worth checking around each radar site
# Ideally this would be computed independently for the terrain around each radar site, but the complexity in doing that
# renders it unfeasable at this time. A "typical" optimistic value should suffice.
radar_dH = 30. # hereafter, assumed height of radar above ground at its location
s0 = 1/(Re+radar_dH)*sqrt(2*Re**3*radar_dH + 5*Re**2*radar_dH**2 + 3*Re*radar_dH**3)
th0 = asin(-(2*Re*radar_dH + radar_dH**2 + s0**2)/(2*s0*(Re+radar_dH)))
print(f"   The beam would just brush the ground at slant range {s0:.0f} m (angle of {degrees(th0):.3f})")
th0 = acos(Re/(Re+radar_dH))
print(f"   Alternate calculation of minimum elevation angle to stay above ground: {degrees(th0):.3f} deg.")
S = np.arange(0.,max_radar_distance + 0.001e3,250.0)
Z = np.zeros(len(S),dtype=float)
xr = np.zeros(len(S),dtype=float)
for ii,s in enumerate(S):
 Z[ii] = sqrt((Re+radar_dH)**2 + s**2 + 2*s*(Re+radar_dH)*sin(th0)) - Re # calc_beam_height_spherical(H,s,th0)
 terma = (Re+radar_dH)**2 + (Re+Z[ii])**2 - s**2
 termb = 2*(Re + radar_dH)*(Re + Z[ii])
 argt = terma/termb
 xr[ii] = acos(argt)*Re

idx = np.argwhere(Z >= max_height)[0][0]
print(f"   The lowest possible elevation angle beam that stays above ground exceeds the highest MRMS data level at a distance of {xr[idx]/1e3:.1f} km")
this_radar_max_leash = xr[idx]

used_radars = []
for i,rad in enumerate(radar_sites_dict.keys()):

 time0 = default_timer()
 used_radars.append(rad)
# if i > 5:
#  break

 if make_alt_by_range_plots:
  plt.figure(figsize=(5,5))
  plt.subplots_adjust(left=0.125,bottom=0.1,right=0.98,top=0.99)

 print("**********************************")
 print(f"*** Working on radar site {rad} ***")
 print("**********************************")
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

 # Obtain NR-gridpoint closest to radar site
 if False:
  lat1d = grid_lats.flatten()
  lon1d = grid_lons.flatten()
  latlon = np.vstack((lon1d,lat1d)).T
  timea = default_timer()
  tree = cKDTree(latlon)
  d,i = tree.query([rdr_lon,rdr_lat],k=1)
  radar_j,radar_i = np.unravel_index(i,grid_lats.shape)
  print(radar_j,radar_i)
  print(grid_lats[radar_j,radar_i])
  print(grid_lons[radar_j,radar_i])
  timeb = default_timer()
  print(f"All that nearest-neighbor finding crap took {timeb-timea:.3f} s")

 timea = default_timer()
 dumln = np.full_like(grid_lons,rdr_lon)
 dumlt = np.full_like(grid_lons,rdr_lat)
 distances = gc2d(grid_lons,grid_lats,dumln,dumlt)
 radar_j,radar_i = np.unravel_index(np.argmin(distances),grid_lons.shape)
 H = terrain_2D[radar_j,radar_i] + radar_dH # Radar site height ASL in the NR grid
 print( (f" The nominal height of this radar is {radar_sites_dict[rad]['ELEV']:.0f} m ASL, which puts it at {radar_sites_dict[rad]['ELEV']-terrain_2D[radar_j,radar_i]:.0f} m above ground."
         f" Regardless, the height in this code has been reset to {H:.0f} m ASL, which is set as {radar_dH:.0f} m above the terrain ({terrain_2D[radar_j,radar_i]:.0f} m) at the location of the radar"))
 sorted_distances = np.sort(distances.flatten())
 sorted_idxs = np.unravel_index(np.argsort(distances.flatten()),distances.shape)
 #print(grid_lats[radar_j,radar_i])
 #print(grid_lons[radar_j,radar_i])
 sorted_j = sorted_idxs[0]
 sorted_i = sorted_idxs[1]
 number_within = len(sorted_distances[sorted_distances <= this_radar_max_leash])
 print(f"There are {number_within} ({100*number_within/(grid_nx*grid_ny):.3f} % of the total domain) points within {this_radar_max_leash/1e3:.1f} km of the radar site")
 if number_within < 10:
   print("There are fewer than 10 domain points within range of this radar site. We will dis-include this radar site.")
   del used_radars[rad]
   continue
 timeb = default_timer()
 print(f"Time to sort the entire distance array: {timeb-timea:.3f} s")

 S = np.arange(0.,max_radar_distance + 0.001e3,250.0)
 if make_alt_by_range_plots:
  for I,ang in enumerate([nt,radians(10.0),radians(20.0),0.0,radians(-1.0)]):
   Z = np.zeros(len(S),dtype=float)
   xr = np.zeros(len(S),dtype=float)
   for ii,s in enumerate(S):
    Z[ii] = sqrt((Re+H)**2 + s**2 + 2*s*(Re+H)*sin(ang)) - Re # - terrain_2D[radar_j,radar_i] to get heights AGL
    terma = (Re+H)**2 + (Re+Z[ii])**2 - s**2
    termb = 2*(Re + H)*(Re + Z[ii])
    argt = terma/termb
    xr[ii] = acos(argt)*Re
  #  print(f"slant range: {s/1000:.3f} km , {Z[ii]:10.2f} m elev (horiz. distance: {xr[ii]/1000.:.3f} km)")
   plt.plot(xr/1000,Z,'-',color=colors[I],linewidth=1,label=f"{degrees(ang):.1f}\N{DEGREE SIGN}")
  plt.grid(linestyle=':',color='0.8')
  plt.tick_params(axis='both',labelsize=6)
  plt.xlabel("horizontal range [km]",fontsize=8)
  plt.ylabel("Height ASL [m]",fontsize=8)
  plt.xticks(np.concatenate((np.arange(10.,100.,10.),np.arange(100.,300.,25.),np.arange(300.,max_radar_distance + 0.1,50.))))
  plt.yticks(np.arange(1000.,25000.1,1000.))
  plt.xlim(0,max_radar_distance/1e3)
  plt.ylim(0,25001)
  plt.legend(loc=0,fontsize=8)
  plt.savefig(f"{out_dir}/site_{rad}_radar_beam_height_curved_Earth_tall.png",dpi=125)

  plt.figure(figsize=(7,5))
  plt.subplots_adjust(left=0.125,bottom=0.1,right=0.98,top=0.99)

 ### Beam hitting the ground
 ### Two scenarios
 ### Scenario 1: The beam just brushes/nudges the ground before going back up. So there is only one solution.
 ### This provides the constraint that dZ/dxr = 0, which is equivalent to dZ/dS = 0
 # H here must be height above ground, not height above sea level
 # This height isn't precisely known, but in the NEXRAD network it is somewhere in the range of 20-50 m. 30 m is probably appropriate
 # Future advancements of this could compute an actual height difference between the radar location and the underlying terrain
# Z = np.zeros(len(S),dtype=float)
# xr = np.zeros(len(S),dtype=float)
# for ii,s in enumerate(S):
#  Z[ii] = sqrt((Re+H)**2 + s**2 + 2*s*(Re+H)*sin(th0)) - Re # calc_beam_height_spherical(H,s,th0)
#  terma = (Re+H)**2 + (Re+Z[ii])**2 - s**2
#  termb = 2*(Re + H)*(Re + Z[ii])
#  argt = terma/termb
#  xr[ii] = acos(argt)*Re

# idx = np.argwhere(Z >= max_height)[0][0]
# print(f"   The lowest possible elevation angle beam that stays above ground exceeds the highest MRMS data level at a distance of {xr[idx]/1e3:.1f} km")
# this_radar_max_leash = xr[idx]
 # Alternate calculation derived from Law of sines
 #th0 = acos(Re/(Re+radar_dH))
 #print(f"   Alternate calculation of minimum elevation angle to stay above ground: {degrees(th0):.3f} deg.")
# Z = np.zeros(len(S),dtype=float)
# xr = np.zeros(len(S),dtype=float)
# for ii,s in enumerate(S):
#  Z[ii] = sqrt((Re+H)**2 + s**2 + 2*s*(Re+H)*sin(th0)) - Re # calc_beam_height_spherical(H,s,th0)
#  terma = (Re+H)**2 + (Re+Z[ii])**2 - s**2
#  termb = 2*(Re + H)*(Re + Z[ii])
#  argt = terma/termb
#  xr[ii] = acos(argt)*Re

# idx = np.argwhere(Z >= max_height)[0][0]
# print(f"   The lowest possible elevation angle beam that stays above ground exceeds the highest MRMS data level at a distance of {xr[idx]/1e3:.1f} km")

 plt.plot(xr/1000,Z,'-',color='black',linewidth=1,label=f"{degrees(th0):.3f}\N{DEGREE SIGN}")

 ### Scenario 2: The beam path intersects the ground twice. Obviously, only the closer (lower s0) value is physically reasonable.
 ### But the equations needed are different than for Scenario 1.
 # This requires use of different problem set, i.e., solving a system of two equations, a line, and a circle
 # The solution is x = -0.5*sin(2*theta)*(Re+H) +/- sqrt(tan(theta)**2*(Re**2+H**2)-2*Re*H)/(1+tan(theta)**2)
 #                 y = (Re + H)*(1-sin(theta)**2) +/- 0.5*sin(2*theta)*sqrt(tan(theta)**2*(Re**2+H**2)-2*Re*H)
 # This method also gives the minimum angle at which a solution is achieved, which is equivalent to the angle at which
 # the beam would just brush the ground
 # angle_threshold = atan2(sqrt(2*Re*H),(Re**2+H**2))

 if make_alt_by_range_plots:
  plt.grid(linestyle=':',color='0.8')
  plt.tick_params(axis='both',labelsize=6)
  plt.xlabel("horizontal range [km]",fontsize=8)
  plt.ylabel("Height ASL [m]",fontsize=8)
  plt.xticks(np.concatenate((np.arange(10.,100.,10.),np.arange(100.,300.,25.),np.arange(300.,max_radar_distance + 0.1,50.))))
  plt.xlim(0,max_radar_distance/1e3)
  if False:
   plt.yticks(np.arange(0,2000.,100.))
   plt.ylim(-10,2000)
  else:
   plt.yticks(np.arange(1000.,25000.1,1000.))
   plt.ylim(0,25001)
  plt.legend(loc=0,fontsize=8)
  plt.savefig(f"{out_dir}/site_{rad}_radar_beam_height_curved_Earth.png",dpi=125)
  plt.close()

 # Now build the required array values for NR-gridpoints around this point

 time1 = default_timer()
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
 time2 = default_timer()
 print(f" Time to calculate bearings using vectorized functions: {time2-time1:.2f} s")
 timea = default_timer()
 diffs = np.zeros_like(grid_lats_r)
 bearing_BM = np.zeros_like(grid_lats_r)
 if False:
  # repeat the scalar versions of just these quantities for the loop for comparison - then delete them
  vector_rad = Re*np.array([cos_rdr_lat*cos_rdr_lon,cos_rdr_lat*sin_rdr_lon,sin_rdr_lat])
  norm_gc = np.cross(vector_rad,vector_np)
 # print(f"Radar [rad] site location in Earth spherical coordinates: {vector_rad[0]/1e3:.0f} km i-hat  +  {vector_rad[1]/1e3:.0f} km j-hat  +  {vector_rad[2]/1e3:.0f} km k-hat")
 # print(f"Normal vector with radar and north pole great circle: {norm_gc[0]/1e3:.0f} km i-hat  +  {norm_gc[1]/1e3:.0f} km j-hat  +  {norm_gc[2]/1e3:.0f} km k-hat")
  #
  for ij in range(len(sorted_distances)):
   j = sorted_j[ij]
   i = sorted_i[ij]
#   print(ij,j,i)
   # Find bearing of gridpoint to radar site - this requires finding the dihedral angle between the planes associated
   # with the great circles containing the radar site/grid point against that going meridionally through the radar site
   term1 = cos(grid_lats_r[j,i])
   term2 = cos(grid_lons_r[j,i])
   term3 = sin(grid_lons_r[j,i])
   term4 = sin(grid_lats_r[j,i])
   vector_pt = Re*np.array([term1*term2,term1*term3,term4])
   norm_pt = np.cross(vector_rad,vector_pt)
   angle = acos(np.inner(norm_gc,norm_pt)/(mag(norm_gc)*mag(norm_pt)))
   if grid_lons[j,i] > rdr_lon:
      bearing_BM[j,i] = angle
   else:
      bearing_BM[j,i] = 2*pi - angle
   diffs[j,i] = bearing_BM[j,i]-bearing[j,i]
   if not np.isclose(bearing_BM[j,i],bearing[j,i],atol=1e-4):
    print(f"{ij:7d} j={j:4d},i={i:4d} {bearing_BM[j,i]:.5f} {bearing[j,i]:.5f} {bearing_BM[j,i]-bearing[j,i]:.8f}")
   # This is merely to compare above with a simple (but likely lessa ccurate) way of computing the bearing
   if False:
    dlat = grid_lats[j,i]-rdr_lat
    dlon = grid_lons[j,i]-rdr_lon
    lazy_angle = (0.5*pi - atan2(dlat,dlon)) % (2*pi)
    diff_ = degrees(bearing[j,i]-lazy_angle)
    diffs[j,i] = diff_
    choice = rng.integers(0,high=100000,size=1)
    if choice == 1:
     EW = "W"
     NS = "S"
     if grid_lons[j,i] > rdr_lon:
      EW = "E"
     if grid_lats[j,i] > rdr_lat:
      NS = "N"
     quadrant = NS+EW
     print(f"NR gridpoint {grid_lons[j,i]:.3f} / {grid_lats[j,i]:.3f}, vector: {vector_pt[0]/1e3:.0f} km i-hat  +  {vector_pt[1]/1e3:.0f} km j-hat  +  {vector_pt[2]/1e3:.0f} km k-hat")
     print(f"normal vector to the great circle: {norm_pt[0]/1e3/Re:.0f} km i-hat  +  {norm_pt[1]/1e3/Re:.0f} km j-hat  +  {norm_pt[2]/1e3/Re:.0f} km k-hat")
     print(f"radar site lon/lat ({rdr_lon} / {rdr_lat}) | lon={grid_lons[j,i]:.4f}, lat={grid_lats[j,i]:.4f}, bearing {degrees(bearing[j,i]):.3f} deg. (simple-trig bearing: {degrees(lazy_angle):.3f} - quadrant: {quadrant}), difference: {diff_:.5f} deg.")
   if sorted_distances[ij] > this_radar_max_leash:
    #print(f"At point #{ij}, the distance has become greater than {max_radar_distance/1e3:.0f} km ({sorted_distances[ij]:.1f} m ). So it's time to end this loop")
    break
  #print(f"bearing RMSD between simple tangent and spherical geometry: {sqrt(np.mean(diffs**2)):.3f} deg.")
  #timeb = default_timer()
 # print(f"RMSD between the two methods of calculating bearing: {sqrt(np.mean(diffs**2)):.3f}")
  #print(f"Time to calculate bearings one at a time: {timeb-timea:.2f} s")
 blocked_idxs = np.empty((2,0),dtype=int)
 # Re-do the loop with bearing now computed and make computations
 # Loop in order of distance from radar first!
 leash_pts = 1
 identified_blocked_pts = 0
 choice = 0
 kill_loop = False
 all_times = np.zeros(2,dtype=float)
 for ij in range(len(sorted_distances)):
   if kill_loop:
    break
   j = sorted_j[ij]
   i = sorted_i[ij]
  # if ij == choice:
  #  print(f"working on gridpoint x={i:4d}, y = {j:4d}, with a distance of {sorted_distances[ij]:.0f} m ({ij+1}st/nd/th closest)")
  #  choice += rng.integers(0,high=100000,size=1)
# for j in range(y1,y2+1):
#  for i in range(x1,x2+1):
   # We already have "distances" array of each grid point from the radar site; no need to recompute - these are "xr"s
#   time1 = default_timer()
   for v in range(len(VCP_angles)):
      vr = radians(VCP_angles[v])
      beam_height = beam_height_direct(H,vr,sorted_distances[ij])
  #    if beam_height > max_height:
  #       print(f"Breaking loop for point j,i = {j},{i}, distance of {sorted_distances[ij]/1e3:.1f} km, elevation angle of {degrees(vr):.1f} (beam height was {beam_height:.1f} m)")
  #       kill_loop = True
  #       break
      # First check - is the beam below the lowest beam height at this grid point?
      if beam_height < min_beam_height_grid[j,i]:
         if beam_height > terrain_2D[j,i]:
            closest_radar_dist_grid[j,i] = distances[j,i]
            closest_radar_dir_grid[j,i] = (bearing[j,i] + pi) % (2*pi)
            min_beam_height_grid[j,i] = beam_height
            break
         else:
            # Beam is below the ground at this point
            # All other gridpoints along this beam beyond this distance need to be set to "blocked"
            # But wait...what if we check the higher elevation angles first?
            exists = False
            if blocked_idxs.shape[1] > 0:
               exists = np.any(np.all(blocked_idxs == [j,i],axis=1))
            if exists:
          #     print(f"gridpoint j/i = {j}/{i}, lon/lat = {grid_lons[j,i]:.3f}/{grid_lats[j,i]:.3f}, distance = {sorted_distances[ij]:.0f} m, bearing {bearing[j,i]:.3f} deg., has already been marked for beam blockage")
               identified_blocked_pts += 1
               continue
        #    print(f"Beam is below ground at j/i = {j}/{i}, lon/lat = {grid_lons[j,i]:.3f}/{grid_lats[j,i]:.3f} distance, {sorted_distances[ij]:.0f} m ({np.abs(beam_height-terrain_2D[j,i]):.1f} m below ground)")
     #       print(f"All grid points within half a beamwidth of this radial ({bearing[j,i]:.3f} deg.) and more distant than this point ({sorted_distances[ij]:.0f} m) will be labeled as potentially blocked")
            idxs = np.argwhere((np.abs(bearing - bearing[j,i]) < 0.5*radians(beam_width)) & (distances > sorted_distances[ij]) & (distances <= this_radar_max_leash))
        #    print(f"This many points added to blocked beam points: {len(idxs)}")
           # ff = open(f"diag_blocked_beam_pt{ij:06d}.txt",'w')
           # ff.writelines(f"Beam is below ground at j/i = {j}/{i}, lon/lat = {grid_lons[j,i]:.3f}/{grid_lats[j,i]:.3f} ({np.abs(beam_height-terrain_2D[j,i]):.1f} m below ground)\n")
           # ff.writelines(f"All grid points within half a beamwidth of this radial ({bearing[j,i]:.3f} deg.) and more distant than this point ({sorted_distances[ij]:.0f} m) will be labeled as potentially blocked\n")
          #  for a in range(len(idxs)):
          #    j1 = idxs[a][0]
          #    i1 = idxs[a][1]
          #    ff.writelines(f"j={j1}, i={i1}, lon/lat={grid_lons[j1,i1]:.3f}/{grid_lats[j1,i1]:.3f}, distance: {distances[j1,i1]:.0f} m, bearing: {bearing[j1,i1]:.3f} deg\n")
          #  ff.close()
            if blocked_idxs.shape[1] == 0:
               blocked_idxs = idxs
            else:
               blocked_idxs = np.vstack((blocked_idxs,idxs))
            blocked_idxs = np.unique(blocked_idxs,axis=0)
          #  print(blocked_idxs.shape)
            # Check next elevation angle to see if it will be above ground
   # Repeat above for max height of radar beam
   radar_z_max = beam_height_direct(H,xt,sorted_distances[ij])
   if radar_z_max <= terrain_2D[j,i]:
      min_beam_height_grid[j,i] = np.nan
      max_beam_height_grid[j,i] = np.nan
      closest_radar_dir_grid[j,i] = np.nan
      closest_radar_dist_grid[j,i] = np.nan
   else:
      if radar_z_max > max_beam_height_grid[j,i]:
        if radar_z_max > max_height + 1e3:
          max_beam_height_grid[j,i] = max_height + 1e3
        else:
          max_beam_height_grid[j,i] = radar_z_max
   if sorted_distances[ij] > this_radar_max_leash:
    print(f"At point #{ij}, the distance has become greater than {this_radar_max_leash/1e3:.1f} km ({sorted_distances[ij]:.1f} m ). So it's time to end this loop")
    break
   else:
    leash_pts += 1
 print(f" {len(blocked_idxs)} grid points ({100*len(blocked_idxs)/float(leash_pts):.2f} % of all points within {this_radar_max_leash/1e3:.0f} km of the radar site) have been marked as the radar beam potentially being below ground")
 #print(f"The above number should be the same as this one: {identified_blocked_pts}")
#   time2 = default_timer()
#   print(f"This gridpoint took {time2-time1:.6f} s")
  # end i loop
 # end j loop
# print(f"bearing RMSD between simple tangent and spherical geometry: {sqrt(np.mean(diffs**2)):.3f} deg.")
 time00 = default_timer()
 print(f" Processing this radar took {time00-time0:.3f} s")
# End radar loop

if False:
 plt.figure(figsize=(10,10))
 ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=prj)
 for n in range(0,len(blocked_idxs),10):
    j = blocked_idxs[n,0]
    i = blocked_idxs[n,1]
    ax.plot(grid_lons[j,i],grid_lats[j,i],'r.',ms=2,transform=prj)
 ax.set_extent([rdr_lon-5,rdr_lon+5,rdr_lat-5,rdr_lat+5],crs=prj)
 ax.add_feature(cfeature.STATES.with_scale('10m'),edgecolor='black',linewidth=1)
 plt.savefig(f"{out_dir}/blocked_gridpoints_{rad}.png",dpi=150)
 plt.close()

time0 = default_timer()
for rz in range(len(MRMS_heights)):
 condition1_grid = ~np.isneginf(max_beam_height_grid) * ~np.isinf(min_beam_height_grid)
 condition2_grid = (MRMS_heights[rz]*1e3 >= min_beam_height_grid) * (1e3*MRMS_heights[rz] <= max_beam_height_grid)
 final_radar_mask[rz,:,:] = np.where(condition1_grid,np.where(condition2_grid,1,0),-1)
time1 = default_timer()
print(f"Time to set final mask is {time1-time0:.1f} s")

ncf = Dataset(f"{out_dir}/radar_mask_try_1.nc",'w',format='NETCDF4')
dimy = ncf.createDimension('latitude',grid_ny)
dimx = ncf.createDimension('longitude',grid_nx)
dimz = ncf.createDimension('height',len(MRMS_heights))
varz = ncf.createVariable('MRMS_heights','i2',dimensions=(dimz))
#varlat = ncf.createVariable('latitudes','f4',dimensions=(dimy,dimx))
#varlon = ncf.createVariable('longitudes','f4',dimensions=(dimy,dimx))
varmask = ncf.createVariable('mask','i1',dimensions=(dimz,dimy,dimx))
varz[:] = MRMS_heights
#varlat[:] = grid_lats
#varlon[:] = grid_lons
varmask[:] = final_radar_mask
ncf.radars_used = used_radars
ncf.values_key = "1 - within mask; 0 - outside of mask (but point was checked); -1 - gridpoint not checked, but assumed outside mask"
ncf.grid_projection = "Same as nature run UPP output"
ncf.close()

for i,z in enumerate(MRMS_heights):
   ncf = Dataset(f"{out_dir}/radar_mask_{1e3*z:05.0f}m.nc",'w',format='NETCDF4')
   dimy = ncf.createDimension('latitude',grid_ny)
   dimx = ncf.createDimension('longitude',grid_nx)
   varmask = ncf.createVariable('mask','i1',dimensions=(dimy,dimx))
   varmask[:] = final_radar_mask[i,:,:]
   ncf.radars_used = used_radars
   ncf.height = f"{1e3*z:.0f} m ASL"
   ncf.values_key = "1 - within mask; 0 - outside of mask (but point was checked); -1 - gridpoint not checked, but assumed outside mask"
   ncf.grid_projection = "Same as nature run UPP output"
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
 plt.savefig(f"{out_dir}/min_beam_height_1.png",dpi=150)
 plt.close()

 plt.figure(figsize=(fig_x,fig_y))
 ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=prj)
 ax.set_extent(extent,crs=prj)
 ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=1)
 cf = ax.contourf(grid_lons,grid_lats,max_beam_height_grid,levels = np.arange(10000.,25000.,1000.),cmap=colormaps['plasma_r'],norm='linear',vmin=10000,vmax=25000.,extend='both',transform=prj)
 cb = plt.colorbar(mappable=cf,orientation='horizontal',fraction=0.075,pad=0.01,shrink=0.8,aspect=40)
 cb.ax.tick_params(labelsize=6)
 cb.set_label("Maximum height of radar coverage [m ASL]",fontsize=8)
 plt.savefig(f"{out_dir}/max_beam_height_1.png",dpi=150)
 plt.close()

 plt.figure(figsize=(fig_x,fig_y))
 ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=prj)
 ax.set_extent(extent,crs=prj)
 ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=1)
 cf = ax.contourf(grid_lons,grid_lats,closest_radar_dist_grid/1e3,levels = distance_levs,cmap=colormaps['CMRmap_r'],norm='linear',vmin=0.,vmax=650.,extend='both',transform=prj)
 cb = plt.colorbar(mappable=cf,orientation='horizontal',fraction=0.075,pad=0.01,shrink=0.8,aspect=40)
 cb.ax.tick_params(labelsize=6)
 cb.set_label("Distance to radar providing minimum height coverage [km]",fontsize=8)
 plt.savefig(f"{out_dir}/nearest_radar_dist_1.png",dpi=150)
 plt.close()

 plt.figure(figsize=(fig_x,fig_y))
 ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=prj)
 ax.set_extent(extent,crs=prj)
 ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='black',linewidth=1)
 cf = ax.contourf(grid_lons,grid_lats,np.rad2deg(closest_radar_dir_grid),levels = angle_levs,cmap = colormaps['twilight'],norm = 'linear',vmin=0.,vmax=360.,transform=prj)
 cb = plt.colorbar(mappable=cf,orientation='horizontal',fraction=0.075,pad=0.01,shrink=0.8,aspect=40)
 cb.ax.tick_params(labelsize=6)
 cb.set_label(r"Angle to radar providing minimum height coveage [$\degree$]",fontsize=8)
 plt.savefig(f"{out_dir}/nearest_radar_angle_1.png",dpi=150)
 plt.close()

ncf = Dataset(f"{out_dir}/radar_mask_method_1_diag.nc",'w',format='NETCDF4')
dimy = ncf.createDimension('latitude',grid_ny)
dimx = ncf.createDimension('longitude',grid_nx)
varn = ncf.createVariable('lowest_height','f',dimensions=(dimy,dimx))
varx = ncf.createVariable('highest_height','f',dimensions=(dimy,dimx))
vard = ncf.createVariable('closest_distance','f',dimensions=(dimy,dimx))
varr = ncf.createVariable('closest_direction','f',dimensions=(dimy,dimx))
varn[:] = min_beam_height_grid
varx[:] = max_beam_height_grid
vard[:] = closest_radar_dist_grid
varr[:] = closest_radar_dir_grid
ncf.radars_used = used_radars
ncf.grid_projection = "Same as nature run UPP output"
ncf.close()
