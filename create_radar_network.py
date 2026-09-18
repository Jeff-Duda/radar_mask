# The choice of creating a network depends on creating a domain mask of where radars can go.
# The following procedure was used.

# Obtain boundary outline coordinates for the CONUS
# The mask is constructed using the following rules:
# -Mask out area within 100 km of a LAND international border
#   -This tends to cut out portions of the Great Lakes region where radars currently exist (e.g., KDTX, KCLE, and KBUF)
#   -To fix this, grab the Great Lakes shorelines and create a joint polygon from the union of the buffer of all shorelines of about 1.0 deg
#   -Add this back into the grid
# -Also added the Olympic peninsula of NW WA state since there is a water border there that the CONUS boundary shapefile doesn't distinguish
# -Pacific and Atlantic (Gulf of Mexico included) coastlines have no restriction, except for where they meet a land border

from cartopy import crs
from cartopy import feature
from matplotlib import use, colormaps
use('agg')
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
import shapely
from cartopy.io.shapereader import Reader
from geopy import distance
from math import sqrt, radians, cos, sin, acos
from sys import exit
from os.path import isfile
from timeit import default_timer

n_radars = 150
minimum_radar_separation = 200e3 # [m]

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

def gc2d(lon1, lat1, lon2, lat2):
    # Input arguments must be 2D arrays with the same dimensions.
    # Returns a 2D array of distances.

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    a = np.cos(lat1) * np.cos(lat2) * np.cos(dlon) + np.sin(lat1)*np.sin(lat2)
    a = np.maximum(-1.0,np.minimum(1.0,a))
    if np.any(np.abs(a) > 1.0):
       print("Problem...a distance is impossible")
       print(a[a > 1.0])
       print(np.argwhere(a > 1.0))
    dist = Re*np.arccos(a)
    return dist

projPC = crs.PlateCarree()
rng = np.random.default_rng()

# Get terrain and landmask information from nature run grid file
geo_file = "/work/noaa/wrfruc/jdduda/NR_to_MRMS_grid.nc"
nc = Dataset(geo_file,'r')
terrain_height = nc.variables['HGT_M'][:][0,:,:]
landmask = nc.variables['LANDMASK'][:][0,:,:]
nc.close()
# Use that information to calculate terrain gradient/slope
terrain_gradient_vect_2D_i,terrain_gradient_vect_2D_j = np.gradient(terrain_height)
joint_arry = np.vstack((terrain_gradient_vect_2D_i.flatten(),terrain_gradient_vect_2D_j.flatten()))
terrain_gradient_scalar = np.linalg.norm(joint_arry,axis=0)
terrain_gradient_scalar = np.reshape(terrain_gradient_scalar,terrain_height.shape)
# Get CONUS boundary coordinates and set map extent
CONUS_5m_shapefile = "/work/noaa/wrfruc/jdduda/CONUS_shapefiles/cb_2018_us_nation_5m.shp"
reader = Reader(CONUS_5m_shapefile)
CONUS_geometry = next(reader.geometries())
polygons = list(CONUS_geometry.geoms)
conus_geom = max(polygons, key=lambda p: p.area)
CONUS_bounds = conus_geom.bounds
x1 = 0.5*np.floor(CONUS_bounds[0]/0.5)
x2 = 0.5*np.ceil(CONUS_bounds[2]/0.5)
y1 = 0.5*np.floor(CONUS_bounds[1]/0.5)
y2 = 0.5*np.ceil(CONUS_bounds[3]/0.5)
#map_extent = [x1,x2,y1,y2]
map_extent = [x1-1,x2+1,y1-1,y2+1]

# Create MRMS grid
MRMS_lat_1 = 20.005
MRMS_lon_1 = -129.995 # equivalent to 230.005 - 360
MRMS_lat_2 = 54.995
MRMS_lon_2 = -60.005 # equivalent to 299.995 - 360
grid_nx = 7000
grid_ny = 3500
lat1d = np.linspace(MRMS_lat_1,MRMS_lat_2,grid_ny)
lon1d = np.linspace(MRMS_lon_1,MRMS_lon_2,grid_nx)
mrms_lons,mrms_lats = np.meshgrid(lon1d,lat1d)
#map_extent = [MRMS_lon_1,MRMS_lon_2,MRMS_lat_1,MRMS_lat_2]
dx = Re*np.cos(np.deg2rad(mrms_lats))*radians(0.01)
dy = np.full_like(mrms_lats,Re*radians(0.01))
grid_spacing = np.sqrt(dx*dy)
terrain_slope = np.rad2deg(np.arctan(terrain_gradient_scalar/grid_spacing))
# Diagnostic for determining slope to avoid for placing radars
if False:
 plt.figure(figsize=(10,10))
 ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=projPC)
 ax.add_feature(feature.STATES.with_scale('10m'),linewidth=0.5,edgecolor='0.5')
 ax.add_feature(feature.BORDERS.with_scale('50m'),linewidth=1,edgecolor='black')
 ax.contour(mrms_lons,mrms_lats,terrain_height,levels=np.arange(100.,4000.,25.),linewidths=0.5,colors='red')
 cf = ax.quiver(mrms_lons[::1,::1],mrms_lats[::1,::1],terrain_gradient_vect_2D_i[::1,::1],terrain_gradient_vect_2D_j[::1,::1],angles='xy',scale=500,width=0.001,headwidth=5,headlength=3,headaxislength=2)
 #cb = plt.colorbar(mappable=cf,orientation='vertical',aspect=40,fraction=0.05,pad=0.01,shrink=0.7,ticks=np.arange(0.,75.1,5.0))
 #cb.set_label("Gradient magnitude",fontsize=8)
 #cb.set_label("Terrain slope [\N{DEGREE SIGN}]",fontsize=8)
 #cb.ax.tick_params(labelsize=6)
 ax.set_extent([-103,-102,39,40],crs=projPC)
 plt.savefig("nature_run_terrain_gradient_vectors.png",dpi=200)
 plt.close()

 plt.figure(figsize=(10,10))
 ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=projPC)
 ax.add_feature(feature.STATES.with_scale('10m'),linewidth=0.5,edgecolor='0.5')
 ax.add_feature(feature.BORDERS.with_scale('50m'),linewidth=1,edgecolor='black')
 cf = ax.contourf(mrms_lons,mrms_lats,terrain_gradient_scalar,cmap='plasma',transform=projPC)
 #cf = ax.contourf(mrms_lons,mrms_lats,terrain_slope,levels=np.arange(0.,15.1,1.0),cmap='cubehelix_r',transform=projPC,extend='max')
 cb = plt.colorbar(mappable=cf,orientation='vertical',aspect=40,fraction=0.05,pad=0.01,shrink=0.5)#,ticks=np.arange(0,15.1,1.0))
 cb.set_label("Gradient magnitude",fontsize=8)
 #cb.set_label("Terrain slope [\N{DEGREE SIGN}]",fontsize=8)
 cb.ax.tick_params(labelsize=6)
 ax.set_extent([-125,-103,27.5,50],crs=projPC)
 plt.savefig("nature_run_terrain_gradient_west_US.png",dpi=500)
 plt.close()
 exit()

# Creating CONUS mask not near international borders
reader = Reader("/work/noaa/wrfruc/jdduda/CONUS_shapefiles/tl_2019_us_coastline.shp")
plt.figure(figsize=(10,4.75))
ax = plt.gcf().add_axes([0.001,0.001,0.998,0.998],projection=projPC)
ax.add_feature(feature.STATES.with_scale('10m'),linewidth=0.5,edgecolor='0.5')
ax.add_feature(feature.BORDERS.with_scale('50m'),linewidth=1,edgecolor='black')
ax.add_geometries([conus_geom],crs=projPC,facecolor='none',edgecolor='black',linewidth=1,zorder=1)
i = 0
for record in reader.records():
   if "great lakes" in record.attributes['NAME'].lower():
#    if record.geometry.length >= 0.5:
      i += 1
      lake_geom = shapely.buffer(record.geometry,distance=1)
      if i == 1:
         union_lake_geom = lake_geom
      else:
         union_lake_geom = shapely.union(union_lake_geom,lake_geom)
#      ax.add_geometries([record.geometry],crs=projPC,facecolor='none',edgecolor='salmon',linewidth=0.5,zorder=10)
#      print(f"Great Lakes shapefile {i} has length {record.geometry.length:.4f}")
#ax.add_geometries(union_lake_geom,crs=projPC,facecolor='salmon',edgecolor='none',linewidth=0.5,zorder=1)
land_border_shpfile = "/work/noaa/wrfruc/jdduda/CONUS_shapefiles/data/DoS_LSIB_v11_4_24Feb2025.shp"
land_border_reader = Reader(land_border_shpfile)
for record in land_border_reader.records():
   if ("UNITED STATES" in record.attributes['COUNTRY1'].upper() or "UNITED STATES" in record.attributes['COUNTRY2'].upper()) and \
      ("CANADA" in record.attributes['COUNTRY1'].upper() or "CANADA" in record.attributes['COUNTRY2'].upper()):
#      ax.add_geometries([record.geometry],crs=projPC,facecolor='none',edgecolor='magenta',linewidth=1)
      # Index 1 of geoms appears to be the land border between Alaska and the Yukon
      # Index 0 of geoms is the classic land border between the northern lower 48 and southern Canada
      canada_us_border = record.geometry.geoms[0]
      border_lons,border_lats = canada_us_border.xy
      # Create buffer around border
      canada_border_buffer_poly = shapely.buffer(record.geometry.geoms[0],distance=0.9,single_sided=False)
 #     border_lons = canada_us_border[0]
 #     border_lats = canada_us_border[1]
#      shp_feat1 = feature.ShapelyFeature([canada_us_border],crs=projPC,facecolor='none',edgecolor='blue',linewidth=1)
#      ax.add_geometries(canada_border_buffer_poly,crs=projPC,facecolor='none',edgecolor='dodgerblue',linewidth=1,zorder=3)
#      ax.add_feature(shp_feat1)
   if ("UNITED STATES" in record.attributes['COUNTRY1'].upper() or "UNITED STATES" in record.attributes['COUNTRY2'].upper()) and \
      ("MEXICO" in record.attributes['COUNTRY1'].upper() or "MEXICO" in record.attributes['COUNTRY2'].upper()):
      border_lons,border_lats = record.geometry.xy
      mexico_border_buffer_poly = shapely.buffer(record.geometry,distance=0.9,single_sided=False)
#      buffer_buffer_m = shapely.buffer(record.geometry,distance=0.1,single_sided=False)
#      mbbp = shapely.union(mexico_border_buffer_poly,buffer_buffer_m)
#      ax.add_geometries([record.geometry],crs=projPC,facecolor='none',edgecolor='darkorchid',linewidth=1)
#      ax.add_geometries(mexico_border_buffer_poly,crs=projPC,facecolor='none',edgecolor='deeppink',linewidth=1,zorder=3)
geom_difference_canada = shapely.difference(conus_geom,canada_border_buffer_poly)
geom_difference_mexico = shapely.difference(conus_geom,mexico_border_buffer_poly)

# Add Olympic peninsula
olympia_data = np.loadtxt("Olympic_peninsula_boundary.dat")
olympia_data.astype(float)
olympia_data[:,[1,0]] = olympia_data[:,[0,1]]
Olympic_peninsula_poly = shapely.Polygon(olympia_data)
Olympic_peninsula_geom = shapely.intersection(Olympic_peninsula_poly,geom_difference_mexico)

# Construct final mask
piece3 = shapely.intersection(union_lake_geom,geom_difference_mexico)
piece4 = shapely.intersection(geom_difference_canada,geom_difference_mexico)
# FINAL MASK
mask_geom = shapely.union_all([piece3,piece4,Olympic_peninsula_geom])
# FINAL MASK
# Plot parts of the mask for diagnostics
ax.add_geometries(union_lake_geom,crs=projPC,facecolor='none',edgecolor='salmon',linewidth=0.5,zorder=2)
ax.add_geometries(geom_difference_canada,crs=projPC,facecolor='none',edgecolor='orange',linewidth=0.5,zorder=2)
ax.add_geometries(geom_difference_mexico,crs=projPC,facecolor='none',edgecolor='lime',linewidth=0.5,zorder=2)
ax.add_geometries(mask_geom,crs=projPC,facecolor='lightsteelblue',edgecolor='none',linewidth=1,zorder=1)
ax.set_extent(map_extent,crs=projPC)
plt.savefig("masked_CONUS_radar_domain.png",dpi=200)
plt.close()

# Mask MRMS grid to CONUS mask constructed above...no radars can be placed outside of this masked area
conus_mask = shapely.contains_xy(mask_geom,mrms_lons,mrms_lats)

fig_x = 10
AR = (map_extent[3]-map_extent[2])/(map_extent[1]-map_extent[0])
fig_y = AR*fig_x+1.5

sampler = qmc.LatinHypercube(d=2,optimization='random-cd')
sample = sampler.random(n=n_radars)
radar_coords_first_guess = qmc.scale(sample,l_bounds=[x1,y1],u_bounds=[x2,y2])
print(f"Initial radar network discrepancy: {qmc.discrepancy(sample)}")
radar_coords = radar_coords_first_guess.copy()

# Compute spacing between all pairs of first-guess radars
network_spacing = np.full((n_radars,n_radars),np.nan,dtype=float)
for a in range(n_radars):
 for b in range(n_radars):
  network_spacing[a,b] = gc1(radar_coords_first_guess[a,0],radar_coords_first_guess[a,1],radar_coords_first_guess[b,0],radar_coords_first_guess[b,1])
np.fill_diagonal(network_spacing,np.inf)
# For each radar determine the distance to its nearest neighbor as well as which radar that is
closest_radar = np.min(network_spacing,axis=1)
closest_radar_b = np.argmin(network_spacing,axis=1)
# Setup 2D array of distances from each grid point to nearest radar
network_distances = np.full(mrms_lons.shape,1e7,dtype=float)
distances_masked = np.ma.masked_where(~conus_mask,network_distances)
# Loop through radars and eliminate radars that are either outside the CONUS (mask) or are too close to others
not_in_domain = []
radar_too_close = []
timea = default_timer()
for a in range(n_radars):
 if not shapely.contains_xy(conus_geom,x=radar_coords_first_guess[a,0],y=radar_coords_first_guess[a,1]):
  not_in_domain.append(a) 
 elif closest_radar[a] < minimum_radar_separation:
  B = closest_radar_b[a]
  print(f"Radar ID {a} is too close to a neighbor (radar {B}: {closest_radar[a]:.0f} m)")
  radar_too_close.append(a)
  # Update/re-calculate network spacing array assuming this radar is no longer present 
  network_spacing[a,B] = np.inf
  closest_radar = np.min(network_spacing,axis=1)
  closest_radar_b = np.argmin(network_spacing,axis=1)
  crb = np.min(network_spacing,axis=0)
  print(f"Removing this radar now means the closest radar to radar ID {B} is now {crb[B]:.0f} m")
#  distances = gc2d(mrms_lons,mrms_lats,radar_coords_first_guess[a,0],radar_coords_first_guess[a,1])
#  network_distances = np.minimum(network_distances,distances)
#  distances_masked = np.ma.minimum(distances_masked,distances)
#  distances_masked.mask = ~conus_mask
  # Old method not used right now
  if False:
   i = 1
   while True:
    # Pick a new location until it is within the domain
    new_lat = CONUS_bounds[1] + (CONUS_bounds[3]-CONUS_bounds[1])*rng.random(size=1)
    new_lon = CONUS_bounds[0] + (CONUS_bounds[2]-CONUS_bounds[0])*rng.random(size=1)
    if shapely.contains_xy(conus_geom,x=new_lon,y=new_lat):
     break
    i += 1
   radar_coords[a,0] = new_lon[0]
   radar_coords[a,1] = new_lat[0]
   print(f"{a:3d}: Radar site at {radar_coords_first_guess[a,0]:.3f},{radar_coords_first_guess[a,1]:.3f}, in CONUS = {shapely.contains_xy(conus_geom,x=radar_coords_first_guess[a,0],y=radar_coords_first_guess[a,1])} has been changed to {radar_coords[a,0]:.3f},{radar_coords[a,1]:.3f}...in CONUS = {shapely.contains_xy(conus_geom,x=radar_coords[a,0],y=radar_coords[a,1])}. It took {i} tries.")
   # End not used old stuff
 else:
  distances = gc2d(mrms_lons,mrms_lats,radar_coords_first_guess[a,0],radar_coords_first_guess[a,1])
  network_distances = np.minimum(network_distances,distances)
  distances_masked = np.ma.minimum(distances_masked,distances)
  distances_masked.mask = ~conus_mask
timeb = default_timer()
print(f"Time to calculate initial distance array: {timeb-timea:.2f} s")

distance_levs = np.arange(25.,401.,25.)
plt.figure(figsize=(fig_x,fig_y))
ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=projPC)
ax.add_feature(feature.STATES.with_scale('10m'),linewidth=0.5,edgecolor='0.5')
ax.add_feature(feature.BORDERS.with_scale('50m'),linewidth=1,edgecolor='black')
ct = ax.contourf(mrms_lons,mrms_lats,network_distances/1e3,levels=distance_levs,cmap='gnuplot_r',transform=projPC)
cb = plt.colorbar(mappable=ct,orientation='horizontal',fraction=0.1,aspect=40,pad=0.01,shrink=0.5,ticks=distance_levs,extend='both')
cb.set_label("Distance to closest radar [km]",fontsize=8)
cb.ax.tick_params(labelsize=6)
for a in range(n_radars):
   if a in not_in_domain:
      ax.plot(radar_coords_first_guess[a,0],radar_coords_first_guess[a,1],'X',ms=6,mew=0.0,color='red',transform = projPC)
   elif a in radar_too_close:
      ax.plot(radar_coords_first_guess[a,0],radar_coords_first_guess[a,1],'x',ms=6,color='skyblue',transform = projPC)
   else:
      ax.plot(radar_coords_first_guess[a,0],radar_coords_first_guess[a,1],'.',ms=4,color='navy',transform = projPC)
#ax.add_geometries([conus_geom],crs=projPC,facecolor='none',edgecolor='lime')
ax.set_extent(map_extent)
#plt.figtext(0.99,0.99,f"discrepancy = {qmc.discrepancy(sample):.4e}",fontsize=8,fontweight=300,ha='right',va='top',bbox={'facecolor':'white','edgecolor':'black'})
plt.savefig(f"radar_network_diag_000.png",dpi=200)
plt.close()

# Replace first guess radars either outside the CONUS masking domain or that are too close to others
for i,A in enumerate(not_in_domain + radar_too_close):
   timea = default_timer()

   if A in not_in_domain:
      print(f"Radar site #{A} was not within the CONUS boundary")
   if A in radar_too_close:
      print(f"Radar site #{A} was too close to another radar")
   # Current method replaces radars outside of CONUS at the location of the maximum distance from existing radars that IS NOT OVER WATER OR IN STEEP TERRAIN
   if True:
    sorted_idx = np.ma.argsort(distances_masked.flatten(),endwith=False)[::-1]
    for I in range(len(sorted_idx)):
      idx_x,idx_y = np.unravel_index(sorted_idx[I],distances_masked.shape)
      if not distances_masked.mask[idx_x,idx_y]:
#         print(I,len(sorted_idx),sorted_idx[I],idx_x,idx_y,distances_masked[idx_x,idx_y])
         # Find out if this gridpoint is on too much a slope or over a body of water
         if landmask[idx_x,idx_y] == 0 or terrain_slope[idx_x,idx_y] >= 5.0:
          if landmask[idx_x,idx_y] == 0:
            print(f"Cannot place radar at this location ({mrms_lons[idx_x,idx_y]:.3f},{mrms_lats[idx_x,idx_y]:.3f}) since it is over water (landmask = {landmask[idx_x,idx_y]})")
            continue
          elif terrain_slope[idx_x,idx_y] >= 5.0:
            print(f"Cannot place radar at this location ({mrms_lons[idx_x,idx_y]:.3f},{mrms_lats[idx_x,idx_y]:.3f}) since it is on steep terrain with slope {terrain_slope[idx_x,idx_y]:.2f})")
            continue
         else:
          break
    if I == len(sorted_idx)-1:
       print("Problem: couldn't find a new place to put a radar")
       exit(1)
    print(f"{A:3d}: first guess location: {radar_coords_first_guess[A,0]:.3f} {radar_coords_first_guess[A,1]:.3f} (mask distance: {distances_masked[idx_x,idx_y]:.0f} m) new location: {mrms_lons[idx_x,idx_y]:.3f} {mrms_lats[idx_x,idx_y]:.3f}")
    radar_coords[A,0] = mrms_lons[idx_x,idx_y]
    radar_coords[A,1] = mrms_lats[idx_x,idx_y]
    distances = gc2d(mrms_lons,mrms_lats,radar_coords[A,0],radar_coords[A,1])
    distances_masked = np.ma.minimum(distances_masked,distances)
    distances_masked.mask = ~conus_mask

   # Previous method just replaces radars outside of CONUS at the location of the maximum distance from existing radars
   if False:
    idx = np.ma.argmax(distances_masked)
    max_x,max_y = np.unravel_index(idx,distances_masked.shape)
    print(f"{A:3d}: first guess location: {radar_coords_first_guess[A,0]:.3f} {radar_coords_first_guess[A,1]:.3f} (mask distance: {distances_masked[max_x,max_y]:.0f} m) new location: {mrms_lons[max_x,max_y]:.3f} {mrms_lats[max_x,max_y]:.3f}")
    radar_coords[A,0] = mrms_lons[max_x,max_y]
    radar_coords[A,1] = mrms_lats[max_x,max_y]
    distances = gc2d(mrms_lons,mrms_lats,radar_coords[A,0],radar_coords[A,1])
    distances_masked = np.ma.minimum(distances_masked,distances)
    distances_masked.mask = ~conus_mask
   timeb = default_timer()
   print(f"Processing this radar took {timeb-timea:.2f} s")

   distance_levs = np.arange(25.,401.,25.)
   plt.figure(figsize=(fig_x,fig_y))
   ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=projPC)
   ax.add_feature(feature.STATES.with_scale('10m'),linewidth=0.5,edgecolor='0.5')
   ax.add_feature(feature.BORDERS.with_scale('50m'),linewidth=1,edgecolor='black')
   ct = ax.contourf(mrms_lons,mrms_lats,distances_masked/1e3,levels=distance_levs,cmap='gnuplot_r',transform=projPC)
   cb = plt.colorbar(mappable=ct,orientation='horizontal',fraction=0.1,aspect=40,pad=0.01,shrink=0.5,ticks=distance_levs,extend='both')
   cb.set_label("Distance to closest radar [km]",fontsize=8)
   cb.ax.tick_params(labelsize=6)
   for a in range(n_radars):
      if a in not_in_domain:
         ax.plot(radar_coords_first_guess[a,0],radar_coords_first_guess[a,1],'X',ms=6,mew=0.0,color='red',transform = projPC)
      elif a in radar_too_close:
         ax.plot(radar_coords_first_guess[a,0],radar_coords_first_guess[a,1],'x',ms=6,color='black',transform = projPC)
         if a <= A:
            ax.plot(radar_coords[a,0],radar_coords[a,1],'+',ms=4,color='saddlebrown',transform = projPC)
      else:
         ax.plot(radar_coords_first_guess[a,0],radar_coords_first_guess[a,1],'.',ms=4,color='navy',transform = projPC)
      if a == A:
         ax.plot(radar_coords[a,0],radar_coords[a,1],'+',ms=6,color='black',transform=projPC)
         ax.plot(radar_coords[a,0],radar_coords[a,1],'o',ms=8,mec='black',mfc='none',mew=1,transform=projPC)
#   ax.add_geometries([conus_geom],crs=projPC,facecolor='none',edgecolor='lime')
   ax.set_extent(map_extent)
#   plt.figtext(0.99,0.99,f"discrepancy = {qmc.discrepancy(sample):.4e}",fontsize=8,fontweight=300,ha='right',va='top',bbox={'facecolor':'white','edgecolor':'black'})
   plt.savefig(f"radar_network_diag_{i+1:03d}.png",dpi=200)
   plt.close()

#print(f"After adjustment, discrepancy is now {qmc.discrepancy(qmc.scale(radar_coords,l_bounds=[x1,y1],u_bounds=[x2,y2],reverse=True))}")
plt.figure(figsize=(fig_x,fig_y))
ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=projPC)
ax.add_feature(feature.STATES.with_scale('10m'),linewidth=0.5,edgecolor='0.5')
ax.add_feature(feature.BORDERS.with_scale('50m'),linewidth=1,edgecolor='black')
ax.plot(radar_coords_first_guess[:,0],radar_coords_first_guess[:,1],'.',ms=4,color='navy',transform = projPC)
for a in range(n_radars):
   if a in not_in_domain:
      ax.plot(radar_coords_first_guess[a,0],radar_coords_first_guess[a,1],'X',ms=6,mew=0.0,color='red',transform = projPC)
   elif a in radar_too_close:
      ax.plot(radar_coords_first_guess[a,0],radar_coords_first_guess[a,1],'x',ms=6,color='black',transform = projPC)
      ax.plot(radar_coords[a,0],radar_coords[a,1],'+',ms=4,color='saddlebrown',transform = projPC)
   else:
      ax.plot(radar_coords_first_guess[a,0],radar_coords_first_guess[a,1],'.',ms=4,color='navy',transform = projPC)
ax.add_geometries([conus_geom],crs=projPC,facecolor='none',edgecolor='lime')
ax.set_extent(map_extent)
#plt.figtext(0.99,0.99,f"discrepancy = {qmc.discrepancy(sample):.4e}",fontsize=8,fontweight=300,ha='right',va='top',bbox={'facecolor':'white','edgecolor':'black'})
p = 0
while True:
   img_file = f"random_radar_network_{p}.png"
   if isfile(img_file):
      p += 1
   else:
     break
plt.savefig(img_file,dpi=200)
plt.close()

fl = open(f"radar_network_locations_{p}.txt",'w')
for a in range(n_radars):
   fl.writelines(f"RADAR{a:03d} {radar_coords[a,0]:.5f}   {radar_coords[a,1]:.5f}\n")
fl.close()
print("Complete")
