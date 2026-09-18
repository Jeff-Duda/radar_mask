from cartopy import crs
from cartopy import feature
from matplotlib import use, colormaps
use('agg')
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import shapely
from cartopy.io.shapereader import Reader
from geopy import distance
from math import sqrt, radians, cos, sin, acos
from sys import exit

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
plt.figure(figsize=(10,5))
ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=projPC)
ax.add_feature(feature.STATES.with_scale('10m'),linewidth=0.5,edgecolor='0.5')
ax.add_feature(feature.BORDERS.with_scale('50m'),linewidth=1,edgecolor='black')
ax.add_geometries([conus_geom],crs=projPC,facecolor='none',edgecolor='black',linewidth=1,zorder=1)
i = 0
j = 0
for record in reader.records():
#   if "pacific" in record.attributes['NAME'].lower():
#      print(record.attributes['NAME'],record.geometry.length)
#      j += 1
#      pacific_coast_geom = shapely.buffer(record.geometry,distance=1)
#      if j == 1:
#         union_pacific_coast_geom = pacific_coast_geom
#      else:
#         union_pacific_coast_geom = shapely.union(union_pacific_coast_geom,pacific_coast_geom)
   if "great lakes" in record.attributes['NAME'].lower():
#    if record.geometry.length >= 0.5:
      i += 1
      lake_geom = shapely.buffer(record.geometry,distance=1)
      if i == 1:
         union_lake_geom = lake_geom
      else:
         union_lake_geom = shapely.union(union_lake_geom,lake_geom)
#      lake_geom = shapely.buffer(union_lake_geom,distance=1.0)
#      ax.add_geometries([record.geometry],crs=projPC,facecolor='none',edgecolor='salmon',linewidth=0.5,zorder=10)
#      print(f"Great Lakes shapefile {i} has length {record.geometry.length:.4f}")
#ax.add_geometries(union_lake_geom,crs=projPC,facecolor='none',edgecolor='salmon',linewidth=0.5,zorder=2)
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
ax.add_geometries(geom_difference_canada,crs=projPC,facecolor='none',edgecolor='orange',linewidth=0.5,zorder=2)
ax.add_geometries(geom_difference_mexico,crs=projPC,facecolor='none',edgecolor='lime',linewidth=0.5,zorder=2)
ax.add_geometries(mask_geom,crs=projPC,facecolor='lightsteelblue',edgecolor='none',linewidth=1,zorder=1)
#ax.set_extent([-94,-74.0,40,50],crs=projPC)
ax.set_extent(map_extent,crs=projPC)
#plt.savefig("Great_Lakes_mask_diag_0p5.png",dpi=200)
plt.savefig("masked_CONUS_radar_domain.png",dpi=200)
plt.close()

