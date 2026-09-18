from cartopy import config
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from matplotlib import use, colormaps
use('agg')
#from scipy.signal import convolve2d
from astropy.convolution import convolve, convolve_fft
import grib2io as grb2
import matplotlib.pyplot as plt
from geopy import distance
from datetime import datetime,timedelta
import numpy as np
from netCDF4 import Dataset
from math import degrees, radians, pi, sqrt
from timeit import default_timer
from os.path import isfile

config['data_dir'] = "/work/noaa/wrfruc/jdduda/cartopy_shapefiles"

time1 = default_timer()
mask_file = "/work/noaa/wrfruc/jdduda/radar_mask/radar_mask_without_refractivity.nc"
nci = Dataset(mask_file,'r')
slant_range = nci.variables['slant_range'][:]
slant_range[slant_range > 1e6] = np.nan
distance_3D = nci.variables['distance_to_closest_radar_3D'][:]
nci.close()

start_time = "20220430-1230"
end_time = "20220430-1230"
stdt = datetime.strptime(start_time,'%Y%m%d-%H%M')
etdt = datetime.strptime(end_time,'%Y%m%d-%H%M')
root = "/work/noaa/wrfruc/jdduda/radar_mask/"
MRMS_dir = f"/work2/noaa/wrfruc/Ruifang.Li/RadarNext_OSSE/Data/refl_simulated"
prod_name = "MergedReflectivityQC"

data = np.loadtxt("/home/jduda/radar_colors_RGB",skiprows=1)
dBZ_levs = data[:,0]
dBZ_colors = data[:,1:]/255.
dBZ_colors = np.vstack(([1,1,1],dBZ_colors))
# Two values for "no signal" and "data outside domain of radars": -99 and -999, respectively
no_data_levs = [-900.,-90.]
no_data_cols = ['0.67','peachpuff','none']

maj_axis = distance.ELLIPSOIDS['WGS-84'][1]
min_axis = distance.ELLIPSOIDS['WGS-84'][0]
Re = sqrt(maj_axis*min_axis)*1000.
MRMS_heights = [0.50,0.75,1.0,1.25,1.50,1.75,2.0,2.25,2.50,2.75,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,10.,11.,12.,13.,14.,15.,16.,17.,18.,19] # [km]

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
smoothed_reflectivity = np.zeros((len(MRMS_heights),grid_ny,grid_nx),dtype=float)

terrain_file = "/work/noaa/wrfruc/jdduda/NR_terrain_MRMS_grid.nc"
nc = Dataset(terrain_file,'r')
terrain_2D = nc.variables['HGT_M'][:][0,:,:]
nc.close()

linear_beam_width = radians(1.0)*slant_range
dx = Re*np.cos(grid_lats_r)*radians(0.01)
dy = np.full_like(grid_lats_r,Re*radians(0.01))
print(f"dy = {Re*radians(0.01):.1f} m")
mean_dx = np.sqrt(dx*dy)
beam_grid_size_ratio = linear_beam_width / mean_dx
time2 = default_timer()
print(f"Time to process data was {time2-time1:.1f} s")

proj = ccrs.PlateCarree()

lat2 = 50
lat1 = 43
lon1 = -104
lon2 = -96
#lat2 = 53
#lat1 = 22.5
#lon1 = -130
#lon2 = -60
cb_fract = 0.1
fig_x = 10
AR = (lon2-lon1)/(lat2-lat1)
fig_y = fig_x/(AR*(1-cb_fract))

if False:
 plt.figure(figsize=(fig_x,fig_y))
 ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=proj)
 ax.set_extent([lon1,lon2,lat1,lat2])
 ax.add_feature(cfeature.STATES.with_scale('10m'),linewidth=0.5,edgecolor='0.5')
 ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidth=1,edgecolor='black')
 pc = ax.pcolormesh(grid_lons,grid_lats,dx,cmap='cividis_r',shading='nearest')
 cb = plt.colorbar(mappable=pc,orientation='horizontal',pad=0.01,fraction=cb_fract,aspect=40,shrink=0.5)
 cb.set_label('dx (not dy) [m]',fontsize=8)
 cb.ax.tick_params(labelsize=6)
 plt.savefig("MRMS_dx.png",dpi=150)
 plt.close()

 ratio_levels = np.arange(1.0,9.1,1.0)
 cmap = colormaps['gnuplot_r']
 colors = cmap(np.linspace(0,1,len(ratio_levels)+1))

 width_levels = np.arange(0,8000.1,1000.)
 cmap = colormaps['rainbow']
 width_colors = cmap(np.linspace(0,1,len(width_levels)+1))

 for z in range(len(MRMS_heights)):
  print(MRMS_heights[z],np.nanmin(beam_grid_size_ratio[z,:,:]),np.nanmax(beam_grid_size_ratio[z,:,:]),np.nanmean(beam_grid_size_ratio[z,:,:]),np.count_nonzero(~np.isnan(beam_grid_size_ratio[z,:,:])))
  plt.figure(figsize=(fig_x,fig_y))
  ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=proj)
  ax.set_extent([lon1,lon2,lat1,lat2])
  ax.add_feature(cfeature.STATES.with_scale('10m'),linewidth=0.5,edgecolor='0.5')
  ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidth=1,edgecolor='black')
 # pc = ax.pcolormesh(grid_lons,grid_lats,linear_beam_width[z,:,:],cmap='inferno_r',shading='nearest')
  pc = ax.contourf(grid_lons,grid_lats,linear_beam_width[z,:,:],levels=width_levels,colors=width_colors,extend='both')
  cb = plt.colorbar(mappable=pc,orientation='horizontal',pad=0.01,fraction=cb_fract,aspect=40,shrink=0.5,ticks=width_levels)
  cb.set_label('linear width of 1.0 deg. radar beam [m]',fontsize=8)
  cb.ax.tick_params(labelsize=6)
  plt.figtext(0.01,0.01,f"Height: {1e3*MRMS_heights[z]:.0f} m",ha='left',va='bottom',fontsize=10,fontweight=500,bbox=dict(facecolor='white',edgecolor='black',pad=2))
  plt.savefig(f"beam_size_z{1e3*MRMS_heights[z]:.0f}m.png",dpi=150)
  plt.close()

  plt.figure(figsize=(fig_x,fig_y))
  ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=proj)
  ax.set_extent([lon1,lon2,lat1,lat2])
  ax.add_feature(cfeature.STATES.with_scale('10m'),linewidth=0.5,edgecolor='0.5')
  ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidth=1,edgecolor='black')
 # pc = ax.pcolormesh(grid_lons,grid_lats,beam_grid_size_ratio[z,:,:],cmap='cividis_r',shading='nearest')
  pc = ax.contourf(grid_lons,grid_lats,beam_grid_size_ratio[z,:,:],levels=np.arange(1.0,9.1,1.0),colors=colors,extend='both')
  cb = plt.colorbar(mappable=pc,orientation='horizontal',pad=0.01,fraction=cb_fract,aspect=40,shrink=0.5,ticks=ratio_levels)
 # cb.set_ticks(np.arange(1.0,8.1,0.5))
  cb.set_label('beam-width-to-grid-size ratio [-]',fontsize=8)
  cb.ax.tick_params(labelsize=6)
  plt.figtext(0.01,0.01,f"Height: {1e3*MRMS_heights[z]:.0f} m",ha='left',va='bottom',fontsize=10,fontweight=500,bbox=dict(facecolor='white',edgecolor='black',pad=2))
  plt.savefig(f"beam_grid_size_ratio_z{1e3*MRMS_heights[z]:.0f}m.png",dpi=150)
  plt.close()

time = stdt
while time <= etdt:
  time0 = default_timer()
  MRMS_tstr = time.strftime("%Y%m%d-%H%M00")
  print(f"Loop time is now {MRMS_tstr}")
  for z,elev in enumerate(MRMS_heights):
   time1 = default_timer()
   dir = f"{MRMS_dir}/{time.strftime('%Y%m%d-%H%M')}"
   # File name = MergedReflectivityQC_07.50_20220430-123000.grib2
   file = f"{dir}/{prod_name}_{elev:05.2f}_{MRMS_tstr}.grib2"
   print(file)
   if isfile(file):
    grb = grb2.open(file)
    rec = grb[0]
    input_reflectivity = np.flipud(rec.data)
    grb.close()

    if False:
     plt.figure(figsize=(fig_x,fig_y))
     ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=proj)
     ax.set_extent([lon1,lon2,lat1,lat2],crs=proj)
     ax.coastlines(linewidth=0.5,resolution='50m')
     ax.add_feature(cfeature.BORDERS,edgecolor='black',linewidth=1,zorder=3)
     ax.add_feature(cfeature.STATES,edgecolor='0.5',linewidth=0.5,zorder=2)
     cp = ax.contourf(grid_lons,grid_lats,input_reflectivity,colors=dBZ_colors,levels=dBZ_levs,extend='both',transform=proj,transform_first=True)
     cb = plt.colorbar(mappable=cp,orientation='horizontal',fraction=cb_fract,shrink=0.5,aspect=40,pad=0.02,ticks=dBZ_levs)
     cb.set_label(f"Reflectivity at {1e3*elev:.0f} m ASL",fontsize=8)
     cb.ax.tick_params(labelsize=6)
     plt.figtext(0.01,0.01,f"Valid time: {MRMS_tstr}",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='monospace')
 #    plt.figtext(0.01,0.01,f"Height: {1e3*elev:.0f} m ASL",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='serif')
     plt.savefig(f"{root}/zoom1_nature_run_masked_reflectivity_{1e3*elev:05.0f}m_{MRMS_tstr}.png",dpi=250)
     plt.close()

    input_reflectivity = np.where(input_reflectivity < -10.,np.nan,input_reflectivity)
    smoothed_reflectivity[z,:,:] = np.where(beam_grid_size_ratio[z,:,:] <= 2.0,input_reflectivity,-99)
    # Perform smoothing based on beam_grid_size_ratio
    for size_bin in np.arange(2.0,9.1,1.0):
       timea = default_timer()
       sz = int(size_bin)
       idxs = np.argwhere((beam_grid_size_ratio[z,:,:] >= size_bin) & (beam_grid_size_ratio[z,:,:] <= size_bin + 1.0))
       rows = idxs[:,0]
       cols = idxs[:,1]
       oned = np.arange(2*sz+1)
       boxx,boxy = np.meshgrid(oned,oned)
       distance = np.sqrt((boxx-sz)**2 + (boxy-sz)**2)
       mask = np.where(distance <= sz,1.0,0.0)
       mask /= np.sum(mask)
       bin_convolved_refl = convolve_fft(input_reflectivity,mask,boundary='fill',nan_treatment='interpolate',preserve_nan=True,fill_value=0.)
#       bin_convolved_refl = convolve2d(input_reflectivity,mask,mode='same',boundary='fill',fillvalue=0.)
       if True:
        plt.figure(figsize=(fig_x,fig_y))
        ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=proj)
        ax.set_extent([lon1,lon2,lat1,lat2],crs=proj)
        ax.coastlines(linewidth=0.5,resolution='50m')
        ax.add_feature(cfeature.BORDERS,edgecolor='black',linewidth=1,zorder=3)
        ax.add_feature(cfeature.STATES,edgecolor='0.5',linewidth=0.5,zorder=2)
        cp = ax.contourf(grid_lons,grid_lats,bin_convolved_refl,colors=dBZ_colors,levels=dBZ_levs,extend='both',transform=proj,transform_first=True)
        ct = ax.contour(grid_lons,grid_lats,beam_grid_size_ratio[z,:,:],levels=np.arange(2.0,9.1,1.0),colors=[1.0,0.5,0.5],linewidths=0.5)
        clbls = ax.clabel(CS=ct,levels=ct.levels,colors='red',fontsize=6,fmt='%.0f')
        for cl in clbls:
         cl.set_rotation(0)
        cb = plt.colorbar(mappable=cp,orientation='horizontal',fraction=cb_fract,shrink=0.5,aspect=40,pad=0.02,ticks=dBZ_levs)
        cb.set_label(f"Reflectivity at {1e3*elev:.0f} m ASL",fontsize=8)
        cb.ax.tick_params(labelsize=6)
        plt.figtext(0.01,0.01,f"Convolution radius: {sz} grid squares",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='serif')
        plt.savefig(f"{root}/zoom_{sz:02d}-convolved_reflectivity_{1e3*elev:05.0f}m_{MRMS_tstr}.png",dpi=250)
        plt.close()
       smoothed_reflectivity[z,rows,cols] = bin_convolved_refl[rows,cols]
       timeb = default_timer()
       print(f"Time to partially make smoothed reflectivity array for bin size {sz} was {timeb-timea:.1f} s")

    smoothed_reflectivity[z,:,:] = np.where(np.isnan(input_reflectivity),-10.0,smoothed_reflectivity[z,:,:])
    plt.figure(figsize=(fig_x,fig_y))
    ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=proj)
    ax.set_extent([lon1,lon2,lat1,lat2],crs=proj)
    ax.coastlines(linewidth=0.5,resolution='50m')
    ax.add_feature(cfeature.BORDERS,edgecolor='black',linewidth=1,zorder=3)
    ax.add_feature(cfeature.STATES,edgecolor='0.5',linewidth=0.5,zorder=2)
    cp = ax.contourf(grid_lons,grid_lats,smoothed_reflectivity[z,:,:],colors=dBZ_colors,levels=dBZ_levs,extend='both',transform=proj,transform_first=True)
    ct = ax.contour(grid_lons,grid_lats,beam_grid_size_ratio[z,:,:],levels=np.arange(2.0,9.1,1.0),colors=[1.0,0.5,0.5],linewidths=0.5)
    clbls = ax.clabel(CS=ct,levels=ct.levels,colors='red',fontsize=6,fmt='%.0f')
    for cl in clbls:
     cl.set_rotation(0)
    cb = plt.colorbar(mappable=cp,orientation='horizontal',fraction=cb_fract,shrink=0.5,aspect=40,pad=0.02,ticks=dBZ_levs)
    cb.set_label(f"Reflectivity at {1e3*elev:.0f} m ASL",fontsize=8)
    cb.ax.tick_params(labelsize=6)
    plt.figtext(0.01,0.01,f"Valid time: {MRMS_tstr}",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='monospace')
#    plt.figtext(0.01,0.01,f"Height: {1e3*elev:.0f} m ASL",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='serif')
    plt.savefig(f"{root}/zoom1_nature_run_smoothed_reflectivity_{1e3*elev:05.0f}m_{MRMS_tstr}.png",dpi=250)
    plt.close()

    # Create output GRIB2 message and write file
    section2 = b"Creation note: Smoothed reflectivity based on ratio between beam width and grid size"
    out_msg = grb2.Grib2Message(section0=rec.section0,section1=rec.section1,section2=section2,section3=rec.section3,section4=rec.section4,section5=rec.section5)
    out_msg.data = smoothed_reflectivity[z,:,:]
    out_msg.pack()

    print(out_msg._msg[0:4],out_msg._msg[-4:])

    grb_new = grb2.open(f"{prod_name}_{elev:05.2f}_{MRMS_tstr}_smoothed.grib2",'w')
    grb_new.write(out_msg)
    grb_new.close()

    time2 = default_timer()
    print(f"Time to completely process this vertical level: {time2-time1:.1f} s")
   # endif file exists
  # enddo heights

  time += timedelta(minutes=15)
  time00 = default_timer()
  print(f"Time to completely process this output time: {time00-time0:.1f} s")
# End time loop
