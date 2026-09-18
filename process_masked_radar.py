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
from math import degrees, radians, pi, sqrt, exp
from timeit import default_timer
from os.path import isfile
import pandas as pd

config['data_dir'] = "/work/noaa/wrfruc/jdduda/cartopy_shapefiles"

time1 = default_timer()
mask_file = "/work/noaa/wrfruc/jdduda/radar_mask/radar_mask_MRMS.nc"
nci = Dataset(mask_file,'r')
radar_mask = nci.variables['mask'][:]
slant_range = nci.variables['slant_range'][:]
slant_range[slant_range > 1e6] = np.nan
elevation_angle_3D = nci.variables['elevation_angle'][:]
azimuths = nci.variables['azimuth_to_closest_radar'][:]
wavelength_band = nci.variables['wavelength_band_num'][:]
nearest_ID = nci.variables['illuminated_radar_ID'][:]
max_radar_ID = np.max(nearest_ID)
nci.close()

US_radar_file = "/work/noaa/wrfruc/jdduda/radar_mask/nexrad-stations-valid-2026.txt"
radar_df = pd.read_table(US_radar_file,sep="\\s+",header=0,skiprows=[1],index_col='ICAO')
print(radar_df)
n_US_radars = len(radar_df)

start_time = "20220505-1800"
end_time = "20220505-1800"
stdt = datetime.strptime(start_time,'%Y%m%d-%H%M')
etdt = datetime.strptime(end_time,'%Y%m%d-%H%M')
root = "/work/noaa/wrfruc/jdduda/radar_mask/"
MRMS_dir = "/work2/noaa/wrfruc/Ruifang.Li/RadarNext_OSSE/Data/refl_simulated"
prod_name = "MergedReflectivityQC"

def attenuate_gas(elevation_angle,range,band):
   # Inputs and units:
   # -elevation_angle [degrees]
   # -range [km]
   # -wavelength band [-]
   scale = np.where(band == 1,1.0,0.)
   scale = np.where(band == 2,1.2,scale)
   scale = np.where(band == 3,1.5,scale)
#   if band == "S":
#       scale = 1.0
#   elif band == "C":
#       scale = 1.2
#   elif band == "X":
#       scale == 1.5
   atten = scale*(0.4 + 3.45*np.exp(-elev/1.8))* \
           (1-np.exp(-range/(27.8+154*np.exp(-elev/2.2))))
   return atten # [dB]

def calc_rain_attenuation(rain_rate,band,temperature=18):
   # Inputs and units:
   # -rainfall rate ([mm/hr] - should be the direct output from R_Z)
   # -Wavelength band [numerical]
   # -temp refers to temperature in deg. C
   scale_factor = 2*np.exp(-0.035*temperature)

   atten = np.where(band == 1,0.000343*rain_rate**0.97,0.)
   atten = np.where(band == 2,0.0018*rain_rate**1.05,atten)
   atten = np.where(band == 3,0.01*rain_rate**1.21,atten)

   return atten # [db/km]

def R_Z(refl,config=2):
   # Convert reflectivity to rainfall rate
   # config is an option to select the specific power law coefficients
   if config == 1: # stratiform rate from Marshall/Palmer
    A = 200
    b = 1.6
   elif config == 2: # Corresponds to the rain attenuation function
    A = 400
    b = 1.4
   elif config == 3:
    A = 300
    b = 1.5
   else:
    print("config must be in [1,2,3]. Setting to 1")
    A = 200
    b = 1.6
   # input reflectivity must be converted from logarithmic units to linear units
   refl_lin = 10**(refl/10)
   rainfall = refl_lin/A + exp(1./b)
   return rainfall # [mm/hr]

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

# Full domain view
lat2 = 53
lat1 = 22.5
lon1 = -130
lon2 = -60
# Zoomed in view
# Texas
lat1 = 28
lat2 = 35
lon1 = -100.5
lon2 = -93.5
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
  smoothed_reflectivity = np.zeros((len(MRMS_heights),grid_ny,grid_nx),dtype=float)
  processed_reflectivity = np.zeros((len(MRMS_heights),grid_ny,grid_nx),dtype=float)
  for z,elev in enumerate(MRMS_heights):
   time1 = default_timer()
   radars_attenuated_lats = []
   radars_attenuated_lons = []
   dir = f"{MRMS_dir}/{time.strftime('%Y%m%d-%H%M')}"
   # File name = MergedReflectivityQC_07.50_20220430-123000.grib2
   file = f"{dir}/{prod_name}_{elev:05.2f}_{MRMS_tstr}.grib2"
   print(file)
   if isfile(file):
    grb = grb2.open(file)
    rec = grb[0]
    input_reflectivity = np.flipud(rec.data)
    grb.close()

    input_reflectivity = np.where(input_reflectivity < -10.,np.nan,input_reflectivity)
    smoothed_reflectivity[z,:,:] = np.where(beam_grid_size_ratio[z,:,:] <= 2.0,input_reflectivity,-99)
    # Perform smoothing based on beam_grid_size_ratio
    timea = default_timer()
    for size_bin in np.arange(2.0,9.1,1.0):
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
       smoothed_reflectivity[z,rows,cols] = bin_convolved_refl[rows,cols]
    timeb = default_timer()
    print(f"Time to smooth reflectivity array was {timeb-timea:.2f} s")

    smoothed_reflectivity[z,:,:] = np.where(np.isnan(input_reflectivity),-10.0,smoothed_reflectivity[z,:,:])
#    smoothed_reflectivity[z,:,:] = np.where(radar_mask[z,:,:] < 1,np.nan,radar_mask[z,:,:])

    gas_attenuation = attenuate_gas(elevation_angle_3D[z,:,:],slant_range[z,:,:]/1e3,wavelength_band)
    rain_attenuation = np.zeros_like(gas_attenuation)
    rainrate = np.where(smoothed_reflectivity[z,:,:] > 0.,R_Z(smoothed_reflectivity[z,:,:]),0.)
    print(np.min(rainrate),np.max(rainrate),np.mean(rainrate),np.std(rainrate))
    rain_attenuation_rate = calc_rain_attenuation(rainrate,wavelength_band)
    # I don't want NaNs to be part of the range-integration below, so set attenuation rate to 0 where there are no hydrometeors
    rain_attenuation_rate[smoothed_reflectivity[z,:,:] < 5.0] = 0.0
    for nr in range(max_radar_ID):
#       iftrue = True
       time_i = default_timer()
       idx_rdr = np.argwhere(nearest_ID[z,:,:] == nr+1)
       if len(idx_rdr) == 0:
        continue
       this_radar_rows = idx_rdr[:,0]
       this_radar_cols = idx_rdr[:,1]
       # First, let's check whether there's enough reflectivity coverage to even bother calculating attenuation for this site
       term_a = np.sum(rain_attenuation_rate[this_radar_rows,this_radar_cols])
       term_b = np.sum(smoothed_reflectivity[z,this_radar_rows,this_radar_cols][smoothed_reflectivity[z,this_radar_rows,this_radar_cols] >= 0.])
       term_c = np.count_nonzero(rain_attenuation_rate[this_radar_rows,this_radar_cols] > 0.001)
       print(f"{nr:3d} {len(idx_rdr):7d} {term_a:15.5f} ({np.count_nonzero(rain_attenuation_rate[this_radar_rows,this_radar_cols] > 0.)/len(idx_rdr):.2%}) {term_b:15.5f}")
       if term_a < 1.0 or term_c <= 10:
          continue
       # At each radar, identify the azimuths where there was radar reflectivity
       this_radar_mask = (nearest_ID[z,:,:] == nr+1) & (smoothed_reflectivity[z,:,:] > 20.0) # This threshold can probably be lifted as high as 20-30 dBZ if need to cut out more computations
       idx_azs = np.argwhere(this_radar_mask)
       this_azimuth_rows = idx_azs[:,0]
       this_azimuth_cols = idx_azs[:,1]
       az_0 = azimuths[this_azimuth_rows,this_azimuth_cols]
       counts,bins = np.histogram(az_0,bins=np.arange(0,2*pi,radians(1.0)))
       if nr <= n_US_radars:
          radars_attenuated_lons.append(radar_df['LON'].values[nr])
          radars_attenuated_lats.append(radar_df['LAT'].values[nr])
#       for a in range(len(counts)):
#          print(f"{a:3d}, az={degrees(bins[a]):.1f} to {degrees(bins[a+1]):.1f}, count: {counts[a]:4d}")
       for a in range(len(counts)):
          if counts[a] > 0:
             # Identify the gridpoints in this azimuth
             this_azimuth_mask = (nearest_ID[z,:,:] == nr+1) & (azimuths >= bins[a]) & (azimuths < bins[a+1])
             idx_a = np.argwhere(this_azimuth_mask)
             rows_az = idx_a[:,0]
             cols_az = idx_a[:,1]
             term_aa = np.sum(rain_attenuation_rate[rows_az,cols_az])
             if term_aa < 0.01:
   #             print(f"Not enough attenuation at this azimuth bin ({degrees(bins[a]):.1f}-{degrees(bins[a+1]):.1f}) to justify taking the time to compute")
                continue
             # Extract the range values in this azimuth
 #            timeF = default_timer()
             ranges_0 = slant_range[z,rows_az,cols_az]
             sort_idx = np.argsort(ranges_0)
             sort_rows = rows_az[sort_idx]
             sort_cols = cols_az[sort_idx]
             dr_ = np.diff(np.sort(ranges_0))
             # integrate rain_attenuation_rate outward from radar site
 #            if iftrue:
 #               print(f"integrating along azimuth {degrees(bins[a]):.3f}...")
 #            rain_attenuation[sort_rows[1:],sort_cols[1:]] = rain_attenuation[sort_rows[:-1],sort_cols[:-1]] + 0.5*dr_[:-1]/1e3*(rain_attenuation_rate[sort_rows[:-1],sort_cols[:-1]] + rain_attenuation_rate[sort_rows[1:],sort_cols[1:]])
             for b in range(len(ranges_0)-1):
                rain_attenuation[sort_rows[b+1],sort_cols[b+1]] = np.maximum(rain_attenuation[sort_rows[b+1],sort_cols[b+1]],rain_attenuation[sort_rows[b],sort_cols[b]] + 0.5*dr_[b]/1e3*(rain_attenuation_rate[sort_rows[b],sort_cols[b]] + rain_attenuation_rate[sort_rows[b+1],sort_cols[b+1]]))
 #               if iftrue:
 #                  print(f"{b:4d}: {sort_rows[b]:4d},{sort_cols[b]:4d} {ranges_0[sort_idx[b]]/1e3:6.1f} km, {dr_[b]:6.1f} m, {rain_attenuation_rate[sort_rows[b],sort_cols[b]]:.10f} {rain_attenuation[sort_rows[b+1],sort_cols[b+1]]:.10f}")
 #            if iftrue:
 #             iftrue = False
 #            timeFF = default_timer()
 #            print(f"time to process this azimuth containing substantial rain attenuation: {timeFF-timeF:.4f} s")
       time_j = default_timer()
       print(f"processing rain attenuation for this radar took {time_j - time_i:.2f} s")
    processed_reflectivity[z,:,:] = smoothed_reflectivity[z,:,:] - gas_attenuation - rain_attenuation

    plt.figure(figsize=(fig_x,fig_y))
    ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=proj)
    ax.set_extent([lon1,lon2,lat1,lat2],crs=proj)
    ax.coastlines(linewidth=0.5,resolution='50m')
    ax.add_feature(cfeature.BORDERS,edgecolor='black',linewidth=1,zorder=3)
    ax.add_feature(cfeature.STATES,edgecolor='0.5',linewidth=0.5,zorder=2)
    cp = ax.contourf(grid_lons,grid_lats,smoothed_reflectivity[z,:,:],colors=dBZ_colors,levels=dBZ_levs,extend='both',transform=proj,transform_first=True)
    if False:
     ct = ax.contour(grid_lons,grid_lats,beam_grid_size_ratio[z,:,:],levels=np.arange(2.0,9.1,1.0),colors=[1.0,0.5,0.5],linewidths=0.5)
     clbls = ax.clabel(CS=ct,levels=ct.levels,colors='red',fontsize=4,fmt='%.0f')
     for cl in clbls:
      cl.set_rotation(0)
    cb = plt.colorbar(mappable=cp,orientation='horizontal',fraction=cb_fract,shrink=0.5,aspect=40,pad=0.02,ticks=dBZ_levs)
    cb.set_label(f"Reflectivity at {1e3*elev:.0f} m ASL",fontsize=8)
    cb.ax.tick_params(labelsize=6)
    ax.plot(radar_df['LON'].values,radar_df['LAT'].values,'bX',markersize=8,mew=1.0,mec='maroon',transform=proj)
    ax.plot(radars_attenuated_lons,radars_attenuated_lats,'o',color='purple',markersize=12,transform=proj)
    plt.figtext(0.01,0.01,f"Valid time: {MRMS_tstr}",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2})
#    plt.figtext(0.01,0.01,f"Height: {1e3*elev:.0f} m ASL",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='serif')
#    plt.savefig(f"{root}/nature_run_smoothed_reflectivity_{1e3*elev:05.0f}m_{MRMS_tstr}.png",dpi=250)
    plt.savefig(f"{root}/zoom_Texas_nature_run_smoothed_reflectivity_{1e3*elev:05.0f}m_{MRMS_tstr}.png",dpi=250)
    plt.close()

    plt.figure(figsize=(fig_x,fig_y))
    ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=proj)
    ax.set_extent([lon1,lon2,lat1,lat2],crs=proj)
    ax.coastlines(linewidth=0.5,resolution='50m')
    ax.add_feature(cfeature.BORDERS,edgecolor='black',linewidth=1,zorder=3)
    ax.add_feature(cfeature.STATES,edgecolor='0.5',linewidth=0.5,zorder=2)
    cp = ax.contourf(grid_lons,grid_lats,processed_reflectivity[z,:,:],colors=dBZ_colors,levels=dBZ_levs,extend='both',transform=proj,transform_first=True)
    cb = plt.colorbar(mappable=cp,orientation='horizontal',fraction=cb_fract,shrink=0.5,aspect=40,pad=0.02,ticks=dBZ_levs)
    cb.set_label(f"Fully-processed reflectivity at {1e3*elev:.0f} m ASL",fontsize=8)
    cb.ax.tick_params(labelsize=6)
    ax.plot(radar_df['LON'].values,radar_df['LAT'].values,'bX',markersize=8,mew=1.0,mec='maroon',transform=proj)
    ax.plot(radars_attenuated_lons,radars_attenuated_lats,'o',color='purple',markersize=12,mfc='none',mew=1,transform=proj)
    plt.figtext(0.01,0.01,f"Valid time: {MRMS_tstr}",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='monospace')
#    plt.figtext(0.01,0.01,f"Height: {1e3*elev:.0f} m ASL",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='serif')
#    plt.savefig(f"{root}/nature_run_processed_reflectivity_{1e3*elev:05.0f}m_{MRMS_tstr}.png",dpi=250)
    plt.savefig(f"{root}/zoom_Texas_nature_run_processed_reflectivity_{1e3*elev:05.0f}m_{MRMS_tstr}.png",dpi=250)
    plt.close()

    cmap = colormaps['cubehelix_r']
    levels = np.arange(0.0,4.1,0.25)
    cols = cmap(np.linspace(0,1,len(levels)+1))

    if time == stdt:
     plt.figure(figsize=(fig_x,fig_y))
     ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=proj)
     ax.set_extent([lon1,lon2,lat1,lat2],crs=proj)
     ax.coastlines(linewidth=0.5,resolution='50m')
     ax.add_feature(cfeature.BORDERS,edgecolor='black',linewidth=1,zorder=3)
     ax.add_feature(cfeature.STATES,edgecolor='0.5',linewidth=0.5,zorder=2)
     cp = ax.contourf(grid_lons,grid_lats,gas_attenuation,colors=cols,levels=levels,extend='both',transform=proj,transform_first=True)
     ct = ax.contour(grid_lons,grid_lats,smoothed_reflectivity[z,:,:],levels=[10,30,50],colors='black',linewidths=[0.5,1.0,1.5])
     cb = plt.colorbar(mappable=cp,orientation='horizontal',fraction=cb_fract,shrink=0.5,aspect=40,pad=0.02,ticks=levels)
     cb.set_label(f"Reduction of reflectivity due to gas attenuation [dBZ]",fontsize=8)
     cb.ax.tick_params(labelsize=6)
     plt.figtext(0.01,0.01,f"Valid time: {MRMS_tstr}",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='monospace')
 #    plt.figtext(0.01,0.01,f"Height: {1e3*elev:.0f} m ASL",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='serif')
     plt.savefig(f"{root}/gas_attenuation_{1e3*elev:05.0f}m_{MRMS_tstr}.png",dpi=250)
     plt.close()

    if False:
     cmap = colormaps['gnuplot2_r']
     levels = np.geomspace(1,2000,20)
     cols = cmap(np.linspace(0,1,len(levels)+1))

     plt.figure(figsize=(fig_x,fig_y))
     ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=proj)
     ax.set_extent([lon1,lon2,lat1,lat2],crs=proj)
     ax.coastlines(linewidth=0.5,resolution='50m')
     ax.add_feature(cfeature.BORDERS,edgecolor='black',linewidth=1,zorder=3)
     ax.add_feature(cfeature.STATES,edgecolor='0.5',linewidth=0.5,zorder=2)
     cp = ax.contourf(grid_lons,grid_lats,np.where(smoothed_reflectivity[z,:,:] < 5.,0.,rainrate),colors=cols,levels=levels,extend='both',transform=proj,transform_first=True)
     ct = ax.contour(grid_lons,grid_lats,smoothed_reflectivity[z,:,:],levels=[10,30,50],colors='black',linewidths=[0.1,0.25,0.5])
     cb = plt.colorbar(mappable=cp,orientation='horizontal',fraction=cb_fract,shrink=0.9,aspect=50,pad=0.02,ticks=levels)
     cb.set_label(f"Rainfall rate based on reflectivity",fontsize=8)
     cb.ax.tick_params(labelsize=6)
     plt.figtext(0.01,0.01,f"Valid time: {MRMS_tstr}",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='monospace')
 #    plt.figtext(0.01,0.01,f"Height: {1e3*elev:.0f} m ASL",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='serif')
     plt.savefig(f"{root}/rain_rate_{1e3*elev:05.0f}m_{MRMS_tstr}.png",dpi=250)
     plt.close()

    cmap = colormaps['winter_r']
    levels = np.geomspace(1e-3,3.0,20)
    cols = cmap(np.linspace(0,1,len(levels)+1))

    plt.figure(figsize=(fig_x,fig_y))
    ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=proj)
    ax.set_extent([lon1,lon2,lat1,lat2],crs=proj)
    ax.coastlines(linewidth=0.5,resolution='50m')
    ax.add_feature(cfeature.BORDERS,edgecolor='black',linewidth=1,zorder=3)
    ax.add_feature(cfeature.STATES,edgecolor='0.5',linewidth=0.5,zorder=2)
    to_plot = np.where(smoothed_reflectivity[z,:,:] >= 5.0,rain_attenuation_rate,np.nan)
    cp = ax.contourf(grid_lons,grid_lats,to_plot,colors=cols,levels=levels,extend='both',transform=proj,transform_first=True)
#    ct = ax.contour(grid_lons,grid_lats,smoothed_reflectivity[z,:,:],levels=range(5,51,5),colors='black',linewidths=[0.1,0.1,0.1,0.25,0.25,0.25,0.5,0.5,0.5,0.75])
    cb = plt.colorbar(mappable=cp,orientation='horizontal',fraction=cb_fract,shrink=0.9,aspect=50,pad=0.02,ticks=levels)
    cb.set_label(f"Rainfall attenuation rate [dBZ/km]",fontsize=8)
    cb.ax.tick_params(labelsize=6)
    ax.plot(radar_df['LON'].values,radar_df['LAT'].values,'bX',markersize=8,mew=1.0,mec='maroon',transform=proj)
    ax.plot(radars_attenuated_lons,radars_attenuated_lats,'o',color='purple',markersize=12,mfc='none',mew=1,transform=proj)
    plt.figtext(0.01,0.01,f"Valid time: {MRMS_tstr}",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='monospace')
#    plt.figtext(0.01,0.01,f"Height: {1e3*elev:.0f} m ASL",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='serif')
    plt.savefig(f"{root}/zoom_Texas_rain_attenuation_rate_{1e3*elev:05.0f}m_{MRMS_tstr}.png",dpi=250)
    plt.close()

    cmap = colormaps['gnuplot2_r']
    levels = np.arange(0.1,2.6,0.1)
    cols = cmap(np.linspace(0,1,len(levels)+1))

    plt.figure(figsize=(fig_x,fig_y))
    ax = plt.gcf().add_axes([0.01,0.01,0.98,0.98],projection=proj)
    ax.set_extent([lon1,lon2,lat1,lat2],crs=proj)
    ax.coastlines(linewidth=0.5,resolution='50m')
    ax.add_feature(cfeature.BORDERS,edgecolor='black',linewidth=1,zorder=3)
    ax.add_feature(cfeature.STATES,edgecolor='0.5',linewidth=0.5,zorder=2)
    cp = ax.contourf(grid_lons,grid_lats,rain_attenuation,colors=cols,levels=levels,extend='both',transform=proj,transform_first=True)
    ct = ax.contour(grid_lons,grid_lats,smoothed_reflectivity[z,:,:],levels=range(20,51,5),colors='black',linewidths=[0.1,0.25,0.25,0.25,0.5,0.5,0.5,0.75])
#    ct = ax.contour(grid_lons,grid_lats,smoothed_reflectivity[z,:,:],levels=[10,30,50],colors='black',linewidths=[0.1,0.25,0.5])
    cb = plt.colorbar(mappable=cp,orientation='horizontal',fraction=cb_fract,shrink=0.5,aspect=40,pad=0.02,ticks=levels)
    cb.set_label(f"Reduction in radar reflectivity due to rain attenuation [dBZ]",fontsize=8)
    cb.ax.tick_params(labelsize=6)
    ax.plot(radar_df['LON'].values,radar_df['LAT'].values,'bX',markersize=8,mew=1.0,mec='maroon',transform=proj)
    ax.plot(radars_attenuated_lons,radars_attenuated_lats,'o',color='purple',markersize=12,mfc='none',mew=1,mec='lime',transform=proj)
    plt.figtext(0.01,0.01,f"Valid time: {MRMS_tstr}",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='monospace')
#    plt.figtext(0.01,0.01,f"Height: {1e3*elev:.0f} m ASL",ha='left',va='bottom',fontsize=8,fontweight=300,bbox={'facecolor':'white','edgecolor':'black','pad':2},fontfamily='serif')
#    plt.savefig(f"{root}/rain_attenuation_{1e3*elev:05.0f}m_{MRMS_tstr}.png",dpi=500)
    plt.savefig(f"{root}/zoom_Texas_rain_attenuation_{1e3*elev:05.0f}m_{MRMS_tstr}.png",dpi=250)
    plt.close()

    # Create output GRIB2 message and write file
    section2 = b"Creation note: Smoothed reflectivity based on ratio between beam width and grid size"
    out_msg = grb2.Grib2Message(section0=rec.section0,section1=rec.section1,section2=section2,section3=rec.section3,section4=rec.section4,section5=rec.section5)
    out_msg.data = processed_reflectivity[z,:,:]
    out_msg.pack()

    print(out_msg._msg[0:4],out_msg._msg[-4:])

    grb_new = grb2.open(f"{prod_name}_{elev:05.2f}_{MRMS_tstr}_processed.grib2",'w')
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
