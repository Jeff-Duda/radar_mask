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
import pandas as pd
from math import cos, sin, tan, pi, degrees, radians, acos, asin, atan2, sqrt, isinf
from sys import exit

def gc2d(lon1, lat1, lon2, lat2):

    # Input arguments must be 2D arrays with the same dimensions.
    # Returns a 2D array of distances.

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    a = np.cos(lat1) * np.cos(lat2) * np.cos(dlon) + np.sin(lat1)*np.sin(lat2)
    dist = Re*np.arccos(a)
    return dist

def gc_scalar(lon1,lat1,lon2,lat2):

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
   term0 = ((Re+H)**2 + (Re+zzz)**2 - S**2 ) / (2*(Re+H)*(Re+zzz))
   equals = cos(dist/Re)
   errors = np.abs(term0 - equals)
   # Find the lowest error and use that as the solution
   idx = np.argmin(errors)
   beam_height = zzz[idx]
   return beam_height

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

plt.figure(figsize=(5,5))
plt.subplots_adjust(left=0.125,bottom=0.1,right=0.98,top=0.99)

# timea = default_timer()
# S = np.arange(0.,max_radar_distance + 0.001e3,250.0)
# if make_alt_by_range_plots:
#  for I,ang in enumerate([nt,radians(10.0),radians(20.0),0.0,radians(-1.0)]):
#   Z = np.zeros(len(S),dtype=float)
#   xr = np.zeros(len(S),dtype=float)
#   for ii,s in enumerate(S):
#    Z[ii] = sqrt((Re+H)**2 + s**2 + 2*s*(Re+H)*sin(ang)) - Re
#    terma = (Re+H)**2 + (Re+Z[ii])**2 - s**2
#    termb = 2*(Re + H)*(Re + Z[ii])
#    argt = terma/termb
#    xr[ii] = acos(argt)*Re
  #  print(f"slant range: {s/1000:.3f} km , {Z[ii]:10.2f} m elev (horiz. distance: {xr[ii]/1000.:.3f} km)")

#   plt.plot(xr/1000,Z,'-',color=colors[I,:],linewidth=1,label=f"{degrees(ang):.1f}\N{DEGREE SIGN}")
 if False:
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
 # This height isn't precisely known, but in the NEXRAD network it is somewhere in the range of 20-50 m. 30 m is probably appropriate
 # Future advancements of this could compute an actual height difference between the radar location and the underlying terrain
 #hagl = 30.
 #s0 = 1/(Re+hagl)*sqrt(2*Re**3*hagl + 5*Re**2*hagl**2 + 3*Re*hagl**3)
 #th0 = asin(-(2*Re*hagl + hagl**2 + s0**2)/(2*s0*(Re+hagl)))
 #print(f"   The beam would just brush the ground at slant range {s0:.0f} m (angle of {degrees(th0):.3f})")
# Z = np.zeros(len(S),dtype=float)
# xr = np.zeros(len(S),dtype=float)
# for ii,s in enumerate(S):
#  Z[ii] = sqrt((Re+H)**2 + s**2 + 2*s*(Re+H)*sin(th0)) - Re # calc_beam_height_spherical(H,s,th0)
#  terma = (Re+H)**2 + (Re+Z[ii])**2 - s**2
#  termb = 2*(Re + H)*(Re + Z[ii])
#  argt = terma/termb
#  xr[ii] = acos(argt)*Re

 # Alternate calculation derived from Law of sines
# th0 = acos(Re/(Re+30.))
# print(f"Alternate calculation of minimum elevation angle to stay above ground: {degrees(th0):.3f} deg.")
# Z = np.zeros(len(S),dtype=float)
# xr = np.zeros(len(S),dtype=float)
# for ii,s in enumerate(S):
#  Z[ii] = sqrt((Re+H)**2 + s**2 + 2*s*(Re+H)*sin(th0)) - Re # calc_beam_height_spherical(H,s,th0)
#  terma = (Re+H)**2 + (Re+Z[ii])**2 - s**2
#  termb = 2*(Re + H)*(Re + Z[ii])
#  argt = terma/termb
#  xr[ii] = acos(argt)*Re

# plt.plot(xr/1000,Z,'-',color='black',linewidth=1,label=f"{degrees(th0):.3f}\N{DEGREE SIGN}")

 ### Scenario 2: The beam path intersects the ground twice. Obviously, only the closer (lower s0) value is physically reasonable.
 ### But the equations needed are different than for Scenario 1.
 # This requires use of different problem set, i.e., solving a system of two equations, a line, and a circle
 # The solution is x = -0.5*sin(2*theta)*(Re+H) +/- sqrt(tan(theta)**2*(Re**2+H**2)-2*Re*H)/(1+tan(theta)**2)
 #                 y = (Re + H)*(1-sin(theta)**2) +/- 0.5*sin(2*theta)*sqrt(tan(theta)**2*(Re**2+H**2)-2*Re*H)
 # This method also gives the minimum angle at which a solution is achieved, which is equivalent to the angle at which
 # the beam would just brush the ground
 # angle_threshold = atan2(sqrt(2*Re*H),(Re**2+H**2))

 plt.grid(linestyle=':',color='0.8')
 plt.tick_params(axis='both',labelsize=6)
 plt.xlabel("horizontal range [km]",fontsize=8)
 plt.ylabel("Height ASL [m]",fontsize=8)
 plt.xticks(np.concatenate((np.arange(10.,100.,10.),np.arange(100.,300.,25.),np.arange(300.,max_radar_distance + 0.1,50.))))
 plt.xlim(0,max_radar_distance/1e3)
 plt.yticks(np.arange(1000.,25000.1,1000.))
 plt.ylim(0,25001)
 plt.legend(loc=0,fontsize=8)
 plt.savefig(f"{out_dir}/site_{rad}_radar_beam_height_curved_Earth.png",dpi=125)
 plt.close()
 timeb = default_timer()
 print(f"Plotting beam paths took {timeb-timea:.2f} s")
