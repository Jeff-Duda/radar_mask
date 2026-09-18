import numpy as np
from geopy import distance
from geopy.point import Point
from timeit import default_timer
from math import cos, sin, tan, pi, degrees, radians, acos, asin, atan2, sqrt, isinf
from sys import exit

maj_axis = distance.ELLIPSOIDS['WGS-84'][1]
min_axis = distance.ELLIPSOIDS['WGS-84'][0]
Re = sqrt(maj_axis*min_axis)*1000.

def gc1d(lon1, lat1, lon2, lat2):

    # Input arguments must be 1D arrays with the same dimensions.
    # Returns a 1D array of distances.

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    a = np.cos(lat1) * np.cos(lat2) * np.cos(dlon) + np.sin(lat1)*np.sin(lat2)
    dist = Re*np.arccos(a)
    return dist


def gc2d(lon1, lat1, lon2, lat2):

    # Input arguments must be 2D arrays with the same dimensions.
    # Returns a 2D array of distances.
    # Input lats and lons must be in radians

    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
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
