#Created on Thu Aug 20 13:09:39 2026
from matplotlib import use
use('agg')
import numpy as np
import matplotlib.pyplot as plt
from math import radians, degrees, exp

colors = ['dodgerblue','darkorange','tomato','violet','indigo','lime','darkkhaki','darkturquoise']
S = np.arange(0,460.,0.1)
angles = [0.0,0.5,1.0,2.0,5.0,10.0,20.0]

def gas_attenuation(elev,r):
    atten = (0.4 + 3.45*np.exp(-elev/1.8))* \
        (1-np.exp(-r/(27.8+154*np.exp(-elev/2.2))))
    return atten

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

def Z_R(rain_rate,config=2):
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
   refl = A*rain_rate**b
   return 10*np.log10(refl)

plt.figure(figsize=(6,5))
plt.subplots_adjust(left=0.075,bottom=0.075,right=0.99,top=0.99)
for a in range(len(angles)):
    plt.plot(S,gas_attenuation(angles[a],S),'-',color=colors[a],linewidth=1,label=f"{angles[a]}" + u'\N{DEGREE SIGN}')
plt.grid(linestyle='--',linewidth=1.0,color='0.7')
plt.tick_params(labelsize=6,axis='both')
plt.xlabel("Range [km]",fontsize=8)
plt.ylabel("two-way attenuation (dB)",fontsize=8)
plt.legend(loc=0,fontsize=8)
plt.xticks(np.arange(0,500.,50.))
plt.yticks(np.arange(0.,5.0,0.5))
plt.xlim(0,460)
plt.ylim(0,5)
plt.savefig("gas_attenuation.png",dpi=150)
plt.close()

# From Doviak and Zrnic, page 226
refl = np.arange(5,80.1,0.5)
refl_lin = 10**(refl/10)
plt.figure(figsize=(6,5))
plt.subplots_adjust(left=0.1,bottom=0.075,right=0.99,top=0.99)
plt.semilogy(refl,R_Z(refl,1),'-',color='purple',linewidth=1,label=r"Z = 200R$^{1.6}$ (R = $\frac{Z}{200} + e^{1/1.6}$) - stratiform")
plt.semilogy(refl,R_Z(refl,2),'-',color='crimson',linewidth=1,label=r"Z = 400R$^{1.4}$ (R = $\frac{Z}{400} + e^{1/1.4}$)")
plt.semilogy(refl,R_Z(refl,3),'-',color='yellowgreen',linewidth=1,label=r"Z = 300R$^{1.5}$ (R = $\frac{Z}{300} + e^{1/1.5}$)")
plt.grid(linestyle='--',linewidth=1.0,color='0.7')
plt.tick_params(labelsize=6,axis='both')
plt.xlabel("reflectivity [dBZ]",fontsize=8)
plt.ylabel("estimated rainfall rate [mm/h]",fontsize=8)
plt.legend(loc=0,fontsize=8)
plt.xticks(np.arange(0,80.1,10.))
plt.xlim(0,70)
plt.ylim(1.0,1e5)
plt.savefig("R-Z_relationships.png",dpi=150)
plt.close()

rainfall_rate = R_Z(refl,2)
print(np.min(rainfall_rate),np.max(rainfall_rate))
plt.figure(figsize=(6,5))
ax = plt.gcf().add_axes([0.1,0.075,0.88,0.85])
ax2 = plt.twiny(ax)
ax.semilogy(rainfall_rate,calc_rain_attenuation(rainfall_rate,1),'-',color='purple',linewidth=1,label="S-band")
ax.semilogy(rainfall_rate,calc_rain_attenuation(rainfall_rate,2),'-',color='saddlebrown',linewidth=1,label="C-band")
ax.semilogy(rainfall_rate,calc_rain_attenuation(rainfall_rate,3),'-',color='darkturquoise',linewidth=1,label="X-band")
#ax2.semilogy(refl,calc_rain_attenuation(rainfall_rate,1),'-',color='purple',linewidth=1)
#ax2.semilogy(refl,calc_rain_attenuation(rainfall_rate,2),'-',color='saddlebrown',linewidth=1)
#ax2.semilogy(refl,calc_rain_attenuation(rainfall_rate,3),'-',color='darkturquoise',linewidth=1)
#secax = ax.secondary_xaxis('top',functions=(R_Z,Z_R))
ax.grid(linestyle='--',linewidth=1.0,color='0.7')
ax.tick_params(labelsize=6,axis='both')
ax.set_xlabel("rainfall rate [mm/hr]",fontsize=8)
ax.set_ylabel("specific attenuation [dB/km]",fontsize=8)
#plt.legend(loc=0,fontsize=8)
plt.figlegend(loc='upper left',bbox_to_anchor=(0.1,0.98),fontsize=8)
#secax.tick_params(labelsize=6)
#secax.set_xticks(np.arange(0,70.1,10.))
#secax.set_xlabel("Reflectivity from rain [dBZ]",fontsize=8)
ax2.tick_params(axis='x',labelsize=6)
ax2.set_xlabel("Reflectivity from rain [dBZ]",fontsize=8)
ax2.set_xticks(np.arange(0,70.1,5.))
print(Z_R(200),Z_R(250))
ax2.set_xlim(Z_R(0.1),Z_R(250))
ax.set_xlim(0.1,250)
ax.set_ylim(1e-4,1e1)
plt.savefig("rain_attenuation_rate_relationships.png",dpi=150)
plt.close()
