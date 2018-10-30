from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d, Axes3D
import scipy.integrate
import time
import sys
import os
import multiprocessing
from functools import partial

start_time = time.time()




G = 6.6726e-11 #Nm^2kg^-2 - Newtons universal gravitational constant
mA = 1.23e12 * 1.99e30 #kg - mass of Andromeda
mM = 5.8e11 * 1.99e30 #kg - mass of Milky Way
RA = 67e3 * 3.09e16 #m - radius of Androm.
RM = 40e3 * 3.09e16 #m - radius of MW
pc = 3.086e16

r_200_M = 200 * 1e3 * pc
r_s_M = 10.7 * 1e3 * pc

r_200_A = 240 * 1e3 * pc
r_s_A = 34.6 * 1e3 * pc

soft_len = 5000 * pc



def DM_density(r,rs,r_200):
	c = r_200/rs
	H = 70 *1e3 /(1e6 * pc) #Hubble constant in SI units
	rho_crit = 3 * H**2/(8*np.pi*G) #critical density
	rho_0 = 200/3 * rho_crit * c**3/(np.log(1+c)-c/(1+c)) 
	rho = rho_0 * (r/rs)**(-1) * (1+r/rs)**(-2)
	return rho

def Mass_r(r,r_s,r_200):
	c = r_200/r_s #concentration factor
	H = 70 *1e3 /(1e6 * pc) #Hubble constant in SI units
	rho_crit = 3 * H**2/(8*np.pi*G) #critical density
	rho_0 = 200/3* rho_crit * c**3/(np.log(1+c)-c/(1+c))
	
	m = 4*np.pi*rho_0*r_s**3 * (-1*r/(r_s+r) + np.log((r_s+r)/r_s)) #mass enclosed within r
	return m
	
#def NFW_CDF(r,r_s,r_200):
	
	
def rot_curve(r,r_s,r_200):
	 ang_v = (G*Mass_r(r,r_s,r_200)/r)**0.5
	 return ang_v
	

if __name__=='__main__':
	rs = np.linspace(0,260*1e3*pc,1000)
	rhos = DM_density(rs,r_s_M,r_200_M)
	rho_200 = DM_density(r_200_M,r_s_M,r_200_M)
	
	ms = Mass_r(rs,r_s_M,r_200_M)
	m_200 = Mass_r(r_200_M,r_s_M,r_200_M)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(rs/r_200_M,rhos/rho_200,'r--')
	plt.xlabel(r'r/$R_{200}$')
	plt.ylabel(r'$\rho/<\rho_{200}>$')
	ax.set_yscale('log')
	ax.set_xscale('log')
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(rs/r_200_M,ms/m_200,'r--')
	plt.xlabel(r'r/$R_{200}$')
	plt.ylabel(r'$m/m_{200}$')
	#ax.set_yscale('log')
	#ax.set_xscale('log')
	
	
	fig=plt.figure()
	ax = fig.add_subplot(111)	
	plt.plot(rs/r_200_M,rot_curve(rs,r_s_M,r_200_M),'r--')
	plt.show()
	

