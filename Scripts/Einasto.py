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
from scipy.special import gamma,gammainc

start_time = time.time()




G = 6.6726e-11 #Nm^2kg^-2 - Newtons universal gravitational constant
mA = 1.23e12 * 1.99e30 #kg - mass of Andromeda
mM = 5.8e11 * 1.99e30 #kg - mass of Milky Way
RA = 67e3 * 3.09e16 #m - radius of Androm.
RM = 40e3 * 3.09e16 #m - radius of MW
pc = 3.086e16

r_200_M = 210 * 1e3 * pc
r_s_M = 19 * 1e3 * pc

r_200_A = 270 * 1e3 * pc
r_s_A = 24 * 1e3 * pc

soft_len = 5000 * pc



def DM_density_Ein(r,rho_2,r_2,alpha):
	rho = rho_2 * np.exp(-2/alpha * ((r/r_2)**alpha-1))
	return rho

def Mass_r(r,r_s,r_200):

	return 1

def gam(s,x):
	return gamma(s) * gammainc(s,x)
	
def EIN_norm(r,a,r_200,r_2):
	return a*r_200**3*np.exp(-2/a*(r/r_2)**a) / (3*r_2**3*(a/2)**(3/a)*gam(3/a,2/a*(r_200/r_2)**a))
	
	
def rot_curve(r,r_s,r_200):
	 ang_v = (G*Mass_r(r,r_s,r_200)/r)**0.5
	 return ang_v
	

if __name__=='__main__':
	rs = np.linspace(0,260*1e3*pc,1000)
	n_dens = EIN_norm(rs,3,r_200_A,r_s_A)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(rs/r_200_A,n_dens,'r--')
	plt.xlabel(r'r/$R_{200}$')
	plt.ylabel(r'$n(r)/\langle n \rangle$')
	ax.set_yscale('log')
	ax.set_xscale('log')

	plt.show()	

