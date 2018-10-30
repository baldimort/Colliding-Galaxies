from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d, Axes3D
import scipy.integrate
import time
from functools import partial
import multiprocessing
from matplotlib import rc
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['mathtext.fontset'] = 'cm'

start_time = time.time()


G = 6.6726e-11 #Nm^2kg^-2 - Newtons universal gravitational constant
mA = 1.23e12 * 1.99e30 #kg - mass of Andromeda
mM = 5.8e11 * 1.99e30 #kg - mass of Milky Way
RA = 67e3 * 3.09e16 #m - radius of Androm.
RM = 40e3 * 3.09e16 #m - radius of MW 40e3
pc = 3.086e16 #m
r_200_M = 210 * 1e3 * pc
r_s_M = 19 * 1e3 * pc
soft_len = 500 * pc



	
def Star_state((x,y,z,vx,vy,vz),t):#(x1,y1,z1),(x2,y2,z2)): #state of star, t, positions of galactic cores
	'''Differential equation of star depending on position of glactic core'''
	
	r_MW = np.array([0,0,0]) #position vecotrs of MW, Androm and star
	r_star = np.array((x,y,z))
	
	dist_MW = sum((r_star - r_MW)**2)**0.5 #distances from galaxies
	soft_dist = (dist_MW**2+soft_len**2)**0.5
		
	accns = -G * mM * (r_star - r_MW)/soft_dist**3 
	
	return (vx,vy,vz,accns[0],accns[1],accns[2])

def Solve_star_taylor(time_base,dt,star):
	results = []
	results.append(star)
	for j,t in enumerate(time_base):
		results.append(Taylor(dt,t,results[-1]))
	return np.array(results[:-1])
	
def Solve_star_RK(time_base,dt,star):
	results = []
	results.append(star)
	for j,t in enumerate(time_base):
		results.append(RK(dt,t,results[-1]))
	return np.array(results[:-1])

def Star_interactions(stars,time_base):
	'''Loop through each star calculating each state'''
	star_solns = np.zeros((stars.shape[0],time_base.size,stars.shape[1]))
	dt = (time_base[-1]-time_base[0])/time_base.size
	
	for i,val in enumerate(stars):
		if (i%10)==0: print 'Star = %s' % i
		star_solns[i] = scipy.integrate.odeint(Star_state,val,time_base,(dt),mxstep=5000)
	
	return star_solns
	


	
def Taylor(a, t, (xn,yn,zn,xn_vel,yn_vel,zn_vel)):
	'''Performs the taylor method'''
	star_state = np.array(Star_state((xn,yn,zn,xn_vel,yn_vel,zn_vel),t))

	xs = np.array([xn,yn,zn])
	vels = star_state[0:3]
	accs = star_state[3:]
	

	xs_new = xs + a * vels + a**2/2 * accs
	vs_new = vels + a * accs
	
	return [xs_new[0],xs_new[1],xs_new[2],vs_new[0],vs_new[1],vs_new[2]]

def RK(a, t, (xn,yn,zn,xn_vel,yn_vel,zn_vel)):
	'''Performs the RK4 method'''
	star_state = np.array(Star_state((xn,yn,zn,xn_vel,yn_vel,zn_vel),t))
	
	xs = np.array([xn,yn,zn])
	vels = star_state[0:3]
	accs = star_state[3:]
	
	z1 = xs + a/2 * vels
	z1_v = vels + a/2 * accs
	z1_a = np.array(Star_state((z1[0],z1[1],z1[2],z1_v[0],z1_v[1],z1_v[2]),t+a/2))[3:]
	
	z2 = xs + a/2 * z1_v
	z2_v = vels + a/2 * z1_a
	z2_a = np.array(Star_state((z2[0],z2[1],z2[2],z2_v[0],z2_v[1],z2_v[2]),t+a/2))[3:]
	
	z3 = xs + a* z2_v
	z3_v = vels + a* z2_a
	z3_a = np.array(Star_state((z3[0],z3[1],z3[2],z3_v[0],z3_v[1],z3_v[2]),t+a/2))[3:]
	
	x_new = xs + a/6 * (vels + 2*z1_v + 2*z2_v+ z3_v)
	x_vel_new = vels +a/6 * (accs + 2*z1_a + 2*z2_a + z3_a)
	
	new_states = [x_new[0],x_new[1],x_new[2],x_vel_new[0],x_vel_new[1],x_vel_new[2]]
	return new_states

def SAVEFIG(message,filetag,f):
	a = raw_input(message)
	if a == 'y': f.savefig("../Figures/"+time.strftime("%Y%m%d_%H%M%S_")+filetag+'.png',dpi=600,format='png')

	
if __name__ == '__main__':
	
	"""INTEGRATOR ACCURACY"""	
	
	t_final_sim = 20e9
	n_points = 1e5
	dt = 20e9/1e5
	
	ts = np.arange(0,40e9,dt)
	ts = ts * 365 * 24 * 3600
	
	radius = 20e3 * pc
	v = (G*mM/radius)**0.5
	#N = 1e6
	Orbits = 100
	#dt = 
	#ts = np.linspace(0,2*np.pi/(v/radius)*Orbits,N)
	#print "T_f = %.1f Gyrs" % (ts[-1]/(1e9*3600*24*365))
		
	def star_orbit(v,r,t):
		x = r*np.cos(t*v/r)
		y = r*np.sin(t*v/r)
		return (x,y)
	
	xs, ys = star_orbit(v,radius,ts)
	results = Solve_star_taylor(ts,ts[1]-ts[0],[xs[0],ys[0],0,0,v,0])
	xs_T,ys_T = results[:,0], results[:,1]
	
	results_R = Solve_star_RK(ts,ts[1]-ts[0],[xs[0],ys[0],0,0,v,0])
	xs_R,ys_R = results_R[:,0], results_R[:,1]
	
	results_S = scipy.integrate.odeint(Star_state,np.array([xs[0],ys[0],0,0,v,0]),ts)
	xs_S,ys_S = results_S[:,0],results_S[:,1]
	
	fig2 = plt.figure(figsize=(6.4*3/4,6.4*3/4))	
	
	rc('font',**{'family':'serif','serif':['Times']})
	rc('text', usetex=True)
	
	a1 = fig2.add_subplot(111)
	
	
	a1.plot(ts/(1e9*3600*24*365),((xs_T-xs)**2+(ys_T-ys)**2)**0.5/radius,'b--',label='Taylor',alpha=0.5)
	a1.plot(ts/(1e9*3600*24*365),((xs_R-xs)**2+(ys_R-ys)**2)**0.5/radius,'r-',label='RK4')
	a1.plot(ts/(1e9*3600*24*365),((xs_S-xs)**2+(ys_S-ys)**2)**0.5/radius,'g--',label='SciPy')
	
	#a1.set_yscale('log')
	a1.legend(fontsize=12)
	a1.set_xlabel(r'$t (Gyrs)$',fontsize=12)
	a1.set_ylabel(r'$\Delta R/R_{0}$',fontsize=12)
	a1.set_xlim([0,40])
	a1.set_ylim([0,3])
	a1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
	plt.tight_layout()
	
	SAVEFIG('save fig? (y/n)','INTEGRATOR_ANALYSIS',fig2)
	plt.show()




