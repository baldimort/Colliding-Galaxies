from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d, Axes3D
import scipy.integrate
import time
from functools import partial
import multiprocessing
import sys
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


def SAVEFIG(message,filetag):
	a = raw_input(message)
	if a == 'y' or a=='Y': plt.savefig("../Figures/"+time.strftime("%Y%m%d_%H%M%S_")+filetag+"_%.0f Gyrs.png"%(time_base[-1]/(3600*24*365*1e9)),dpi=600,format='png')
	
def Galaxies_state((x1,y1,z1,vx1,vy1,vz1,x2,y2,z2,vx2,vy2,vz2),t,soft_len,shit):
	'''Gravitational interactions between two galactic core bodies'''
	r_MW = np.array((x1,y1,z1)) #position vecotrs of MW and Androm.
	r_A = np.array((x2,y2,z2))
	
	dist_gals = sum((r_A - r_MW)**2)**0.5 #distance between galaxies
	soft_dist = (dist_gals**2+soft_len**2)**0.5
	
	a_MW = G * mA * (r_A - r_MW)/soft_dist**3 #acceleration of MW due to Androm.
	a_A = G * mM * (r_MW - r_A)/soft_dist**3 #acceleration of A due to MW
	
	return np.array([vx1,vy1,vz1,a_MW[0],a_MW[1],a_MW[2],vx2,vy2,vz2,a_A[0],a_A[1],a_A[2]])

def Galaxy_pos(t,states,dt):
	'''returns position of galaxies given t and states'''
	try: return states[int(np.floor(t/dt))]
	except Exception:
		print 'BODGE'
		return states[-1] #<----------------------------SOMETHING ISNT RIGHT HERE - BODGE ALERT
	

	
def Taylor(a, t, (xn,yn,zn,xn_vel,yn_vel,zn_vel),gal_states):
	'''Performs the taylor method'''
	star_state = np.array(Star_state((xn,yn,zn,xn_vel,yn_vel,zn_vel),t,a,gal_states))

	xs = np.array([xn,yn,zn])
	vels = star_state[0:3]
	accs = star_state[3:]
	

	xs_new = xs + a * vels + a**2/2 * accs
	vs_new = vels + a * accs
	
	return [xs_new[0],xs_new[1],xs_new[2],vs_new[0],vs_new[1],vs_new[2]]

def RK(a, t, (xn,yn,zn,xn_vel,yn_vel,zn_vel)):
	'''Performs the RK4 method'''
	x_accel, y_accel, z_accel = 1,1,1
	new_states = []
	
	x, x_v, x_a = np.array([xn,yn,zn]), np.array([xn_vel,yn_vel,zn_vel]), np.array([x_accel,y_accel,z_accel])
	
	z1 = x +a/2 * x_v
	z1_v = x_v + a/2 * x_a
	z1_a = np.array(f_rocket((z1[0],z1[1],z1_v[0],z1_v[1]),t)[2:]) #NEEDS ADJUSTING FOR 3D operates on both dimensions simultaneously using arrays	
	
	z2 = x + a/2 * z1_v
	z2_v = x_v + a/2 * z1_a
	z2_a = np.array(f_rocket((z2[0],z2[1],z2_v[0],z2_v[1]),t)[2:])
	
	z3 = x + a* z2_v
	z3_v = x_v + a* z2_a
	z3_a = np.array(f_rocket((z3[0],z3[1],z3_v[0],z3_v[1]),t)[2:])
	
	x_new = x + a/6 * (x_v + 2*z1_v + 2*z2_v+ z3_v)
	x_vel_new = x_v +a/6 * (x_a + 2*z1_a + 2*z2_a + z3_a)
	
	new_states = [x_new[0],x_new[1],x_vel_new[0],x_vel_new[1]]
	return new_states

def defleciton(initial_v,final_v):
	cos_theta = np.dot(initial_v,final_v)/(np.sum(initial_v**2)**0.5*np.sum(final_v**2)**0.5)
	return np.arccos(cos_theta)
	
if __name__ == '__main__':
	
	soft_lens = np.linspace(0,float(sys.argv[2])*1e3*pc,int(sys.argv[1])) #argv[1] is the number of plots, argv[2] is the max softening length
	deflection_angles_MW = []
	deflection_angles_M31 = []
	for soft_len in soft_lens:
	
		N_points = 1e4
		t_final = 5e9 #years
		
		t_final_secs = t_final * 365 * 24 * 3600 
		
		time_base = np.linspace(0,t_final_secs,N_points)
		#setup timebase
		dt = (time_base[-1]-time_base[0])/time_base.size
		
		initial_cons = (-300e3 * pc,0,0,60e3,5e3,0,300e3 * pc,0,0,-60e3,-5e3,0) #MW POS-VEL, ANDROM POS-VEL
		galaxy_states = scipy.integrate.odeint(Galaxies_state,initial_cons,time_base,args=(soft_len,'wanker'))
		
		rs = np.linspace(0,r_200_M,1000)
				
		galaxy_states[:,0:3] = galaxy_states[:,0:3]/pc/1e3
		galaxy_states[:,6:9] = galaxy_states[:,6:9]/pc/1e3
		#coversion to kPc
		
		deflection_angles_MW.append(defleciton(galaxy_states[0,3:6],galaxy_states[-1,3:6]))
		deflection_angles_M31.append(defleciton(galaxy_states[0,9:],galaxy_states[-1,9:]))

		"""fig = plt.figure()
		ax1 = fig.add_subplot(111)
		#ax2 = fig.add_subplot(212,projection='3d')
		ax1.set_xlabel('x (kpc)')
		ax1.set_ylabel('y (kpc)')
		ax1.set_ylim([-300,100])
		
		ax1.set_aspect('equal', 'datalim')
		'''ax1.set_xlim((-50,50))
		ax1.set_ylim((-50,50))'''

		ax1.plot(galaxy_states[:,0],galaxy_states[:,1],'r--',label='MW')
		ax1.plot(galaxy_states[:,6],galaxy_states[:,7],'g--',label='M31')
		ax1.plot([],[],label=r'$\epsilon$ = %.1f kpc'%(soft_len/pc/1e3),color='None')
		ax1.legend(loc=3)
		
		third = int(np.around(galaxy_states.shape[0]*1/3))

			
		ax1.arrow(galaxy_states[third,0],galaxy_states[third,1],
				galaxy_states[third+1,0]-galaxy_states[third,0],galaxy_states[third+1,1]-galaxy_states[third,1],color='red',shape='full', lw=5, length_includes_head=True, head_width=.05)
		ax1.arrow(galaxy_states[third,6],galaxy_states[third,7],
				galaxy_states[third+1,6]-galaxy_states[third,6],galaxy_states[third+1,7]-galaxy_states[third,7],color='green',shape='full', lw=5, length_includes_head=True, head_width=.05)
		ax1.arrow(galaxy_states[2*third,0],galaxy_states[2*third,1],
				galaxy_states[2*third+1,0]-galaxy_states[2*third,0],galaxy_states[2*third+1,1]-galaxy_states[2*third,1],color='red',shape='full', lw=5, length_includes_head=True, head_width=.05)
		ax1.arrow(galaxy_states[2*third,6],galaxy_states[2*third,7],
				galaxy_states[2*third+1,6]-galaxy_states[2*third,6],galaxy_states[2*third+1,7]-galaxy_states[2*third,7],color='green',shape='full', lw=5, length_includes_head=True, head_width=.05)
				
				
		xlims = ax1.get_xlim()
		ylims = ax1.get_ylim()
		
		xtext = (xlims[1] - xlims[0])*0.1 + xlims[0]
		ytext = (ylims[1] - ylims[0])*0.1 + ylims[0]
		
		pos = np.array([xtext,ytext]).min()
		
		#plt.text(pos,pos,r"$\epsilon=%.1f$ kpc"%(soft_len/(pc*1e3)),fontname='Arial',fontsize=18)
		
		SAVEFIG('Save softening plot? (y/n)','SOFTENING_%.1fkpc'%(soft_len/(pc*1e3)))"""
	
	size=12
	fig=plt.figure(figsize=(6.4*3/4,6.4*3/4))
	ax =fig.add_subplot(111)
	
	ax.plot(np.array(soft_lens)/(1e3*pc),np.array(deflection_angles_MW)*180/np.pi,'r--',label='MW')
	ax.plot(np.array(soft_lens)/(1e3*pc),np.array(deflection_angles_M31)*180/np.pi,'g-',label='M31')
	
	ax.legend(loc=3,fontsize=size)
	ax.axvline(7.5,linestyle='--')
	ax.set_ylabel(r'$\theta$ (deg)',fontsize=size)
	ax.set_xlabel(r'$\epsilon$ (kpc)',fontsize=size)
	ax.set_xlim([0,10])
	
	ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
	plt.tight_layout()
	SAVEFIG('Save softening deflection plot? (y/n)','SOFTENING_DEFLECTION_')
	
	
	plt.show()






