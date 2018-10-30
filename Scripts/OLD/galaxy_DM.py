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

r_200_M = 210 * 1e3 * pc
r_s_M = 19 * 1e3 * pc

r_200_A = 270 * 1e3 * pc
r_s_A = 24 * 1e3 * pc

soft_len = 50 * pc



def DM_density(r,rs,r_200):
	c = r_200/rs
	rho_0 = 200/3 * c**3/(np.log(1+c)-c/(1+c))
	rho = rho_0 * (r/rs)**(-1) * (1+r/rs)**(-2)
	return rho

def Mass_r(r,r_s,r_200):
	c = r_200/r_s #concentration factor
	H = 70 *1e3 /(1e6 * pc) #Hubble constant in SI units
	rho_crit = 3 * H**2/(8*np.pi*G) #critical density
	rho_0 = 200/3* rho_crit * c**3/(np.log(1+c)-c/(1+c))
	
	m = 4*np.pi*rho_0*r_s**3 * (-1*r/(r_s+r) + np.log((r_s+r)/r_s)) #mass enclosed within r
	return m
	
def Galaxies_state((x1,y1,z1,vx1,vy1,vz1,x2,y2,z2,vx2,vy2,vz2),t):
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
	
def Star_state((x,y,z,vx,vy,vz),t,dt,states):#(x1,y1,z1),(x2,y2,z2)): #state of star, t, positions of galactic cores
	'''Differential equation of star depending on positions of glactic cores'''
	
	N = int(np.floor(t/dt))
	
		
	r_MW = states[N,0:3] #position vecotrs of MW, Androm and star
	r_A = states[N,6:9]
	r_star = np.array((x,y,z))
	
	dist_MW = sum((r_star - r_MW)**2)**0.5 #distances from galaxies
	soft_dist_MW = (dist_MW**2+soft_len**2)**0.5
	
	dist_Androm = sum((r_star - r_A)**2)**0.5
	soft_dist_Androm = (dist_Androm**2+soft_len**2)**0.5
	
	accns = -G * Mass_r(dist_MW,r_s_M,r_200_M) * (r_star - r_MW)/dist_MW**3 - G * Mass_r(dist_Androm,r_s_A,r_200_A) * (r_star - r_A)/dist_Androm**3
	
	return (vx,vy,vz,accns[0],accns[1],accns[2])

def Solve_star_rk(time_base,dt,galaxy_states,star):
	results = []
	results.append(star)
	for j,t in enumerate(time_base):
		results.append(RK(dt,t,results[-1],galaxy_states))
	return np.array(results[1:])	
	

def Initalalise_stars_disk((x,y,z),(vx,vy,vz),N,radius,inc,rot,mass,gal): #N is no. of stars, radius is of galaxy, inc is angle to x, rot is rotation around z
	'''Generates the orbits of stars around a galaxy given the galactic core and galactic velocity'''
	star_r = np.random.rand(N,2) #random stars
	star_r[:,0] = star_r[:,0] * radius
	star_r[:,1] = star_r[:,1] * 2* np.pi
	random_offset = 0.3e3*pc*(2*np.random.rand(N)-1)
	
	star_states = np.zeros((N,6)) #x,y,z,vx,vy,vz for each star
	star_states[:,0] = star_r[:,0]*np.cos(star_r[:,1])*np.cos(inc) + random_offset*np.sin(inc) #inital conversion to cartesian at origin
	star_states[:,1] = star_r[:,0]*np.sin(star_r[:,1])
	star_states[:,2] = star_r[:,0]*np.sin(inc)*np.cos(star_r[:,1]) + random_offset*np.cos(inc)
	
	rotation = np.array([[np.cos(rot),-np.sin(rot),0],[np.sin(rot),np.cos(rot),0],[0,0,1]]) #rotation matrix
	rotated = np.dot(rotation,star_states[:,0:3].T) #performs rotation on position
	star_states[:,0:3] = rotated.T
	
	if gal == "MW": ang_v = (G*Mass_r(star_r[:,0],r_s_M,r_200_M)/star_r[:,0])**0.5
	elif gal == "A": ang_v = (G*Mass_r(star_r[:,0],r_s_A,r_200_A)/star_r[:,0])**0.5	#<------------- radial velocity of stars given radius - change to dark halo curve
	
	
	star_states[:,3] = -ang_v*np.sin(star_r[:,1])*np.cos(inc)
	star_states[:,4] = ang_v*np.cos(star_r[:,1])
	star_states[:,5] = -ang_v*np.sin(star_r[:,1])*np.sin(inc)
	
	star_states[:,3:] = np.dot(rotation,star_states[:,3:].T).T #performs rotation on velocities
	
	for i,val in enumerate((x,y,z)): star_states[:,i] =  star_states[:,i] + val #moves stars to location of galactic core
	for i,val in enumerate((vx,vy,vz)): star_states[:,i+3] += val
	
	return star_states

def Taylor(a, t, (xn,yn,zn,xn_vel,yn_vel,zn_vel),gal_states):
	'''Performs the taylor method'''
	star_state = np.array(Star_state((xn,yn,zn,xn_vel,yn_vel,zn_vel),t,a,gal_states))

	xs = np.array([xn,yn,zn])
	vels = star_state[0:3]
	accs = star_state[3:]
	

	xs_new = xs + a * vels + a**2/2 * accs
	vs_new = vels + a * accs
	
	return [xs_new[0],xs_new[1],xs_new[2],vs_new[0],vs_new[1],vs_new[2]]

def RK(a, t, (xn,yn,zn,xn_vel,yn_vel,zn_vel),gal_states):
	'''Performs the RK4 method'''
	star_state = np.array(Star_state((xn,yn,zn,xn_vel,yn_vel,zn_vel),t,a,gal_states))
	
	xs = np.array([xn,yn,zn])
	vels = star_state[0:3]
	accs = star_state[3:]
	
	z1 = xs + a/2 * vels
	z1_v = vels + a/2 * accs
	z1_a = np.array(Star_state((z1[0],z1[1],z1[2],z1_v[0],z1_v[1],z1_v[2]),t+a/2,a,gal_states))[3:]
	
	z2 = xs + a/2 * z1_v
	z2_v = vels + a/2 * z1_a
	z2_a = np.array(Star_state((z2[0],z2[1],z2[2],z2_v[0],z2_v[1],z2_v[2]),t+a/2,a,gal_states))[3:]
	
	z3 = xs + a* z2_v
	z3_v = vels + a* z2_a
	z3_a = np.array(Star_state((z3[0],z3[1],z3[2],z3_v[0],z3_v[1],z3_v[2]),t+a/2,a,gal_states))[3:]
	
	x_new = xs + a/6 * (vels + 2*z1_v + 2*z2_v+ z3_v)
	x_vel_new = vels +a/6 * (accs + 2*z1_a + 2*z2_a + z3_a)
	
	new_states = [x_new[0],x_new[1],x_new[2],x_vel_new[0],x_vel_new[1],x_vel_new[2]]
	return new_states

	
if __name__ == '__main__':
	pool = multiprocessing.Pool(processes=4)
		
	N_points = 1e4
	t_final = 20e9 #years
	
	t_final_secs = t_final * 365 * 24 * 3600 
	stars_per_gal = 1000

	factor = 1.1	

	time_base_gals = np.linspace(0,t_final_secs*factor,N_points*factor)
	#setup timebase
	
	print "Galaxy core simulation: started"
	initial_cons = (-300e3 * pc,0,0,50e3,0,0,300e3 * pc,0,0,-60e3,10e3,0) #MW POS-VEL, ANDROM POS-VEL
	galaxy_states = scipy.integrate.odeint(Galaxies_state,initial_cons,time_base_gals)
	print "Galaxy core simulation: complete"

	time_base = time_base_gals[:int(N_points)]
	dt = (time_base[-1]-time_base[0])/time_base.size

	#calculates galactic core paths
	
	#MW stars
	print "Galaxy 1 stars simulation: started"
	stars = Initalalise_stars_disk(initial_cons[0:3],initial_cons[3:6],stars_per_gal,RM,0,0.0,mM,'MW')	
	result = pool.map(partial(Solve_star_rk,time_base,dt,galaxy_states),stars)
	star_solns_M = np.array(result)
	print "Galaxy 1 stars simulation: complete"
	
	#Androm stars
	print "Galaxy 2 stars simulation: started"
	stars = Initalalise_stars_disk(initial_cons[6:9],initial_cons[9:],stars_per_gal,RA,np.pi/4,0.8,mA,'A')
	result = pool.map(partial(Solve_star_rk,time_base,dt,galaxy_states),stars)
	star_solns_A = np.array(result)
	print "Galaxy 2 stars simulation: complete"
	pool.close()
	pool.join()
	#solves for stellar disk
	
	star_solns_A[:,:,0:3] = star_solns_A[:,:,0:3] / pc / 1e3
	star_solns_M[:,:,0:3] = star_solns_M[:,:,0:3] / pc / 1e3
	galaxy_states[:,0:3] = galaxy_states[:,0:3] / pc / 1e3
	galaxy_states[:,6:9] = galaxy_states[:,6:9] / pc / 1e3
	#coversion to kPc
	
	#SAVING DATA
	directory = time.strftime("%Y%m%d_%H%M%S")
	os.mkdir(directory+"_DATA")
	os.chdir(directory+"_DATA")
	np.save("MW_DATA",star_solns_M)
	np.save("M31_DATA",star_solns_A)
	np.save("GAL_CORE_DATA",galaxy_states)
	
	fig = plt.figure()
	ax1 = fig.add_subplot(111,projection='3d')
	ax1.set_xlabel('x (kpc)')
	ax1.set_ylabel('y (kpc)')
	ax1.set_zlabel('z (kpc)')
	
	ax1.set_xlim((-200,200))
	ax1.set_ylim((-100,100))
	ax1.set_zlim((-100,100))
	
	MW_core, = ax1.plot([],[],[],'go',label='MW',ms=2)
	Androm_core, = ax1.plot([],[],[],'bo',label='Andromeda',ms=2)
	star_points_A = []
	star_points_M = []
	for i in range(stars_per_gal):
		p, = ax1.plot([],[],[],'o',color='#9F14E9',ms=1)
		q, = ax1.plot([],[],[],'o',color='#FAA821',ms=1)
		star_points_A.append(p)
		star_points_M.append(q)
	
	
	frame_factor = 1e0
	
	
	def update(j):
		i = int(j * frame_factor)
		MW_core.set_data(galaxy_states[i,0],galaxy_states[i,1])
		MW_core.set_3d_properties(galaxy_states[i,2])
		Androm_core.set_data(galaxy_states[i,6],galaxy_states[i,7])
		Androm_core.set_3d_properties(galaxy_states[i,8])
		for s,k in enumerate(star_points_A):
			k.set_data(star_solns_A[s,i,0],star_solns_A[s,i,1])
			k.set_3d_properties(star_solns_A[s,i,2])
		for s,k in enumerate(star_points_M):
			k.set_data(star_solns_M[s,i,0],star_solns_M[s,i,1])
			k.set_3d_properties(star_solns_M[s,i,2])
		#ax1.relim()
	print 'Time taken = %.2f' % (time.time()-start_time)	 
	video_ani = animation.FuncAnimation(fig,update,frames=int(np.floor(time_base.size/frame_factor)),
				interval=1,repeat_delay=0)	
	
	
	plt.rcParams['animation.ffmpeg_path'] = r'/usr/bin/ffmpeg'
	FFwriter = animation.FFMpegWriter(fps=60, bitrate=-1, extra_args=['-vcodec', 'h264'])		
	save = raw_input("Save video?(y/n)")
	if save == "y": video_ani.save(time.strftime("%Y%m%d_%H%M%S")+"vid.mp4", writer = FFwriter)
	plt.show()



