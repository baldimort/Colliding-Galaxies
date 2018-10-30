from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d, Axes3D
import scipy.integrate
import time
from functools import partial
import multiprocessing

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
	
	r_MW = np.array([0,0,0]) #position vecotrs of MW, Androm and star
	r_star = np.array((x,y,z))
	
	dist_MW = sum((r_star - r_MW)**2)**0.5 #distances from galaxies
	soft_dist = (dist_MW**2+soft_len**2)**0.5
		
	accns = -G * Mass_r(dist_MW,r_s_M,r_200_M) * (r_star - r_MW)/soft_dist**3 
	
	return (vx,vy,vz,accns[0],accns[1],accns[2])

def Solve_star_taylor(time_base,dt,galaxy_states,star):
	results = []
	results.append(star)
	for j,t in enumerate(time_base):
		results.append(Taylor(dt,t,results[-1],galaxy_states))
	return np.array(results[1:])

def Star_interactions(stars,galaxy_states,time_base):
	'''Loop through each star calculating each state'''
	star_solns = np.zeros((stars.shape[0],time_base.size,stars.shape[1]))
	dt = (time_base[-1]-time_base[0])/time_base.size
	
	for i,val in enumerate(stars):
		if (i%10)==0: print 'Star = %s' % i
		star_solns[i] = scipy.integrate.odeint(Star_state,val,time_base,(dt,galaxy_states),mxstep=5000)
	
	return star_solns
	
def Initalalise_stars_disk((x,y,z),(vx,vy,vz),N,radius,inc,rot,mass): #N is no. of stars, radius is of galaxy, inc is angle to x, rot is rotation around z
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
	
	ang_v = (G*Mass_r(star_r[:,0],r_s_M,r_200_M)/star_r[:,0])**0.5 #radial velocity of stars given radius - change to dark halo curve
	'''ang_v_orig = (G*mM/star_r[:,0])**0.5 

	plt.figure()
	plt.subplot(211)
	plt.plot(star_r[:,0],Mass_r(star_r[:,0],r_s_M,r_200_M),'go') #mass profile
	plt.subplot(212)
	plt.plot(star_r[:,0],ang_v,'ro',ms=2) # rotation curve
	plt.plot(star_r[:,0],ang_v_orig,'bo',ms=2)
	plt.show()'''

	star_states[:,3] = -ang_v*np.sin(star_r[:,1])*np.cos(inc)
	star_states[:,4] = ang_v*np.cos(star_r[:,1])
	star_states[:,5] = -ang_v*np.sin(star_r[:,1])*np.sin(inc)
	
	star_states[:,3:] = np.dot(rotation,star_states[:,3:].T).T #performs rotation on velocities
	
	for i,val in enumerate((x,y,z)): star_states[:,i] =  star_states[:,i] + val #moves stars to location of galactic core
	for i,val in enumerate((vx,vy,vz)): star_states[:,i+3] += val
	
	return star_states

'''def Initalalise_stars2((x,y,z),(vx,vy,vz),N,radius,inc,rot,mass):
	r, theta = np.random.rand(N) * radius,np.random.rand(N) * 2 * np.pi
	
	stars = np.zeros((6,N))
	stars[0] = r * np.cos(theta)
	stars[1] = r * np.sin(theta)
	stars[2] = 0
	
	v = (G*mass/r)**0.5
	
	stars[3] = -1*v*np.sin(theta)
	stars[4] = v*np.cos(theta)
	stars[5] = 0
	
	for i,val in enumerate((x,y,z,vx,vy,vz)): stars[i] += val
	
	return stars.T'''
	
def Taylor(a, t, (xn,yn,zn,xn_vel,yn_vel,zn_vel),gal_states):
	'''Performs the taylor method'''
	star_state = np.array(Star_state((xn,yn,zn,xn_vel,yn_vel,zn_vel),t,a,gal_states))

	xs = np.array([xn,yn,zn])
	vels = star_state[0:3]
	accs = star_state[3:]
	

	xs_new = xs + a * vels + a**2/2 * accs
	vs_new = vels + a * accs
	
	return [xs_new[0],xs_new[1],xs_new[2],vs_new[0],vs_new[1],vs_new[2]]
"""
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
"""

	
if __name__ == '__main__':
	pool = multiprocessing.Pool(processes=4)
	
	N_points = 1e4
	t_final = 5e9 #years
	
	t_final_secs = t_final * 365 * 24 * 3600 
	stars_per_gal = 5000
	
	time_base = np.linspace(0,t_final_secs,N_points)
	#setup timebase
	dt = (time_base[-1]-time_base[0])/time_base.size
	
	galaxy_states = np.zeros((time_base.size,6))
	#stationary galaxy at origin

	#MW stars
	stars = Initalalise_stars_disk([0,0,0],[0,0,0],stars_per_gal,RM,np.pi/4,0.4,mM)
	#star_solns = Star_interactions(stars,galaxy_states,time_base)	
	result = pool.map(partial(Solve_star_taylor,time_base,dt,galaxy_states),stars)	
	star_solns = np.array(result)
	
	pool.close()
	pool.join()
	
	rs = np.linspace(0,r_200_M,1000)
			
	star_solns = star_solns[:,:,0:3] / pc / 1e3
	galaxy_states[:,0:3] = galaxy_states[:,0:3]/pc/1e3
	galaxy_states[:,3:] = galaxy_states[:,3:]/pc/1e3
	#coversion to kPc

	fig = plt.figure()
	ax1 = fig.add_subplot(111,projection='3d')
	#ax2 = fig.add_subplot(212,projection='3d')
	ax1.set_xlabel('x (kpc)')
	ax1.set_ylabel('y (kpc)')
	ax1.set_zlabel('z (kpc)')
	
	ax1.set_aspect('equal', 'datalim')
	ax1.set_xlim((-50,50))
	ax1.set_ylim((-50,50))
	ax1.set_zlim((-50,50))
	ax1.view_init(30,210)
	
	'''MW_core, = ax1.plot([],[],[],'go',label='MW')
	#ax2.plot(star_solns[:,0,0],star_solns[:,0,1],star_solns[:,0,2],'go',ms=1)

	star_points = []
	for i in range(stars_per_gal):
		p, = ax1.plot([],[],[],'ro',ms=1)
		#ax1.plot(star_solns[i,:,0],star_solns[i,:,1],star_solns[i,:,2],'r--')
		star_points.append(p)
	
	ax1.legend()
	frame_factor = 1
	#time_dis = ax1.text(-110,25,-50,"Time = 0 Gyr")
	
	def update(j):
		i = int(j * frame_factor)
		MW_core.set_data(galaxy_states[i,0],galaxy_states[i,1])
		MW_core.set_3d_properties(galaxy_states[i,2])
		#time_dis.set_text("Time = %.2f Gyr" % (time_base[i]/(365 * 24 * 3600 * 1e9)))
		#ax1.view_init(30,360/time_base.size*i)
		for s,j in enumerate(star_points):
			j.set_data(star_solns[s,i,0],star_solns[s,i,1])
			j.set_3d_properties(star_solns[s,i,2])
		#ax1.relim()
	
	video_ani = animation.FuncAnimation(fig,update,frames=int(np.floor(time_base.size/frame_factor)),interval=1,repeat_delay=0)	'''
	plt.plot(star_solns[:,-1,0],star_solns[:,-1,1],star_solns[:,-1,2],'ro',ms=1)
	
	
	fig.show()
	
	
	
	
	
	'''plt.rcParams['animation.ffmpeg_path'] = r'/usr/bin/ffmpeg'
	FFwriter = animation.FFMpegWriter(fps=60, bitrate=-1, extra_args=['-vcodec', 'h264'])		
	save_ = raw_input("Save video?(y/n)")
	if save_ == "y": video_ani.save("vid4.mp4", writer = FFwriter)'''
	plt.show()

	"""
	

	
	

	
	ax1.plot(states[:,0],states[:,1],states[:,2],'r--',label='MW')
	ax1.plot(states[:,6],states[:,7],states[:,8],'b--',label='Androm')
	ax1.legend()
	
	

	print 'Tot time = %.1f s' % (time.time() - start_time)
	print time.strftime('%H:%M:%S')
	
	plt.show()"""
'''
Galaxy Initialisation
Conversion of integrators to 3D

'''


'''
f = open('points.txt','a')
g = open('energies.txt','a')


for i in range(2,31):
	rest of code...
	
	f.write(x)
	g.write(energy)
	
	
f.close()
g.close()

e$?jkv5k?du0L5uQ

'''
	











