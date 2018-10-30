from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate
import time
import warnings

warnings.filterwarnings("ignore")

start_time = time.time()

mE = 5.9742e24 #kg - mass of the earth
mM = 7.35e22 #kg - mass of the moon
G = 6.6726e-11 #Nm^2kg^-2 - Newtons universal gravitational constant
d = 3.844e8 #m - Earth-Moon distance

def period_orbit():
	'''Calculates period of the earth/moon orbits'''
	return np.sqrt(4*np.pi**2*d**3/(G*(mM+mE)))

def radius_e():
	'''Calculates radius of earth orbit'''
	return d*mM/(mM+mE)

def radius_m():
	'''Calculates radius of moon orbit'''
	return d*mE/(mM+mE)
	
def earth_pos(t):
	'''Calculates the position of earth in cartesian coordinates'''
	x = radius_e()*np.cos(t*2*np.pi/period_orbit())
	y = radius_e()*np.sin(t*2*np.pi/period_orbit())
	return (x,y)
	
def moon_pos(t):
	'''Calculates the position of moon in cartesian coordinates'''
	x = -1*radius_m()*np.cos(t*2*np.pi/period_orbit())
	y = -1*radius_m()*np.sin(t*2*np.pi/period_orbit())
	return (x,y)
	
def f_rocket((x,y,vx,vy),t):
	'''Calculates the acceleration of the rocket due to the gravitational force of the Earth and Moon'''
	x_earth, y_earth = earth_pos(t)
	x_moon, y_moon = moon_pos(t)
	
	dist_earth = np.sqrt((x - x_earth)**2 + (y - y_earth)**2)
	dist_moon = np.sqrt((x - x_moon)**2 + (y - y_moon)**2)
	
	x_accn = -G * mE * (x - x_earth)/dist_earth**3 - G * mM * (x - x_moon)/dist_moon**3
	y_accn = -G * mE * (y - y_earth)/dist_earth**3 - G * mM * (y - y_moon)/dist_moon**3
	
	return (vx,vy,x_accn,y_accn)
			
def Taylor(a, t, (xn,yn,xn_vel,yn_vel)):
	'''Performs the taylor method'''
	ban,ana,x_accel, y_accel = f_rocket([xn,yn,xn_vel,yn_vel],t)
	
	x_new = xn + a * xn_vel + a**2/2 * x_accel
	y_new = yn + a * yn_vel + a**2/2 * y_accel
	x_vel_new = xn_vel + a * x_accel
	y_vel_new = yn_vel + a * y_accel
	
	return [x_new,y_new,x_vel_new,y_vel_new]

def RK(a, t, (xn,yn,xn_vel,yn_vel)):
	'''Performs the RK4 method'''
	ban,ana,x_accel, y_accel = f_rocket([xn,yn,xn_vel,yn_vel],t)
	new_states = []
	
	x, x_v, x_a = np.array([xn,yn]), np.array([xn_vel,yn_vel]), np.array([x_accel,y_accel])
	
	z1 = x +a/2 * x_v
	z1_v = x_v + a/2 * x_a
	z1_a = np.array(f_rocket((z1[0],z1[1],z1_v[0],z1_v[1]),t)[2:]) #operates on both dimensions simultaneously using arrays	
	
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

def rocket_path((x0,y0,x0_vel,y0_vel),(t0,t_final,dt),method=Taylor):
	'''Calculates the path of the rocket'''
	time_base = np.arange(t0,t_final,dt)
	#dt = (t_final - t0)/N_points
	states = [[x0,y0,x0_vel,y0_vel]]
	
	for i,t in enumerate(time_base):
		result = method(dt,t,states[i])
		states.append(result)
		if i%5000==0: print 'Step = %s' % i
	return (time_base,np.array(states))

def L2_aprox_moon():
	return radius_m() * (mM/(3.*mE))**(1/3)
	
if __name__ == '__main__':
	
	
	dt = 10
		
	#rocket_initial_cons = [r_L2,0,0,2*np.pi*r_L2/period_orbit()]
	#time_base, states1 = rocket_path(rocket_initial_cons, (0,period_orbit()*1,dt))

	time_base2 = np.arange(0,period_orbit(),dt)
	#states2 = scipy.integrate.odeint(f_rocket,rocket_initial_cons,time_base2)
	
	moon_rock_dist_chgs = []
	moon_rock_dist_chgs_tay = []
	mods = np.linspace(1.06151,1.06152,150)
	mods2 = np.linspace(1.06151,1.06152,150)
	
	for i,val in enumerate(mods):
		if (i%10)==0: print "Step = %s" % i
		
		r_L2 = -1*val*L2_aprox_moon() + moon_pos(0)[0]
		initial_cons = [r_L2,0,0,2*np.pi*r_L2/period_orbit()]
		states = scipy.integrate.odeint(f_rocket,initial_cons,time_base2)
		#appls, states = rocket_path(initial_cons, (0,period_orbit()*1,dt))
		final_dist = np.sqrt(np.sum((states[-1,0:2] - np.array(moon_pos(period_orbit())))**2))
		change = (final_dist - val*L2_aprox_moon())/(val*L2_aprox_moon())
		moon_rock_dist_chgs.append(change)

	'''for i,val in enumerate(mods2):
		if (i%1)==0: print "Step = %s" % i
		
		r_L2 = -1*val*L2_aprox_moon() + moon_pos(0)[0]
		initial_cons = [r_L2,0,0,2*np.pi*r_L2/period_orbit()]
		tb, tay_states = rocket_path(initial_cons, (0,period_orbit()*1,dt), Taylor)
		final_dist = np.sqrt(np.sum((tay_states[-1,0:2] - np.array(moon_pos(period_orbit())))**2))
		change = (final_dist - val*L2_aprox_moon())/(val*L2_aprox_moon())
		moon_rock_dist_chgs_tay.append(change)'''
	
	chgs_scipy = abs(np.array(moon_rock_dist_chgs))
	chgs_tay = abs(np.array(moon_rock_dist_chgs_tay))
	
	name = 'L2_point' + time.strftime('%Y%m%j_%H%M%S')+'.txt'
	f = open(name,'w')
	L2_point_scipy = mods[chgs_scipy.argmin()]*L2_aprox_moon()
	f.write(str(L2_point_scipy))
	f.close()
	
	print "The best modifier = %s with a relative change in position = %.5f%% using scipy" % (mods[chgs_scipy.argmin()],chgs_scipy.min())
	print "L2 Legrange point = %.1f m from moon" % (mods[chgs_scipy.argmin()]*L2_aprox_moon())
	
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	ax1.plot(mods*L2_aprox_moon()/1000.,chgs_scipy,'k--')
	ax1.set_xlabel('Initial Distance from moon (km)')
	ax1.set_ylabel('% Change in distance from moon')
		
	r_L2 = -1*mods[chgs_scipy.argmin()]*L2_aprox_moon() + moon_pos(0)[0]
	initial_cons = [r_L2,0,0,2*np.pi*r_L2/period_orbit()]
	states = scipy.integrate.odeint(f_rocket,initial_cons,time_base2)
	tb, rk_states = rocket_path(initial_cons, (0,period_orbit()*1,dt),RK)
	tb2, tay_states = rocket_path(initial_cons, (0,period_orbit()*1,dt), Taylor)
	
	X0,Y0 = earth_pos(time_base2)
	X1,Y1 = moon_pos(time_base2)
	
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.set_aspect('equal', 'datalim')
	
	earth, = ax1.plot([],[],'bo')
	moon, = ax1.plot([],[],'ko')
	#rocket1, = ax1.plot([],[],'ro')
	rocket2, = ax1.plot([],[],'go')
	
	ax1.plot(X0,Y0,'b--')
	ax1.plot(X1,Y1,'k--')
	
	#ax1.plot(states1[:,0],states1[:,1],'r--')
	ax1.plot(states[:,0],states[:,1],'g-', label='Scipy')
	ax1.plot(rk_states[:,0],rk_states[:,1],'r--',label='RK4')
	ax1.plot(tay_states[:,0],tay_states[:,1],'b--',label='Taylor')
	
	plt.legend()
	
	frame_factor = 5e2
	
	#plt.show(block=False)
	#ax1.set_xlim([X1.min(),X1.max()])
	#ax1.set_ylim([Y1.min(),Y1.max()])
	def update(j):
		i=int(j*frame_factor)
		earth.set_data(X0[i],Y0[i])
		moon.set_data(X1[i],Y1[i])
		#rocket1.set_data(states1[i,0],states1[i,1])
		rocket2.set_data(states[i,0],states[i,1])
		
		#ax1.relim()
		#ax1.autoscale_view()
		
	
	video_ani = animation.FuncAnimation(fig,update,frames=int(np.floor(X0.size/frame_factor)),
					interval=1,repeat_delay=0)	

	print 'Tot time = %.1f s' % (time.time() - start_time)
	plt.show()
	
'''
variable: time integar units of days(?)
variable: rocket_path numpy array (x,y,x_vel,y_vel)
DONE function: earth_pos(t) return (x_earth,y_earth)
DONE function: moon_pos(t) return (x_moon,y_moon)
DONE function: period_orbit() return float in seconds
DONE function: radius_earth_orbit() return float in m
DONE function: radius_moon_orbit() return float in m
DONE function: accn_rocket(x,y) return (x_accel,y_accel)
function: Taylor(a,x0,y0,x0_vel,y0_vel) return (x1,y1,x1_vel,y1_vel)
function: RK(a,x0,y0,x0_vel,y0_vel) return (x1,y1,x1_vel,y1_vel)
function: solve_path_rocket(initialconditions,method) return numpy array (x,y,x_vel,y_vel)
'''