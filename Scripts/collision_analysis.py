from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d, Axes3D
import scipy.integrate,scipy.optimize,time,sys,multiprocessing
from functools import partial
from NFW import DM_density,Mass_r
from scipy.stats import norm
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['mathtext.fontset'] = 'cm'


start_time = time.time()

pc = 3.086e16
r_200_M = 200# * 1e3 * pc
r_s_M = 12.5# * 1e3 * pc

r_200_A = 240# * 1e3 * pc
r_s_A = 34.6# * 1e3 * pc


DIRECTORY = "20180227_022957_DATA"
AXESFONTSIZE = 12

MW_stars = np.load(DIRECTORY+"/MW_DATA.npy")
M31_stars = np.load(DIRECTORY+"/M31_DATA.npy")
GAL_cores = np.load(DIRECTORY+"/GAL_CORE_DATA.npy")
time_base = np.load(DIRECTORY+"/TIME_BASE_DATA.npy")
GAL_cores = GAL_cores[:time_base.size]

f = open("LOGS/"+time.strftime("%Y%m%d_%H%M%S")+'_COLLISION_LOG.txt','a')
f.write("************************************************\
**************************************\n\nCOLLISION LOG\nDATA SET: {0}\nDATE/TIME: {1}\n\n\
*****************************************\
*********************************************\n\n\n".format(DIRECTORY,time.strftime("%Y%m%d_%H%M%S")))

T_CUTOFF = float(sys.argv[1]) * 1e9*3600*24*365
if T_CUTOFF != 0:
	time_base = time_base[time_base<T_CUTOFF]
	MW_stars,M31_stars = MW_stars[:,:time_base.size,:],M31_stars[:,:time_base.size,:]
	GAL_cores = GAL_cores[:time_base.size]
else:
	pass


def SAVEFIG(message,filetag):
	a = raw_input(message)
	if a == 'y': plt.savefig("../Figures/"+time.strftime("%Y%m%d_%H%M%S_")+filetag+"_%.0f Gyrs.png"%(time_base[-1]/(3600*24*365*1e9)),dpi=600,format='png')


'''
fig = plt.figure(dpi=300)
ax1 = fig.add_subplot(111,projection='3d',aspect='equal')
ax1.set_xlabel('x (kpc)')
ax1.set_ylabel('y (kpc)')
ax1.set_zlabel('z (kpc)')

ax1.set_xlim((min([M31_stars[:,:,0].min(),MW_stars[:,:,0].min()]),max([M31_stars[:,:,0].max(),MW_stars[:,:,0].max()])))
ax1.set_ylim((min([M31_stars[:,:,1].min(),MW_stars[:,:,1].min()]),max([M31_stars[:,:,1].max(),MW_stars[:,:,1].max()])))
ax1.set_zlim((min([M31_stars[:,:,2].min(),MW_stars[:,:,2].min()]),max([M31_stars[:,:,2].max(),MW_stars[:,:,2].max()])))


MW_core, = ax1.plot([],[],[],'go',label='MW',ms=2)
Androm_core, = ax1.plot([],[],[],'bo',label='Andromeda',ms=2)
star_points_A = []
star_points_M = []

for i in range(MW_stars.shape[0]):
	p, = ax1.plot([],[],[],'o',color='#9F14E9',ms=0.1)
	q, = ax1.plot([],[],[],'o',color='#b72424',ms=0.1)
	star_points_A.append(p)
	star_points_M.append(q)


frame_factor = 1e0


def update(j):
	i = int(j * frame_factor)
	MW_core.set_data(GAL_cores[i,0],GAL_cores[i,1])
	MW_core.set_3d_properties(GAL_cores[i,2])
	Androm_core.set_data(GAL_cores[i,6],GAL_cores[i,7])
	Androm_core.set_3d_properties(GAL_cores[i,8])
	for s,k in enumerate(star_points_A):
		k.set_data(M31_stars[s,i,0],M31_stars[s,i,1])
		k.set_3d_properties(M31_stars[s,i,2])
	for s,k in enumerate(star_points_M):
		k.set_data(MW_stars[s,i,0],MW_stars[s,i,1])
		k.set_3d_properties(MW_stars[s,i,2])
	#ax1.relim()
	 
#fig.show()
video_ani = animation.FuncAnimation(fig,update,frames=int(np.floor(time_base.size/frame_factor)),interval=1,repeat_delay=0)	


plt.rcParams['animation.ffmpeg_path'] = r'/usr/bin/ffmpeg'
FFwriter = animation.FFMpegWriter(fps=25, bitrate=-1, extra_args=['-vcodec', 'h264'])		
save = raw_input("Save video?(y/n)")
if save == "y": video_ani.save("VIDEOS/"+time.strftime("%Y%m%d_%H%M%S")+"vid.mp4", writer = FFwriter)
plt.close()'''

print "Purple = MW, Orange = M31"

#PLOT OF FINAL GALAXIES
fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection='3d',aspect='equal')
ax2.set_xlabel('x (kpc)')
ax2.set_ylabel('y (kpc)')
ax2.set_zlabel('z (kpc)')

for set in (ax2.set_xlim,ax2.set_ylim,ax2.set_zlim): set(-100,100)

ax2.scatter(MW_stars[:,-1,0],MW_stars[:,-1,1],MW_stars[:,-1,2],color='#b72424',s=0.1,alpha=1)
ax2.scatter(M31_stars[:,-1,0],M31_stars[:,-1,1],M31_stars[:,-1,2],color='#9F14E9',s=0.1,alpha=1)
#ax2.scatter(GAL_cores[-1,0],GAL_cores[-1,1],GAL_cores[-1,2],color='red',label='MW')
#ax2.scatter(GAL_cores[-1,6],GAL_cores[-1,7],GAL_cores[-1,8],color='yellow',label='M31')
#ax2.plot(GAL_cores[:,0],GAL_cores[:,1],GAL_cores[:,2],'r--')
#ax2.plot(GAL_cores[:,6],GAL_cores[:,7],GAL_cores[:,8],'y--')
#ax2.legend()
plt.tight_layout()
plt.axis('off')
ax2.view_init(elev=30,azim=-50)
#ax2.dist=6

SAVEFIG("Save galaxy fig? (y/n)","GALAXY_FIG")




#STAR TRASNFER ANALYSIS
MW_core_position = GAL_cores[-1,0:3]
M31_core_position = GAL_cores[-1,6:9]
new_MW_stars = []
new_M31_stars = []

for a,i in enumerate(MW_stars[:,-1,:]):
	if np.sum((i[0:3]-MW_core_position)**2) < np.sum((i[0:3]-M31_core_position)**2): new_MW_stars.append(MW_stars[a])
	else: new_M31_stars.append(MW_stars[a])

for a,i in enumerate(M31_stars[:,-1,:]):
	if np.sum((i[0:3]-MW_core_position)**2)<np.sum((i[0:3]-M31_core_position)**2): new_MW_stars.append(M31_stars[a])
	else: new_M31_stars.append(M31_stars[a])
	
new_M31 = np.array(new_M31_stars)
new_MW = np.array(new_MW_stars)

f.write("MW Initial Stars,MW final Stars\n%s,%s\n\n" % (MW_stars.shape[0],new_MW.shape[0])) 
f.write("M31 Initial Stars,M31 final Stars\n%s,%s\n\n"% (M31_stars.shape[0],new_M31.shape[0]))

rel_MW = new_MW[:,-1,:] - GAL_cores[-1,0:6]
rel_M31 = new_M31[:,-1,:] - GAL_cores[-1,6:]


'''
#ROTATION SPEED ANALYSIS
fig = plt.figure()
ax3 = fig.add_subplot(211)
ax3a = fig.add_subplot(212,sharex=ax3)

rel_MW = rel_MW[np.sum(rel_MW[:,0:3]**2,axis=1)**0.5<2*r_200_M]
rel_M31 = rel_M31[np.sum(rel_M31[:,0:3]**2,axis=1)**0.5<2*r_200_A]

perp_vels_MW = np.sum((rel_MW[:,3:] - (np.sum(rel_MW[:,3:]*rel_MW[:,0:3],axis=1) * rel_MW[:,0:3].T).T / np.sum(rel_MW[0:3]**2,axis=1))**2,axis=1)**0.5
perp_vels_M31 = np.sum((rel_M31[:,3:] - (np.sum(rel_M31[:,3:]*rel_M31[:,0:3],axis=1) * rel_M31[:,0:3].T).T / np.sum(rel_M31[0:3]**2,axis=1))**2,axis=1)**0.5

ax3.scatter(np.sqrt(np.sum(rel_MW[:,0:3]**2,axis=1))/r_200_M,perp_vels_MW/1e3,s=0.3,color='red',label='MW',alpha=0.8)
ax3a.scatter(np.sqrt(np.sum(rel_M31[:,0:3]**2,axis=1))/r_200_A,perp_vels_M31/1e3,s=0.3,color='green',label='M31',alpha=0.8)
ax3.legend()
ax3a.legend()
ax3a.set_xlabel(r'$r/r_{200}$')
ax3.set_ylabel(r'$v_{\bot,\ MW}\ (kms^{-1})$')
ax3a.set_ylabel(r'$v_{\bot,\ M31}\ (kms^{-1})$')
plt.tight_layout()
'''


MW_n_stars = []
M31_n_stars = []

#Mass change analysis
for e,t in enumerate(time_base):
	MW_transferred = np.sum(np.sum((MW_stars[:,e,0:3]-GAL_cores[e,0:3])**2,axis=1) > np.sum((MW_stars[:,e,0:3]-GAL_cores[e,6:9])**2,axis=1))
	M31_transferred = np.sum(np.sum((M31_stars[:,e,0:3]-GAL_cores[e,0:3])**2,axis=1) < np.sum((M31_stars[:,e,0:3]-GAL_cores[e,6:9])**2,axis=1))
	
	MW_n_stars.append(MW_stars.shape[0]-MW_transferred+M31_transferred)
	M31_n_stars.append(M31_stars.shape[0]+MW_transferred-M31_transferred)
	




#HISTOGRAMS
#fig=plt.figure()
vels_MW = np.sqrt(rel_MW[:,0]**2+rel_MW[:,1]**2+rel_MW[:,2]**2)
vels_M31 = np.sqrt(rel_M31[:,0]**2+rel_M31[:,1]**2+rel_M31[:,2]**2)
#ax5 = fig.add_subplot(211)
#plt.hist(vels_MW,100,color='red',label='MW')
#ax6 = fig.add_subplot(212)
#plt.hist(vels_M31,100,color='green',label='M31')'''

#for ax in (ax5,ax6):
	#ax.set_xlabel(r'Velocity $(kms^{-1})$')
	#ax.set_ylabel('N')
	#ax.legend()

mean_MW = vels_MW.mean()
vel_disper_MW = vels_MW.std()
mean_M31 = vels_M31.mean()
vel_disper_M31 = vels_M31.std()

f.write("MW mean velocity (km/s),Velocity Dispersion\n%s,%s\n\n" % (mean_MW,vel_disper_MW))
f.write("M31 mean velocity (km/s),Velocity Dispersion\n%s,%s\n\n" % (mean_M31,vel_disper_M31))
#ax5.axvline(mean_MW)
#ax6.axvline(mean_M31)
#plt.tight_layout()



# N within r_200
fig = plt.figure(figsize=(6.4*3/4,6.4*3/4))
ax6 = fig.add_subplot(111)
MW_200_stars = []
M31_200_stars = []
for e,t in enumerate(time_base):
	MW_200 = np.sum(np.sum((MW_stars[:,e,0:3]-GAL_cores[e,0:3])**2,axis=1)**0.5 < r_200_M)
	MW_200 += np.sum(np.sum((M31_stars[:,e,0:3]-GAL_cores[e,0:3])**2,axis=1)**0.5 < r_200_M)
	M31_200 = np.sum(np.sum((M31_stars[:,e,0:3]-GAL_cores[e,6:9])**2,axis=1)**0.5 < r_200_A)
	M31_200 += np.sum(np.sum((MW_stars[:,e,0:3]-GAL_cores[e,6:9])**2,axis=1)**0.5 < r_200_A)
	MW_200_stars.append(MW_200)
	M31_200_stars.append(M31_200)
	

plt.plot(time_base/(365*24*3600*1e9),np.array(MW_200_stars)/MW_stars.shape[0],'r--',label='MW')
plt.plot(time_base/(365*24*3600*1e9),np.array(M31_200_stars)/M31_stars.shape[0],'g-',label='M31')
ax6.legend(fontsize=AXESFONTSIZE)
ax6.set_xlabel(r'$t$ (Gyr)',fontsize=AXESFONTSIZE)
ax6.set_ylabel(r'$N_{200}/N_{intial}$',fontsize=AXESFONTSIZE)
ax6.set_xlim([0,20])
ax6.set_ylim([0.4,2])
ax6.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
plt.tight_layout()
SAVEFIG("Save N within r_200 fig? (y/n)","N_IN_200_FIG")


'''
#Number density

fig = plt.figure()
ax8 = fig.add_subplot(212)
ax7 = fig.add_subplot(211,sharex=ax8)

new_MW_rs = np.sum((new_MW[:,-1,0:3]-GAL_cores[-1,0:3])**2,axis=1)**0.5
new_M31_rs = np.sum((new_M31[:,-1,0:3]-GAL_cores[-1,6:9])**2,axis=1)**0.5

new_MW_rs = new_MW_rs[new_MW_rs<r_200_M]
new_M31_rs = new_M31_rs[new_M31_rs<r_200_A]




r_0_200_M = np.logspace(np.log10(0.016*r_200_M),np.log10(new_MW_rs.max()*1.1),1e3)
r_0_200_A = np.logspace(np.log10(0.016*r_200_A),np.log10(new_M31_rs.max()*1.1),1e3)


bins_MW = np.logspace(np.log10(0.02*r_200_M),np.log10(new_MW_rs.max()),20)
hist_MW = np.histogram(new_MW_rs,bins=bins_MW)
counts_MW = hist_MW[0]
counts_MW[counts_MW==0] = 1
densitys_MW = counts_MW / counts_MW.sum() * bins_MW[-1]**3 / (bins_MW[1:]**3 - bins_MW[:-1]**3)



bins_M31 = np.logspace(np.log10(0.02*r_200_A),np.log10(new_M31_rs.max()),20)
hist_M31 = np.histogram(new_M31_rs,bins=bins_M31)
counts_M31 = hist_M31[0]
counts_M31[counts_M31==0] = 1
densitys_M31 = counts_M31 / counts_M31.sum() * bins_M31[-1]**3 / (bins_M31[1:]**3 - bins_M31[:-1]**3)


def Fit_NFW(r,rs,r200):
	return np.log10(DM_density(r*1e3*pc,rs*1e3*pc,r200*1e3*pc)* 4/3*np.pi*(r200*1e3*pc)**3 / Mass_r(r200*1e3*pc,rs*1e3*pc,r200*1e3*pc))

def Normal(x,m,s):
	return 1./(2*np.pi*s**2)**0.5 * np.exp(-(x-m)**2/(2*s**2))
	
def Bootstrap(star_rs,bins,N,galaxy):
	r_ss, r_200s = [], []
	i=0
	tot_stars = star_rs.shape[0]
	densitys_sample = []
	
	while i<N:
		
		indices = np.random.choice(tot_stars,tot_stars)
		star_sample = []
		star_sample = star_rs[indices]
				
		star_sample = np.array(star_sample)
		#bins = np.logspace(np.log10(star_sample.min()),np.log10(star_sample.max()),20)
		counts = np.histogram(star_sample,bins=bins)[0]
		counts[counts==0] = 1
		densitys = counts / counts.sum() * bins[-1]**3 / (bins[1:]**3 - bins[:-1]**3)
		densitys_sample.append(densitys)
		a, b = scipy.optimize.curve_fit(Fit_NFW,(bins[1:]+bins[:-1])/2,np.log10(densitys))[0]
		r_ss.append(a)
		r_200s.append(b)
		i += 1
			
		#except Exception: pass
		
		
	densitys = np.array(densitys_sample)
	
	med_densitys = np.nanmedian(densitys,axis=0)
	med_densitys_3s_ub = np.percentile(densitys,99.7,axis=0)
	med_densitys_3s_lb = np.percentile(densitys,100-99.7,axis=0)
	med_densitys_1s_ub = np.percentile(densitys,50+34.1,axis=0)
	med_densitys_1s_lb = np.percentile(densitys,50-34.1,axis=0)
	
	#a,b = scipy.optimize.curve_fit(Fit_NFW,(bins[1:]+bins[:-1])/2,np.log10(med_densitys))[0]
	
	r_ss = np.array(r_ss)
	ub_s = np.percentile(r_ss,50+34.1)
	lb_s = np.percentile(r_ss,50-34.1)
	median_s = np.median(r_ss)
	f.write('%s Bootstrapping fit (N=%s)\nr_s median,r_s upperbound,r_s lowerbound\n%.2f,%.2f,%.2f\n' % (str(galaxy),N,median_s,ub_s,lb_s))
	
	r_200s = np.array(r_200s)
	ub_200 = np.percentile(r_200s,50+34.1)
	lb_200 = np.percentile(r_200s,50-34.1)
	#mean,sigma = norm.fit(r_200s) 
	median_200 = np.median(r_200s)
	f.write('r_200 median,r_200 upper bound,r_200 lower bound\n%.2f,%.2f,%.2f\n\n' % (median_200,ub_200,lb_200))
	
	#f.write('\nMedian r_s, median r_200\n%s,%s\n\n'%(a,b))
	
	return (med_densitys,0,0,median_s,median_200,med_densitys_3s_ub,med_densitys_3s_lb,med_densitys_1s_ub,med_densitys_1s_lb)
	


if sys.argv[2] == 'y' or sys.argv[2] == 'Y': 
	out_MW = Bootstrap(new_MW_rs,bins_MW,1e3,'MW')
	out_M31 = Bootstrap(new_M31_rs,bins_M31,1e3,'M31')

	model_MW_3 = DM_density(r_0_200_M*1e3*pc,out_MW[3]*1e3*pc,out_MW[4]*1e3*pc) * 4/3*np.pi*(out_MW[4]*1e3*pc)**3 / Mass_r(out_MW[4]*1e3*pc,out_MW[3]*1e3*pc,out_MW[4]*1e3*pc) 
	ax7.plot((bins_MW[1:]+bins_MW[:-1])/2/r_200_M,out_MW[0],'o-',label='MW Median Number Densities',ms=2)
	params_MW = np.around(out_MW[3:5])
	ax7.plot(r_0_200_M/r_200_M,model_MW_3,'r--',label=r'NFW Fit, $\rho_{s}=%.0f$ kpc $\rho_{200}=%.0f$ kpc'%(params_MW[0],params_MW[1]))
	ax7.fill_between((bins_MW[1:]+bins_MW[:-1])/2/r_200_M,out_MW[5],out_MW[6],alpha=0.2,color='gray',linestyle='None',edgecolor='None')
	ax7.fill_between((bins_MW[1:]+bins_MW[:-1])/2/r_200_M,out_MW[7],out_MW[8],alpha=0.4,color='gray',linestyle='None',edgecolor='None')
	
	model_M31_3 = DM_density(r_0_200_A*1e3*pc,out_M31[3]*1e3*pc,out_M31[4]*1e3*pc) * 4/3*np.pi*(out_M31[4]*1e3*pc)**3 / Mass_r(out_M31[4]*1e3*pc,out_M31[3]*1e3*pc,out_M31[4]*1e3*pc) 
	ax8.plot((bins_M31[1:]+bins_M31[:-1])/2/r_200_A,out_M31[0],'o-',label='M31 Median Number Densities',ms=2)
	params_M31 = np.around(out_M31[3:5])
	ax8.plot(r_0_200_A/r_200_A,model_M31_3,'g--',label=r'NFW Fit, $\rho_{s}=%.0f$ kpc $\rho_{200}=%.0f$ kpc'%(params_M31[0],params_M31[1]))
	ax8.fill_between((bins_M31[1:]+bins_M31[:-1])/2/r_200_A,out_M31[5],out_M31[6],alpha=0.2,color='gray',linestyle='None',edgecolor='None')
	ax8.fill_between((bins_M31[1:]+bins_M31[:-1])/2/r_200_A,out_M31[7],out_M31[8],alpha=0.4,color='gray',linestyle='None',edgecolor='None')
	

plt.loglog()
ax8.set_yscale('log')
ax8.legend(loc=3)
ax7.legend(loc=3)
ax8.set_xlabel(r'$\frac{r}{R_{200}}$',fontsize=AXESFONTSIZE)
ax8.set_ylabel(r'$\frac{n_{M31}(r)}{<n_{M31}>}$',fontsize=AXESFONTSIZE)
ax7.set_ylabel(r'$\frac{n_{MW}(r)}{<n_{MW}>}$',fontsize=AXESFONTSIZE)
plt.setp([ax7.get_xticklabels()],visible=False)
fig.subplots_adjust(hspace=0)
plt.tight_layout()
SAVEFIG("Save number density fig? (y/n)","n_DENSITY_FIG")
'''


f.close()
plt.show()






'''zotero citation manager




calculate bins -> calculate densitys of population -> fit to NFW

loop 10^5 times:
	-> take same stars -> sample 1000 -> calculate densitys from sample
	-> fit to NFW profile -> store r_s and r_200

	
fit stored r_s and r_200 values to normal distribution, take mean/median/sigma




confidence intervals not normal np.percentile on bins and params
plt.fill_between








'''
