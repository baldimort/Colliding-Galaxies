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


start_time = time.time()

G = 6.6726e-11
pc = 3.086e16
'''r_200_M = 176.99 # * 1e3 * pc
r_s_M = 47.67 # * 1e3 * pc

r_200_A = 233.08# * 1e3 * pc
r_s_A = 20.42 # * 1e3 * pc'''

r_200_M = 200 #* 1e3 * pc
r_s_M = 12.5 #* 1e3 * pc

r_200_A = 240 #* 1e3 * pc
r_s_A = 34.6 #* 1e3 * pc

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['mathtext.fontset'] = 'cm'


DIRECTORY = "20180227_022957_DATA"
AXESFONTSIZE = 16

MW_stars = np.load(DIRECTORY+"/MW_DATA.npy")
M31_stars = np.load(DIRECTORY+"/M31_DATA.npy")
GAL_cores = np.load(DIRECTORY+"/GAL_CORE_DATA.npy")
time_base = np.load(DIRECTORY+"/TIME_BASE_DATA.npy")
GAL_cores = GAL_cores[:time_base.size]



T_CUTOFF = float(sys.argv[1]) * 1e9*3600*24*365
if T_CUTOFF != 0:
	time_base = time_base[time_base<T_CUTOFF]
	MW_stars,M31_stars = MW_stars[:,:time_base.size,:],M31_stars[:,:time_base.size,:]
	GAL_cores = GAL_cores[:time_base.size]
else:
	pass


def SAVEFIG(message,filetag,f):
	a = raw_input(message)
	if a == 'y': f.savefig("../Figures/"+time.strftime("%Y%m%d_%H%M%S_")+filetag+"_%.0f_Gyrs.png"%(time_base[-1]/(3600*24*365*1e9)),dpi=600,format='png')




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




#ROTATION SPEED ANALYSIS
rel_MW = new_MW[:,-1,:] - GAL_cores[-1,0:6]
rel_M31 = new_M31[:,-1,:] - GAL_cores[-1,6:]

rel_MW = rel_MW[np.sum(rel_MW[:,0:3]**2,axis=1)**0.5<2*r_200_M]
rel_M31 = rel_M31[np.sum(rel_M31[:,0:3]**2,axis=1)**0.5<2*r_200_A]

'''perp_vels_MW = np.sum((rel_MW[:,3:] - (np.sum(rel_MW[:,3:]*rel_MW[:,0:3],axis=1) * rel_MW[:,0:3].T).T / np.sum(rel_MW[0:3]**2,axis=1))**2,axis=1)**0.5
perp_vels_M31 = np.sum((rel_M31[:,3:] - (np.sum(rel_M31[:,3:]*rel_M31[:,0:3],axis=1) * rel_M31[:,0:3].T).T / np.sum(rel_M31[0:3]**2,axis=1))**2,axis=1)**0.5'''


theta, phi = np.linspace(0,np.pi/2.,100),np.linspace(0,2*np.pi,100)

thetas,phis = np.meshgrid(theta,phi)

ns = np.array([np.cos(phis)*np.sin(thetas),np.sin(phis)*np.sin(thetas),np.cos(thetas)])
ns = np.array([ns[0].flatten(),ns[1].flatten(),ns[2].flatten()])

fig = plt.figure(figsize=(2*6.4*3/4,6.4*3/4))
axs = []
axs.append(fig.add_subplot(121))
axs.append(fig.add_subplot(122,sharey=axs[0]))

for e,galaxy in enumerate((rel_MW,rel_M31)):
	ax=axs[e]
	slit_gap = 4

	norm_vec=np.array([0,0,1])
	
	flat_gal = np.zeros([galaxy.shape[0],4])
	flat_gal[:,0] = galaxy[:,0]# * np.sum(galaxy[:,0:3]**2,axis=1)**0.5/np.sum(galaxy[:,0:2]**2,axis=1)**0.5
	flat_gal[:,1] = galaxy[:,1]# * np.sum(galaxy[:,0:3]**2,axis=1)**0.5/np.sum(galaxy[:,0:2]**2,axis=1)**0.5
	flat_gal[:,2] = galaxy[:,3]# * np.sum(galaxy[:,3:]**2,axis=1)**0.5/np.sum(galaxy[:,3:5]**2,axis=1)**0.5
	flat_gal[:,3] = galaxy[:,4]# * np.sum(galaxy[:,3:]**2,axis=1)**0.5/np.sum(galaxy[:,3:5]**2,axis=1)**0.5
	
	'''rs = np.sum(flat_gal[:,0:2]**2,axis=1)**0.5
	r_hat = (flat_gal[:,0:2].T/rs).T
	perp_v = np.sum((flat_gal[:,2:] - (np.sum(flat_gal[:,2:]*r_hat,axis=1)*r_hat.T).T)**2,axis=1)**0.5'''
	
	thetas = np.linspace(0,np.pi,100)

	v_maxs = np.array([0.01,0.02])
	stars_used = 0
	for theta in thetas:
		n_hat = np.array([np.cos(theta),np.sin(theta)])
		rdotn = np.sum(flat_gal[:,0:2]*n_hat,axis=1)
		rdotn = np.array([rdotn,rdotn])
		mod_perps = np.sum((rdotn.T*n_hat)**2,axis=1)**0.5
		stars = flat_gal[np.argwhere(mod_perps<slit_gap/2.)]
		stars = stars[:,0,:]
		
		perp_vectors = np.zeros(stars[:,0:2].shape)
		perp_vectors[:,0] = np.cos(np.arctan(stars[:,1]/stars[:,0]) + np.pi/2)
		perp_vectors[:,1] = np.sin(np.arctan(stars[:,1]/stars[:,0]) + np.pi/2)
		
		v_test = np.sum(stars[:,2:]*perp_vectors,axis=1) / np.sum(perp_vectors[:,0:2]**2,axis=1)**0.5
		
		#v_test = np.sum((stars[:,2:] - (np.sum(stars[:,2:]*stars[:,0:2],axis=1) * stars[:,0:2].T).T / np.sum(stars[0:2]**2,axis=1))**2,axis=1)**0.5
		#v_test = np.sum(stars[:,2:]**2,axis=1)**0.5
		
		if v_test.max()-v_test.min() > v_maxs.max()-v_maxs.min():
			v_maxs = v_test.copy()
			stars_used = stars.copy()
	
	rs = np.sum(stars_used[:,0:2]**2,axis=1)**0.5
	
	v_maxs = v_maxs[np.argsort(rs)]
	mod_vs = abs(v_maxs)
	rs = np.sort(rs)
	
	if galaxy[0,0] == rel_MW[0,0]: v_model = (G*Mass_r(rs*pc*1e3,r_s_M*pc*1e3,r_200_M*pc*1e3)/(rs*pc*1e3))**0.5
	elif galaxy[0,0] == rel_M31[0,0]: v_model = (G*Mass_r(rs*pc*1e3,r_s_A*pc*1e3,r_200_A*pc*1e3)/(rs*pc*1e3))**0.5
	
	bins = np.linspace(0,rs.max(),15)
	inds = np.digitize(rs,bins)
	
	bin_means = np.array([mod_vs[inds == i].mean() for i in range(1, len(bins))])
	bin_means[np.isnan(bin_means)] = 0
	
	means_3s_ub,means_3s_lb,means_1s_ub,means_1s_lb = [],[],[],[]
	
	for i in range(1, len(bins)):
		try:
			means_3s_ub.append(np.percentile(np.array(mod_vs[inds == i]),99.7))
			means_3s_lb.append(np.percentile(np.array(mod_vs[inds == i]),100-99.7))
			means_1s_ub.append(np.percentile(np.array(mod_vs[inds == i]),50+34.1))
			means_1s_lb.append(np.percentile(np.array(mod_vs[inds == i]),50-34.1))
		except Exception: 
			for i in (means_3s_ub,means_3s_lb,means_1s_ub,means_1s_lb): i.append(0)
		
	means_3s_ub,means_3s_lb,means_1s_ub,means_1s_lb = np.array(means_3s_ub)/1e3,np.array(means_3s_lb)/1e3,np.array(means_1s_ub)/1e3,np.array(means_1s_lb)/1e3


	
	ax.plot(rs[np.argwhere(v_maxs>0)],v_maxs[np.argwhere(v_maxs>0)]/1e3,'o',ms=1)
	ax.plot(rs[np.argwhere(v_maxs<0)],abs(v_maxs[np.argwhere(v_maxs<0)])/1e3,'o',ms=1)
	ax.plot((bins[1:] + bins[:-1])/2,bin_means/1e3,'r-o',ms=3,label='Mean')
	ax.plot(rs,v_model/1e3,'--',label='Model rotation curve')
	
	ax.fill_between((bins[1:] + bins[:-1])/2,means_3s_lb,means_3s_ub,alpha=0.15,color='gray',linestyle='None',edgecolor='None')
	
	ax.fill_between((bins[1:] + bins[:-1])/2,means_1s_lb,means_1s_ub,alpha=0.3,color='gray',linestyle='None',edgecolor='None')
	
	
	ax.legend(fontsize=12)
	ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)

	
	
	#if galaxy[0,0] == rel_MW[0,0]: SAVEFIG('save MW? (y/n)','MW_ROT_CURVE',fig)
	#elif galaxy[0,0] == rel_M31[0,0]: SAVEFIG('save M31? (y/n)','M31_ROT_CURVE',fig)

axs[0].set_xlabel(r'$r$ (kpc)',fontsize=12)
axs[1].set_xlabel(r'$r$ (kpc)',fontsize=12)
axs[0].set_ylabel(r'$v_{\bot}\ (kms^{-1})$',fontsize=12)

for ax in axs:
	ax.set_xlim([0,ax.get_xlim()[1]])
	ax.set_ylim([0,ax.get_ylim()[1]])
plt.setp([axs[1].get_yticklabels()],visible=False)
plt.tight_layout()
fig.subplots_adjust(wspace=0)
SAVEFIG('save fig? (y/n)','ROT_CURVE',fig)
plt.show()

'''

ax3.scatter(np.sqrt(np.sum(rel_MW[:,0:3]**2,axis=1))/r_200_M,perp_vels_MW/1e3,s=0.3,color='red',label='MW',alpha=0.8)
ax3a.scatter(np.sqrt(np.sum(rel_M31[:,0:3]**2,axis=1))/r_200_A,perp_vels_M31/1e3,s=0.3,color='green',label='M31',alpha=0.8)
ax3.legend()
ax3a.legend()
ax3a.set_xlabel(r'$r/r_{200}$')
ax3.set_ylabel(r'$v_{\bot,\ MW}\ (kms^{-1})$')
ax3a.set_ylabel(r'$v_{\bot,\ M31}\ (kms^{-1})$')
plt.tight_layout()







plt.show()'''






