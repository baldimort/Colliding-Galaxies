from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d, Axes3D
import scipy.integrate,scipy.optimize,time,sys,multiprocessing
from functools import partial
from NFW import DM_density,Mass_r
from scipy.stats import norm
from Einasto import EIN_norm
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

f = open("LOGS/"+time.strftime("%Y%m%d_%H%M%S")+'_NUMBER_DENSITY_LOG.txt','a')
f.write("************************************************\
**************************************\nNUMBER DENSITY LOG\nDATA SET: {0}\nDATE/TIME: {1}\n\n\
*****************************************\
*********************************************\n\n\n".format(DIRECTORY,time.strftime("%Y%m%d_%H%M%S")))

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




#Number density

fig = plt.figure(figsize=(6.4*3/4,2*6.4*3/4))
ax8 = fig.add_subplot(212)

#fig2 = plt.figure(figsize=(6.4*3/4,4.8*3/4))
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
	
def Fit_Ein(r,a,r200,r2):
	return np.log10(EIN_norm(r,a,r200,r2))

def Normal(x,m,s):
	return 1./(2*np.pi*s**2)**0.5 * np.exp(-(x-m)**2/(2*s**2))
	
def red_chisquared(indata,error,modeldata,dof):
	chi = np.sum((indata - modeldata)**2/error**2)
	return chi/dof
	
def Bootstrap(star_rs,bins,N,galaxy,func):
	outvars = []
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
		out = scipy.optimize.curve_fit(func,(bins[1:]+bins[:-1])/2,np.log10(densitys))[0]
		outvars.append(out)
		i += 1
			
		#except Exception: pass
		
		
	densitys = np.array(densitys_sample)
	
	med_densitys = np.nanmedian(densitys,axis=0)
	med_densitys_3s_ub = np.percentile(densitys,99.7,axis=0)
	med_densitys_3s_lb = np.percentile(densitys,100-99.7,axis=0)
	med_densitys_1s_ub = np.percentile(densitys,50+34.1,axis=0)
	med_densitys_1s_lb = np.percentile(densitys,50-34.1,axis=0)
	
	outvars = np.array(outvars)

	ubs = np.percentile(outvars,50+34.1,axis=0)
	lbs = np.percentile(outvars,50-34.1,axis=0)
	medians = np.median(outvars,axis=0)

	if func == Fit_NFW:	chi = red_chisquared(med_densitys,(med_densitys_3s_ub-med_densitys_3s_lb)/2,10**func((bins[1:]+bins[:-1])/2,medians[0],medians[1]),med_densitys.size-2)
	elif func == Fit_Ein: chi = red_chisquared(med_densitys,(med_densitys_3s_ub-med_densitys_3s_lb)/2,10**func((bins[1:]+bins[:-1])/2,medians[0],medians[1],medians[2]),med_densitys.size-3)
	
	return (med_densitys,med_densitys_3s_ub,med_densitys_3s_lb,med_densitys_1s_ub,med_densitys_1s_lb,medians,ubs,lbs,chi)
	

out_MW = Bootstrap(new_MW_rs,bins_MW,1e3,'MW',Fit_NFW)
out_M31 = Bootstrap(new_M31_rs,bins_M31,1e3,'M31',Fit_NFW)

out_MW_E = Bootstrap(new_MW_rs,bins_MW,1e3,'MW',Fit_Ein)
out_M31_E = Bootstrap(new_M31_rs,bins_M31,1e3,'M31',Fit_Ein)


f.write('NFW Fit\n\nMW\nr_s = {:.01f} ub = {:.01f} lb = {:.01f}\nr_200 = {:.01f} ub = {:.01f} lb = {:.01f}\nchi = {:.02f}\n\nM31\nr_s = {:.01f} ub = {:.01f} lb = {:.01f}\nr_200 = {:.01f} ub = {:.01f} lb = {:.01f}\nchi = {:.02f}\n\n'.format(out_MW[5][0],out_MW[6][0],out_MW[7][0],out_MW[5][1],out_MW[6][1],out_MW[7][1],out_MW[8],out_M31[5][0],out_M31[6][0],out_M31[7][0],out_M31[5][1],out_M31[6][1],out_M31[7][1],out_M31[8]))

f.write('Einasto Fit\n\nMW\na = {:.02f} ub = {:.02f} lb = {:.02f}\nr_200 = {:.01f} ub = {:.01f} lb = {:.01f}\n\
r_2 = {:.01f} ub = {:.01f} lb = {:.01f}\nchi = {:.02f}\n\n\
M31\na = {:.02f} ub = {:.02f} lb = {:.02f}\nr_200 = {:.01f} ub = {:.01f} lb = {:.01f}\n\
r_2 = {:.01f} ub = {:.01f} lb = {:.01f}\nchi = {:.02f}\n\n'.format(out_MW_E[5][0],\
out_MW_E[6][0],out_MW_E[7][0],out_MW_E[5][1],out_MW_E[6][1],out_MW_E[7][1],out_MW_E[5][2],out_MW_E[6][2],out_MW_E[7][2],out_MW_E[8],out_M31_E[5][0],out_M31_E[6][0],out_M31_E[7][0],out_M31_E[5][1],out_M31_E[6][1],out_M31_E[7][1],out_M31_E[5][2],out_M31_E[6][2],out_M31_E[7][2],out_M31_E[8]))


model_MW_3 = DM_density(r_0_200_M*1e3*pc,out_MW[5][0]*1e3*pc,out_MW[5][1]*1e3*pc) * 4/3*np.pi*(out_MW[5][1]*1e3*pc)**3 / Mass_r(out_MW[5][1]*1e3*pc,out_MW[5][0]*1e3*pc,out_MW[5][1]*1e3*pc) 
model_M31_3 = DM_density(r_0_200_A*1e3*pc,out_M31[5][0]*1e3*pc,out_M31[5][1]*1e3*pc) * 4/3*np.pi*(out_M31[5][1]*1e3*pc)**3 / Mass_r(out_M31[5][1]*1e3*pc,out_M31[5][0]*1e3*pc,out_M31[5][1]*1e3*pc) 

model_MW_Ein = EIN_norm(r_0_200_M*1e3*pc,out_MW_E[5][0],out_MW_E[5][1]*1e3*pc,out_MW_E[5][2]*1e3*pc)
model_M31_Ein = EIN_norm(r_0_200_A*1e3*pc,out_M31_E[5][0],out_M31_E[5][1]*1e3*pc,out_M31_E[5][2]*1e3*pc)



ax7.plot((bins_MW[1:]+bins_MW[:-1])/2/r_200_M,out_MW[0],'o-',label=r'MW',ms=3)
params_MW = np.around(out_MW[3:5])
ax7.plot(r_0_200_M/r_200_M,model_MW_3,'r-',label=r'NFW')#, $r_{s}=%.0f$ kpc $r_{200}=%.0f$ kpc'%(out_MW[5][0],out_MW[5][1]))
ax7.fill_between((bins_MW[1:]+bins_MW[:-1])/2/r_200_M,out_MW[1],out_MW[2],alpha=0.15,color='gray',linestyle='None',edgecolor='None')
ax7.fill_between((bins_MW[1:]+bins_MW[:-1])/2/r_200_M,out_MW[3],out_MW[4],alpha=0.3,color='gray',linestyle='None',edgecolor='None')
ax7.plot(r_0_200_M/r_200_M,model_MW_Ein,'--',color='#ff8000',label=r'Einasto')#, $\alpha=%.1f$, $r_{200}=%.0f$ kpc $r_{-2}=%.0f$ kpc'%(out_MW_E[5][0],out_MW_E[5][1],out_MW_E[5][2]))

ax8.plot((bins_M31[1:]+bins_M31[:-1])/2/r_200_A,out_M31[0],'o-',label=r'M31',ms=3)
params_M31 = np.around(out_M31[3:5])
ax8.plot(r_0_200_A/r_200_A,model_M31_3,'r-',label=r'NFW')#, $r_{s}=%.0f$ kpc $r_{200}=%.0f$ kpc'%(out_M31[5][0],out_M31[5][1]))
ax8.fill_between((bins_M31[1:]+bins_M31[:-1])/2/r_200_A,out_M31[1],out_M31[2],alpha=0.15,color='gray',linestyle='None',edgecolor='None')
ax8.fill_between((bins_M31[1:]+bins_M31[:-1])/2/r_200_A,out_M31[3],out_M31[4],alpha=0.3,color='gray',linestyle='None',edgecolor='None')
ax8.plot(r_0_200_A/r_200_A,model_M31_Ein,'--',color='#ff8000',label=r'Einasto')#, $\alpha=%.1f$ $r_{200}=%.0f$ kpc $r_{-2}=%.0f$ kpc'%(out_M31_E[5][0],out_M31_E[5][1],out_M31_E[5][2]))

ax7.loglog()
ax8.loglog()

for ax in (ax7,ax8): ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)

ax8.legend(loc=3,fontsize=AXESFONTSIZE)
ax7.legend(loc=3,fontsize=AXESFONTSIZE)
ax8.set_xlabel(r'$r/R_{200}$',fontsize=AXESFONTSIZE)
#ax7.set_xlabel(r'$r/R_{200}$',fontsize=AXESFONTSIZE)
ax8.set_ylabel(r'$n(r)/\langle n \rangle$',fontsize=AXESFONTSIZE)
ax7.set_ylabel(r'$n(r)/\langle n \rangle$',fontsize=AXESFONTSIZE)
plt.setp([ax7.get_xticklabels()],visible=False)

fig.tight_layout()
#fig2.tight_layout()
#ax7.set_xticklabels([])
fig.subplots_adjust(hspace=0)
#SAVEFIG("Save number density MW fig? (y/n)","n_DENSITY_MW_FIG",fig2)
SAVEFIG("Save number density fig? (y/n)","n_DENSITY_M31_FIG",fig)


f.close()

plt.show()




