import gsd
import gsd.hoomd
import time as time1
import numpy as np
import random
import math
import freud
import sys
#import scipy.spatial as spatial
#import scipy.spatial.distance as dist
##########################################################
samples=11
Ntot=sys.argv[1]
fraction = sys.argv[2]
densratio = sys.argv[3]
molefrac = sys.argv[4]
rmax = sys.argv[5]
kbT=1.0
filename = "PercolationMPI"+str(1)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_rmax_"+str(rmax)+"_kT_"+str(kbT)+".txt"
data = np.loadtxt(filename)
print(data.shape)
sampledata = np.empty((0,len(data),6),dtype=float)

timeframe =np.asarray([1100,1200,1300,1400,1500,1599])
timeindex  = timeframe-1000
print(timeindex)
print(Ntot,fraction,densratio,molefrac,rmax)

for sample in range(1,samples):
    filename = "PercolationMPI"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_rmax_"+str(rmax)+"_kT_"+str(kbT)+".txt"
    data = np.loadtxt(filename)
    sampledata = np.append(sampledata,[data],axis=0)


mean=np.nanmean(sampledata,axis=0)
std=np.nanstd(sampledata[:,:,1:],axis=0)

fname = "Avgperc_oversample"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_rmax_"+str(rmax)+"_kT_"+str(kbT)+".txt"
findata = np.concatenate((mean,std),axis=1)
print(findata.shape)
#exit()
with open(fname, 'w+') as f3:
        np.savetxt(f3, findata)
