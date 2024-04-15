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
kbT=1.0

timeframe =np.asarray([1150,1200,1250,1300,1350,1400,1450,1500,1599])
timeindex  = timeframe-1000
print(timeindex)

ptlarray=(2426844, 2098608, 1934592, 1852584, 1770372, 1688364, 1442136, 1278120, 1196112)
fracarray=("2.39981", "1.99980", "1.79992", "1.69998", "1.59979", "1.49985", "1.19978", "0.99990", "0.89996") 
densarray=("4.30305", "3.58580", "3.22740", "3.04820", "2.86855",  "2.68935", "2.15130", "1.79290", "1.61370") 
molearray=("0.81143", "0.78194", "0.76345", "0.75298", "0.74151", "0.72895", "0.68267", "0.64195", "0.61740") 
fsol = "rhosolution.txt"
rhosol=np.loadtxt(fsol)
print(rhosol.shape)

for i in range(0,len(ptlarray)):
    rho = rhosol[i,2]
    Ntot =ptlarray[i]
    fraction = fracarray[i]
    densratio = densarray[i]
    molefrac = molearray[i]
    
    concdata = np.empty((0,1599,3),dtype=float)
    for sample in range(1,samples):
        fname = "Diffusiondepth"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"

        data = np.loadtxt(fname)
        concdata=np.append(concdata,[data],axis=0)
           
    meandata = np.nanmean(concdata,axis=0)
    stddata = np.nanstd(concdata,axis=0)
    finresults = np.concatenate((meandata,stddata[:,1:]),axis=1)
    fname1 = "Diffusiondepth_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"

    with open(fname1, 'w+') as f3:
        np.savetxt(f3, finresults)
