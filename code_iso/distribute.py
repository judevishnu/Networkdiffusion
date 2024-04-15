import gsd
import gsd.hoomd
import time  as time1
import numpy as np
import random
import math
import freud
import sys
from numpy import  linalg as LA

######################################################################################################
ptlarray=(2426844, 2098608, 1934592, 1852584, 1770372, 1688364, 1442136, 1278120, 1196112)
fracarray=("2.39981", "1.99980", "1.79992", "1.69998", "1.59979", "1.49985", "1.19978", "0.99990", "0.89996") 
densarray=("4.30305", "3.58580", "3.22740", "3.04820", "2.86855",  "2.68935", "2.15130", "1.79290", "1.61370") 
molearray=("0.81143", "0.78194", "0.76345", "0.75298", "0.74151", "0.72895", "0.68267", "0.64195", "0.61740") 
kbT=1.0
binrange = np.linspace(0,40,400)
binval = .5*(binrange[1:]+binrange[:-1])
#print(binrange)

for i in range(0,9):
    Ntot = ptlarray[i]
    fraction=fracarray[i]
    densratio=densarray[i]
    molefrac = molearray[i]
    filen = "ReDistribution_"+"diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+"_segment_102"+".txt"
    data = np.loadtxt(filen)
    histdata = np.histogram(data, bins=binrange,density=True)
    #print(len(histdata[0]),len(histdata[1]))
    #print(histdata[0])
    #print(len(binrange))
    #print(histdata[1])
    findata = np.asarray((binval,histdata[0])).T
    filename3 = "ProbabilityRe_"+"diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+"_segment_"+"102.txt"
    with open(filename3, 'wb+') as f3:
        np.savetxt(f3, findata)


    


