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
rmax = sys.argv[1]
kbT=1.0

#timeframe =np.asarray([1000,1100,1200,1300,1400,1500,1590])
#timeframe =np.asarray([1200,1250,1300,1350,1400,1450,1500,1599])

timeframe =np.asarray([1150,1200,1250,1300,1350,1400,1450,1500,1599])
timeindex  = timeframe-1000
print(timeindex)


#ptlarray=(3368304, 2883192, 2640636, 2519256, 2398080, 2276700, 1912968, 1670412, 1549032)
#fracarray=("2.39992", "1.99994", "1.79994", "1.69986", "1.59995", "1.49987", "1.19996", "0.99997", "0.89989")
#densarray=("6.36029", "5.30024", "4.77022", "4.50498", "4.24019", "3.97496", "3.18014", "2.65012", "2.38489")
#molearray=("0.86414", "0.84128", "0.82670", "0.81835", "0.80917", "0.79899", "0.76077", "0.72604", "0.70457")

ptlarray=(2883192, 2640636, 2519256, 2398080, 2276700, 1912968, 1670412, 1549032)
fracarray=("1.99994", "1.79994", "1.69986", "1.59995", "1.49987", "1.19996", "0.99997", "0.89989")
densarray=("5.30024", "4.77022", "4.50498", "4.24019", "3.97496", "3.18014", "2.65012", "2.38489")
molearray=("0.84128", "0.82670", "0.81835", "0.80917", "0.79899", "0.76077", "0.72604", "0.70457")



fsol = "rhosolution.txt"
rhosol=np.loadtxt(fsol)
print(rhosol.shape)

#concdata = np.empty((0,len(ptlarray),9),dtype=float)
combined =np.zeros((len(ptlarray),11),dtype=float)
for i in range(0,len(ptlarray)):
        rho = rhosol[i,2]
        Ntot =ptlarray[i]
        fraction = fracarray[i]
        densratio = densarray[i]
        molefrac = molearray[i]

        fname = "Avgperc_oversample"+"_Isodiamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_rmax_"+str(rmax)+"_kT_"+str(kbT)+".txt"
        data = np.loadtxt(fname)
        print(data.shape)
        mean = np.nanmean(data[:,1:6],axis=0)
        std = np.nanstd(data[:,1:6],axis=0)
        stdnew = np.sqrt(np.nanmean(np.square(data[:,6:]/np.sqrt(10)),axis=0))
        print(stdnew) 

        combined[i,1:6]=mean
        combined[i,6:]=stdnew
        combined[i,0]= rho
    
fname = "Percolationtotalnew_Isodiamond_"+str(rmax)+".txt"
with open(fname, 'w+') as f3:
   np.savetxt(f3, combined)
       
