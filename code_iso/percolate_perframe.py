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

concdata = np.empty((0,len(ptlarray),11),dtype=float)
for k in range(0,len(timeindex)):
    
    combined = np.zeros((len(ptlarray),11),dtype=float)
    for i in range(0,len(ptlarray)):
        rho = rhosol[i,2]
        Ntot =ptlarray[i]
        fraction = fracarray[i]
        densratio = densarray[i]
        molefrac = molearray[i]

        fname = "Avgperc_oversample"+"_Isodiamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_rmax_"+str(rmax)+"_kT_"+str(kbT)+".txt"
        data = np.loadtxt(fname)
        percdata=data[timeindex[k]]
        print(percdata.shape)
        #combined = np.zeros(len(percdata))
    

        combined[i,1:]=percdata[1:]
        combined[i,0]= rho
    
    concdata = np.append(concdata,[combined],axis=0)
fname = "Percolationframe_Isodiamond_rmax"+str(rmax)+".txt"
with open(fname, 'w+') as f3:
    for m in range(len(concdata)):
        np.savetxt(f3, concdata[m])
        f3.write("\n")
