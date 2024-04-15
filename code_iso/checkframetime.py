import numpy as np
import gsd.hoomd
import sys

sample = sys.argv[1]
N = sys.argv[2]
vol_frac = sys.argv[3]
densratio = sys.argv[4]
molefrac = sys.argv[5]
kbT=1.0
filename="./trajectoryContinueIso"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"


#write = gsd.hoomd.open(filename,mode='wb+')
with gsd.hoomd.open(filename,mode='rb+') as f1:
    for frame in range(len(f1)):
        print(N,vol_frac,densratio,molefrac,sample,len(f1),frame,f1[frame].configuration.step)
  
