import numpy as np
import gsd.hoomd
import sys

sample = sys.argv[1]
N = sys.argv[2]
vol_frac = sys.argv[3]
densratio = sys.argv[4]
molefrac = sys.argv[5]
kbT=1.0

#filename1= "../Finaltraj/finaltrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"

#filename= "../Ctrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"

filename2= "./Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"


filename3= "./Captureperc"+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"




write = gsd.hoomd.open(filename3,mode='wb+')
with gsd.hoomd.open(filename2,mode='rb+') as f1:
    write.extend(f1[len(f1)-3:]) 
write.close()
