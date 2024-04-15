import numpy as np
import gsd.hoomd
import sys

sample = sys.argv[1]
N = sys.argv[2]
vol_frac = sys.argv[3]
densratio = sys.argv[4]
molefrac = sys.argv[5]
kbT=1.0

filename1= "../Finaltraj/finaltrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"

filename= "../Ctrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"

filename2= "./Combined/Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"



write = gsd.hoomd.open(filename2,mode='wb+')
with gsd.hoomd.open(filename1,mode='rb+') as f1:

    with gsd.hoomd.open(filename,mode='rb+') as f:
        print(f[0].configuration.step,f[len(f)-1].configuration.step)
        #for i in range(0,len(f1)):
        #    print(i,f1[i].configuration.step)
        #for j in range(2,len(f)):
        #    print(i+j,f[j].configuration.step)
        write.extend(f)
        write.extend(f1[1:])

        print(len(write))
        #for i in range(len(f1)):
        #    write.append(f1[i])
        #for j in range(1,len(f)):
        #    write.append(f[j])
        #print(write[0].configuration.step,write[len(write)-1].configuration.step)
    #for frame in range(0,len(f)):
    #    print(frame,f[frame].configuration.step,f[len(f)-1].configuration.step)
    #for frame in range(0,800,2):
    #write.append(f[len(f)-1])

write.close()
