import numpy as np
import gsd.hoomd
import freud
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import csv

samples =4
n_bin = 696
data_jump = 1
dt = 0.001
kbT=1.0
Ntot = sys.argv[1]
fraction = sys.argv[2]
densratio = sys.argv[3]
molefrac  = sys.argv[4]
cutoffsp = 1.0
sample=1
init=1
####################################################################################################################
## functions
####################################################################################################################

def Interface(i):  
    data=data_den[i-init] 
    data_1 = data[0:int(len(data)/2)]
    data_2 = data[int(len(data)/2):]
    
    gel_den_1 = data_1[:,1]
    poly_den_1 = data_1[:,2]
    polydif_1 = np.argmin(np.absolute(np.subtract(gel_den_1,poly_den_1)))
    
    gel_den_2 = data_2[:,1]
    poly_den_2 = data_2[:,2]
    polydif_2 = np.argmin(np.absolute(np.subtract(gel_den_2,poly_den_2)))
    I1=data_1[polydif_1,0]
    I2=data_2[polydif_2,0]
    return I1,I2


######################################################################################################################



filename2 = "density_vs_xbox"+"_Isodiamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".bin"
data_den = np.load(filename2)
print (len(data_den))
max_denspoly=np.max(data_den[1000:,:,2])
print(max_denspoly.shape)
#exit()
filename = "trajectoryContinueIso"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
   
datadummy = gsd.hoomd.open(name=filename, mode='rb')
standLen =102#int(datadummy[0].configuration.box[2])
time_1=int((len(datadummy)-1)/data_jump)
DII_sample = np.empty((0,time_1,3),dtype=float)


r0=1.5
Lmax=0
Smax =r0*max_denspoly
if standLen%2==0:
    for i in range(int(standLen/2)):
        Lmax = Lmax+(i-0.5)*r0
else:
    if standLen%2!=0:
        for i in range(int((standLen-1)/2)):
            Lmax = Lmax+(i)*r0

#Inorm = Lmax*Smax
Inorm = Smax


for sample in tqdm(range(1,samples,1)):
    filename = "./trajectoryContinueIso"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
       
    data_par = gsd.hoomd.open(name=filename, mode='rb')
    simly=data_par[0].configuration.box[1]
    simlz=data_par[0].configuration.box[2]
    Area = simly*simlz
    DII_time = np.empty((0,3),dtype=float)
    
    for frame in range(1,len(data_par),data_jump):
        i = int((frame-0)/data_jump)
        print(i)
        I1,I2=Interface(i)

        snap=data_par[frame]
        
        pos=snap.particles.position
        arr_polymer=snap.particles.typeid==1
        polypos=pos[arr_polymer]
        polystrand=polypos.reshape(int(len(polypos)/standLen),standLen,3)
        stran_x=polystrand[:,:,0]
       
    
####################################################################################################################        
        poly_ids1 = np.where(((stran_x>I1) &(stran_x<I1+cutoffsp)))[0]
        poly_ids1 = np.unique(poly_ids1)
        dens_poly1 = len(poly_ids1)/Area
        avlen1 = np.empty(0,dtype=float)
        if len(poly_ids1)!=0:
            for x in poly_ids1:
                px1 = stran_x[x]
                ids1 = np.where((px1>=I1)&(px1<0))[0]
                #diff1 = np.fabs(px1[ids1]-I1)
                diff1 = px1[ids1]-I1
                #max_diff1 = np.max(diff1)
                #max_diff1=np.sum(np.fabs(np.ediff1d(diff1)))
                max_diff1=np.sum(np.ediff1d(diff1)**2)
                avlen1 = np.append(avlen1,max_diff1) 
            
        else:
            avlen1 = np.nan
        avLen1 = np.sqrt(np.nanmean(avlen1))
        #print(avLen1)
        DII_I1 = avLen1* dens_poly1

####################################################################################################################        
       
        poly_ids2 = np.where(((stran_x<I2) & (stran_x>I2-cutoffsp)))[0]
        poly_ids2 = np.unique(poly_ids2)
        dens_poly2 = len(poly_ids2)/Area
        avlen2 = np.empty(0,dtype=float)
        if len(poly_ids2)!=0:
            for y in poly_ids2:
                px2 = stran_x[y]
                ids2 = np.where((px2<=I2)&(px2>0))[0]
                #diff2 = np.fabs(px2[ids2]-I2)
                diff2 = px2[ids2]-I2
                #max_diff2 = np.max(diff2)
                #max_diff2=np.sum(np.fabs(np.ediff1d(diff2)))
                max_diff2=np.sum(np.ediff1d(diff2)**2)
                avlen2 = np.append(avlen2,max_diff2) 
            
        else:
            avlen2 = np.nan
        avLen2 = np.sqrt(np.nanmean(avlen2))
        
        DII_I2 = avLen2* dens_poly2
        #print(DII_I1.T,DII_I2.T)
####################################################################################################################        
        
    
        DII = (DII_I1+DII_I2)/2
        time = dt*snap.configuration.step
        
        DII_time = np.append(DII_time,[[time,DII,DII/Inorm]],axis=0)
        
    DII_sample = np.append(DII_sample,[DII_time],axis=0)

print(DII_sample.shape)
avDIIarray = np.nanmean(DII_sample,axis=0)
#avDIInormarray = avDIIarray[:,1]/Inorm
print(avDIIarray.shape)
stdDIIarray = np.nanstd(DII_sample[:,:,1:3],axis=0)
print(stdDIIarray.shape)        
finresults = np.asarray((avDIIarray[:,0].T,avDIIarray[:,1].T,avDIIarray[:,2].T,stdDIIarray[:,0].T,stdDIIarray[:,1].T)).T
print(finresults.shape)


filename = "torstDIInorm9"+"_Isodiamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"
with open(filename, 'w+') as f3:
    np.savetxt(f3, finresults)
 
print("Programm done, File has been saved!")
        


