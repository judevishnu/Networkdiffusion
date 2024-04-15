import numpy as np
import gsd.hoomd
import freud
#import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import csv
import sys


Ntot = sys.argv[1]
fraction = sys.argv[2]
densratio = sys.argv[3]
molefrac  = sys.argv[4]

samples = 11
n_bin = 200
data_jump = 1
dt = 0.001
kbT=1.0
init=1
####################################################################################################################
## loading in csv dictionary
####################################################################################################################

descriptorDict = defaultdict(list)

#with open('description_file.csv', 'r') as data:

#    csv_reader = csv.reader(data)
#    rows = list(csv_reader)
#for row in rows:
#    descriptorDict[int(row[0])] = row[1]
#a = descriptorDict[Ntot]
#v=a.strip('][').split(', ')

####################################################################################################################
## parameters
####################################################################################################################

#volfrac = v[0]
#molefrac = v[1]
#denseratio = v[2]


####################################################################################################################
## functions
####################################################################################################################

def truncate(n):
    n = float(int(n * 100))
    n/= 100
    return n

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



standLen = 102
sample=1
filename2 = "density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".bin"
data_den = np.load(filename2)
print (len(data_den))
max_denspoly=np.max(data_den[1000:,:,2])

filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
 


datadummy = gsd.hoomd.open(name=filename, mode='rb')
time_1=int((len(datadummy)-1)/data_jump)
DII_sample = np.empty((0,time_1,2),dtype=float)



for sample in tqdm(range(1,samples,1)):
    filename = "./Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
     
    data_par = gsd.hoomd.open(name=filename, mode='rb')
    simly=data_par[0].configuration.box[1]
    simlz=data_par[0].configuration.box[2]
    Area = simly*simlz
    DII_time = np.empty((0,2),dtype=float)

    for frame in range(1,len(data_par),data_jump):
        i = int((frame-0)/data_jump)
        I1,I2=Interface(i)
        print(I1,I2)
        snap=data_par[frame]

        pos=snap.particles.position
        arr_polymer=snap.particles.typeid==1
        polypos=pos[arr_polymer]
        polystrand=polypos.reshape(int(len(polypos)/standLen),standLen,3)
        stran_x=polystrand[:,:,0]


####################################################################################################################
        poly_ids1 = np.where(((stran_x>I1-1) &(stran_x<I1+1)))[0]
        poly_ids1 = np.unique(poly_ids1)
        dens_poly1 = len(poly_ids1)/Area
        avlen1 = np.empty(0,dtype=float)
        if len(poly_ids1)!=0:
            for x in poly_ids1:
                px1 = stran_x[x]
                idsR1 = np.where((px1>=I1)&(px1<0))[0]
                diffR1 = np.fabs(px1[idsR1]-I1)

                idsL1 = np.where((px1<=I1))
                diffL1 = np.fabs(px1[idsL1]-I1)

                #max_diff1 = np.max(diff1)
                dL1=np.sum(diffL1)
                dR1 = np.sum(diffR1)
                min_L1= min(dL1,dR1)
                avlen1 = np.append(avlen1,min_L1)

        else:
            avlen1 = np.nan
        avLen1 = np.nanmean(avlen1)
        #print(avLen1)
        DII_I1 = avLen1* dens_poly1

####################################################################################################################

        poly_ids2 = np.where(((stran_x<I2+1) & (stran_x>I2-1)))[0]
        poly_ids2 = np.unique(poly_ids2)
        dens_poly2 = len(poly_ids2)/Area
        avlen2 = np.empty(0,dtype=float)
        if len(poly_ids2)!=0:
            for y in poly_ids2:
                px2 = stran_x[y]
                idsL2 = np.where((px2<=I2)&(px2>0))[0]
                diffL2 = np.fabs(px2[idsL2]-I2)
                idsR2 = np.where((px2>=I2))[0]
                diffR2 = np.fabs(px2[idsR2]-I2)
                dL2 = np.sum(diffL2)
                dR2 = np.sum(diffR2)
                #max_diff2 = np.max(diff2)
                min_L2= min(dL2,dR2)
                avlen2 = np.append(avlen2,min_L2)

        else:
            avlen2 = np.nan
        avLen2 = np.nanmean(avlen2)

        DII_I2 = avLen2* dens_poly2
        print(DII_I1,DII_I2)
####################################################################################################################


        DII = (DII_I1+DII_I2)/2
        time = dt*snap.configuration.step

        DII_time = np.append(DII_time,[[time,DII]],axis=0)

    DII_sample = np.append(DII_sample,[DII_time],axis=0)

print(DII_sample.shape)
avDIIarray = np.nanmean(DII_sample,axis=0)
print(avDIIarray.shape)
stdDIIarray = np.nanstd(DII_sample[:,:,1],axis=0)
print(stdDIIarray.shape)
finresults = np.asarray((avDIIarray[:,0].T,avDIIarray[:,1].T,stdDIIarray.T)).T
print(finresults.shape)


filenameres = "DII_frompaper"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"

with open(filenameres, 'w+') as f3:
    np.savetxt(f3, finresults)
 
print("Programm done, File has been saved!")
