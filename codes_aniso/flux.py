#!/usr/bin/python3
import hoomd
import hoomd.md
import gsd
import gsd.hoomd
import time as time1
import numpy
import random 
import math
import freud
import sys
########################################################################
#           Set parameters
#######################################################################
samples=3
dia=[]
dia.append(1.0)
dia.append(1.0)
sigma = [] # interaction parameter sigma 

sigma.append([(dia[0]+dia[0])/2.,(dia[0]+dia[1])/2.])
###############################################################################
begining = time1.perf_counter()
dt = 0.001

#samples=int(sys.argv[1])
Ntot = sys.argv[1]
fraction = sys.argv[2]
densratio = sys.argv[3]
molefrac  = sys.argv[4]


kbT = 1.0
cap_X_avg_list = []
cap_COM_avg_list = []
time_list=[]
period = 8e5

msd_sx=[]
msd_sy=[]
msd_sz=[]
msd_s=[]
dt=0.001
sample=1

filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"

traj=gsd.hoomd.open(name=filename, mode='rb')
Lx=traj[0].configuration.box[0]
Ly=traj[0].configuration.box[1]
Lz=traj[0].configuration.box[2]
Lxby2 = Lx/2.0 
Lyby2 = Ly/2.0 
Lzby2 = Lz/2.0 

box = freud.box.Box(Lx = Lx, Ly= Ly, Lz =Lz)

Ny = int(Ly/sigma[0][0])
poly_len = int(Lz/sigma[0][0])
print(poly_len)

typeid =  numpy.copy(traj[0].particles.typeid)
N_remainz = numpy.count_nonzero(typeid == 1)
N_constraint = numpy.count_nonzero(typeid==0)
Nx_remainz = int(N_remainz/2/Ny/poly_len)
cutoffx =2.0*sigma[0][0]
lcutoffx =cutoffx
lengthx = int(Lxby2/lcutoffx)
    
lcx = Lxby2/lengthx

shiftx = int(lengthx)
cellnox= 2*lengthx
totalcell = cellnox
slcx = shiftx*lcx

count_A = numpy.empty(totalcell,dtype=int)
count_B = numpy.empty(totalcell,dtype=int)
count_AB = numpy.empty(totalcell,dtype=int)
bin_val = numpy.empty(totalcell,dtype=float)
#print(len(bin_val))
data1 = numpy.empty((0,len(bin_val),5),dtype=float)
cell=numpy.empty((0),dtype=float)
one=numpy.empty((0),dtype=int)
t=len(traj)
msd_frame = numpy.empty((0,totalcell),dtype=float)
delV = Ly*Lz*cutoffx
cols=totalcell
rows =t

init1=1500
for x in range(totalcell):
    bin_val[x] = (x+0.5)*lcutoffx-Lxby2

msd_samplebin = numpy.empty((0,totalcell,1),dtype=float)
msd_samplevar = numpy.empty((0,totalcell,1),dtype=float)
for sample in range(1,samples):
    filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
    print(filename)
    
    traj=gsd.hoomd.open(name=filename, mode='rb')
    Lx=traj[0].configuration.box[0]
    Ly=traj[0].configuration.box[1]
    Lz=traj[0].configuration.box[2]
    Lxby2 = Lx/2.0 
    Lyby2 = Ly/2.0 
    Lzby2 = Lz/2.0 

    box = freud.box.Box(Lx = Lx, Ly= Ly, Lz =Lz)


    data1 = numpy.empty((0,len(bin_val),5),dtype=float)
    msd_bintime=numpy.empty((0,totalcell,1),dtype=float)
    for frame in range(init1,len(traj)-1):
        msd_bin=numpy.empty((0,1),dtype=float)
        container=[]
    
        msd_box=[]
        time = frame*dt*period
        
        posA=traj[frame].particles.position[:N_constraint]
        posB=traj[frame].particles.position[N_constraint:N_constraint+N_remainz]
        imageB=traj[frame].particles.image[N_constraint:N_constraint+N_remainz]
        posBplus_1=traj[frame+1].particles.position[N_constraint:N_constraint+N_remainz]
        imageBplus_1=traj[frame+1].particles.image[N_constraint:N_constraint+N_remainz]

        pos=traj[frame].particles.position
        count_A.fill(0)
        count_B.fill(0)
        count_AB.fill(0)

        xiB = ((slcx+posB[:,0])/lcx).astype(int)
        xiB=numpy.where(xiB<0,xiB+1,xiB)
        xiB=numpy.where(xiB==cellnox,xiB-1,xiB)

        xiBplus = ((slcx+posBplus_1[:,0])/lcx).astype(int)
        xiBplus =numpy.where(xiBplus<0,xiBplus+1,xiBplus)
        xiBplus =numpy.where(xiBplus==cellnox,xiBplus-1,xiBplus)


        mB=xiB[:]
        mBplus=xiBplus[:]

        for x in range(totalcell):
            ids=numpy.where(mB==x)
            ids=numpy.unique(ids[0])
            idsplus=numpy.where(mBplus==x)
            idsplus=numpy.unique(idsplus[0])

            rho = len(ids)/Lx/Ly 
            rhoplus = len(idsplus)/Lx/Ly 
            drho = (rhoplus-rho)/(dt*period)
            msd_bin=numpy.append(msd_bin,numpy.asarray(drho).reshape(-1,1),axis=0)  
            #print(msdx[0],msdy[0],msdz[0])
            #print(msd_bin.shape,totalcell,msdx[0])
            #print(msd_bin.shape)
        msd_bintime=numpy.append(msd_bintime,[msd_bin],axis=0)
        #print(msd_bintime.shape)
    msd_samplebin=numpy.append(msd_samplebin,[numpy.nanmean(msd_bintime,axis=0)],axis=0)
    print(msd_samplebin.shape)
    msd_samplevar = numpy.append(msd_samplevar,[numpy.nanvar(msd_bintime,axis=0)],axis=0)
print(msd_samplevar.shape)
data9 = (numpy.asarray((msd_samplebin[:,:,0].T)).T).reshape(-1,len(bin_val),1)
data10=( numpy.asarray((msd_samplevar[:,:,0].T)).T).reshape(-1,len(bin_val),1)


print(data9.shape)

#data = numpy.nanmean(msd_samplebin,axis=0)
data = numpy.nanmean(data9,axis=0)
#datastd = numpy.nanstd(msd_samplebin,axis=0)
datastd = numpy.nanstd(data9,axis=0)
#datastd = numpy.sqrt((numpy.nansum(data10,axis=0))/len(data10)/len(data10))

print(datastd.shape,data.shape)
#exit()
print(len(bin_val),len(msd_samplebin[:]))
#data1 = numpy.array((bin_val,data[:,0],data[:,1],data[:,2],data[:,0]/(2*period*dt),data[:,1]/(2*period*dt),data[:,2]/(2*period*dt))).T
#data2 = numpy.concatenate((data1,datastd[:,1:]),axis=1)

bin_val = bin_val.reshape(totalcell,1)
data2 = numpy.concatenate((bin_val,data,datastd),axis=1)

print(data2.shape,bin_val.shape)

#print(data1)
filename3 = "Flux_Positiondep_diffusionconst_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"
with open(filename3, 'wb+') as f2:
    numpy.savetxt(f2, data2)
       
end = time1.perf_counter()
print(abs(end-begining))



