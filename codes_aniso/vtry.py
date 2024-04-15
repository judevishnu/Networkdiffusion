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
samples=11
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
def flux(x,mA,mB,velA,velB):
     idsB=numpy.where(mB==x)
     idsB=numpy.unique(idsB[0])
     idsA=numpy.where(mA==x)
     idsA=numpy.unique(idsA[0])
     rhoA = len(idsA)/delV
     rhoB = len(idsB)/delV
     rho = rhoA+rhoB
     #if len(idsB)!=0:
     velxyzB = numpy.take(velB,idsB,axis=0)
     

     diffarrxyzA = numpy.nansum(numpy.asarray([comxyzA,-avvelABxyz]))
     jBxyz = rho*(diffarrxyzB)
     return jAxyz,jBxyz


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
cellnox= lengthx
totalcell = cellnox
slcx = shiftx*lcx

count_A = numpy.empty(totalcell,dtype=int)
count_B = numpy.empty(totalcell,dtype=int)
count_AB = numpy.empty(totalcell,dtype=int)
bin_val = numpy.empty(totalcell,dtype=float)
#print(len(bin_val))
data1 = numpy.empty((0,len(bin_val),4),dtype=float)
cell=numpy.empty((0),dtype=float)
one=numpy.empty((0),dtype=int)
t=len(traj)
msd_frame = numpy.empty((0,totalcell),dtype=float)
delV = Ly*Lz*cutoffx
cols=totalcell
rows =t

init1=1500
for x in range(totalcell):
    bin_val[x] = (x+0.5)*lcutoffx

d_samplebin = numpy.empty((0,totalcell,2),dtype=float)
d_samplevar = numpy.empty((0,totalcell,2),dtype=float)
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

    boxarr= numpy.asarray([Lx,Ly,Lz])
    data1 = numpy.empty((0,len(bin_val),4),dtype=float)
    d_bintime=numpy.empty((0,totalcell,2),dtype=float)
    for frame in range(init1,len(traj)-1):
        d_bin=numpy.empty((0,2),dtype=float)
        container=[]
        #Lx =traj[frame].configuration.box[0]
        #Ly =traj[frame].configuration.box[1]
        #Lz =traj[frame].configuration.box[2]
        msd_box=[]
        time = frame*dt*period
        
                
        
        posB=traj[frame].particles.position[N_constraint:N_constraint+N_remainz]
        imageB=traj[frame].particles.image[N_constraint:N_constraint+N_remainz]
        posBplus=traj[frame+1].particles.position[N_constraint:N_constraint+N_remainz]
        imageBplus_1=traj[frame+1].particles.image[N_constraint:N_constraint+N_remainz]

        posBpos = posB[posB[:,0]>=0]       
        posBpluspos = posBplus[posB[:,0]>=0]       
        posBnegt = posB[posB[:,0]<=0]       
        posBplusnegt = posBplus[posB[:,0]<=0]       
        posBnegt[:,0]=-posBnegt[:,0]
        posBplusnegt[:,0]=-posBplusnegt[:,0]

        
        xiB = ((posBnegt[:,0])/lcx).astype(int)
        xiB=numpy.where(xiB<0,xiB+1,xiB)
        xiB=numpy.where(xiB==cellnox,xiB-1,xiB)
        mB=xiB[:]
        
        xiB1 = ((posBpos[:,0])/lcx).astype(int)
        xiB1=numpy.where(xiB1<0,xiB1+1,xiB1)
        xiB1=numpy.where(xiB1==cellnox,xiB1-1,xiB1)
        mB1=xiB1[:]
        #print(posBplus.shape,posBplusnegt.shape) 
        for x in range(totalcell):
                                  
            #print(idsB1.shape,posBpluspos.shape) 
            idsB=numpy.where(mB==x)
            idsB=numpy.unique(idsB[0])
            idsB1=numpy.where(mB1==x)
            
            #print(idsB1.shape,posBpluspos.shape) 
            idsB1=numpy.unique(idsB1[0])
            #print(idsB1.shape,posBpluspos.shape) 
            #rhoB = len(idsBplus)/delV
            #posxyzB = numpy.take(posB,idsBplus,axis=0)
            posBpossel = numpy.take(posBpos,idsB1,axis=0)
            posBpluspossel = numpy.take(posBpluspos,idsB1,axis=0)
            posBnegtsel = numpy.take(posBnegt,idsB,axis=0)
            posBplusnegtsel = numpy.take(posBplusnegt,idsB,axis=0)

            diffnegt = (box.wrap(posBplusnegtsel - posBnegtsel))/(period*dt)

            diffpos = (box.wrap(posBpluspossel - posBpossel))/(period*dt)
            avvnegt = numpy.nanmean(diffnegt,axis=0)
            avvpos = numpy.nanmean(diffpos,axis=0)

            d_bin=numpy.append(d_bin,numpy.asarray((avvpos[0],avvnegt[0])).reshape(-1,2),axis=0)  
        print(d_bin.shape) 
        d_bintime=numpy.append(d_bintime,[d_bin],axis=0)
        #print(d_bintime.shape)    
    d_samplebin=numpy.append(d_samplebin,[numpy.nanmean(d_bintime,axis=0)],axis=0)
    print(d_samplebin.shape)
    d_samplevar = numpy.append(d_samplevar,[numpy.nanvar(d_bintime,axis=0)],axis=0)
data9 = numpy.asarray((d_samplebin[:,:,0].T,d_samplebin[:,:,1].T)).T
data10= numpy.asarray((d_samplevar[:,:,0].T,d_samplevar[:,:,1].T)).T

print(data9.shape)
print(data9)
#data = numpy.nanmean(msd_samplebin,axis=0)
data = numpy.nanmean(data9,axis=0)
#datastd = numpy.nanstd(msd_samplebin,axis=0)
datastd = numpy.nanstd(data9,axis=0)


#print(datastd.shape,data.shape)
#exit()
print(len(bin_val),len(d_samplebin[:]))
#data1 = numpy.array((bin_val,data[:,0],data[:,1],data[:,2],data[:,0]/(2*period*dt),data[:,1]/(2*period*dt),data[:,2]/(2*period*dt))).T
#data2 = numpy.concatenate((data1,datastd[:,1:]),axis=1)

bin_val = bin_val.reshape(totalcell,1)
data2 = numpy.concatenate((bin_val,data,datastd),axis=1)

print(data2.shape,bin_val.shape)

#print(data1)
filename3 = "Vtry_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"
with open(filename3, 'wb+') as f2:
    numpy.savetxt(f2, data2)
       
end = time1.perf_counter()
print(abs(end-begining))



