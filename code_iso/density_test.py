#!/usr/bin/python3
import hoomd
import hoomd.md
import gsd
import gsd.hoomd
import time  as time1
import numpy
import random 
import math
import freud
import sys

########################################################################
#           Set parameters
#######################################################################
samples=4
#samples=101
dia=[]
dia.append(1.0)
dia.append(1.0)
sigma = [] # interaction parameter sigma 

sigma.append([(dia[0]+dia[0])/2.,(dia[0]+dia[1])/2.])
###############################################################################
dt = 0.001

#samples=int(sys.argv[1])
Ntot = sys.argv[1]
fraction=sys.argv[2]
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

sample=2

filename = "trajectoryContinueIso"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"



traj=gsd.hoomd.open(name=filename, mode='rb')
print(len(traj))

Lx=traj[0].configuration.box[0]
tempLx=traj[0].configuration.box[0]
Ly=traj[0].configuration.box[1]
Lz=traj[0].configuration.box[2]
Lxby2 = Lx/2.0 
Lyby2 = Ly/2.0 
Lzby2 = Lz/2.0 
Ny = int(Ly/sigma[0][0])
poly_len = 102
print(poly_len)
typeid =  numpy.copy(traj[0].particles.typeid)
N_remainz = numpy.count_nonzero(typeid == 1)
N_constraint = numpy.count_nonzero(typeid==0)
Nx_remainz = int(N_remainz/2/Ny/poly_len)
cutoffx = sigma[0][0]
lcutoffx =cutoffx
lengthx = int(Lxby2/lcutoffx)
    
lcx = Lxby2/lengthx
    
shiftx = int(lengthx)
cellnox= 2*lengthx
    #print(cellnox)

totalcell = cellnox
slcx = shiftx*lcx

    #print(totalcell)

count_A = numpy.empty(totalcell,dtype=int)
count_B = numpy.empty(totalcell,dtype=int)
count_AB = numpy.empty(totalcell,dtype=int)
bin_val = numpy.empty(totalcell,dtype=float)
print(len(bin_val))
data1 = numpy.empty((0,len(bin_val),3),dtype=float)
step=1
start=1
t=(math.ceil((len(traj)-start)/step))
#t=int(len(traj)/step)
    #print(t)
delV = Ly*Lz*cutoffx
for x in range(totalcell):
    bin_val[x] = (x+0.5)*lcutoffx-Lxby2
    print(len(traj),Lxby2)

data2 = numpy.empty((0,t,len(bin_val),3),dtype=float)
###################################################################
begini = time1.perf_counter()

for sample in range(1,samples):
    
    

    
    filename = "trajectoryContinueIso"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
    print(filename)
    traj=gsd.hoomd.open(name=filename, mode='rb')
    Lx=traj[0].configuration.box[0]
    Ly=traj[0].configuration.box[1]
    Lz=traj[0].configuration.box[2]
    print(Lx,Ly,Lz) 
    data1 = numpy.empty((0,len(bin_val),3),dtype=float)
    for frame in range(start,len(traj),step):
        
        time = frame*dt*period
        print(time)
        posA=traj[frame].particles.position[:N_constraint]
        posB=traj[frame].particles.position[N_constraint:N_constraint+N_remainz]
        pos=traj[frame].particles.position
        image=traj[frame].particles.image
        count_A.fill(0)
        count_B.fill(0)
        count_AB.fill(0)
        
        
        xi = ((slcx +pos[:,0])/lcx).astype(int)
    
        xi=numpy.where(xi<0 ,xi+1, xi)
        xi=numpy.where(xi==cellnox,xi-1,xi)
        

        xiB = ((slcx +posB[:,0])/lcx).astype(int)
        xiB=numpy.where(xiB<0,xiB+1,xiB)
        xiB=numpy.where(xiB==cellnox,xiB-1,xiB)
        
        xiA = ((slcx +posA[:,0])/lcx).astype(int)
        xiA=numpy.where(xiA<0,xiA+1,xiA)
        xiA=numpy.where(xiA==cellnox,xiA-1,xiA)
        
        mA = xiA[:]
        mB = xiB[:]
        m = xi[:]

        #print(len(count_A))
        count_A=numpy.bincount(mA,minlength = len(bin_val))
        
        #print(count_A)
        
        count_B=numpy.bincount(mB,minlength = len(bin_val))
        
        count_AB=numpy.bincount(m,minlength = len(bin_val))
        """
        for x in numpy.nditer(mA,order='C'): 
            count_A[x]=count_A[x]+1
        for y in numpy.nditer(mB,order='C'):
            count_B[y]=count_B[y]+1
        for z in numpy.nditer(m,order='C'):
            count_AB[z]=count_AB[z]+1
        """
    
        densityA = count_A[:]/delV
        densityB = count_B[:]/delV
    
        #molefA = numpy.divide(count_A,count_AB)
        #molefB = numpy.divide(count_B,count_AB)
        data = numpy.asarray((bin_val,densityA,densityB)).T
        
        data1=numpy.append(data1,[data],axis=0)
        #print(densityA,count_A) 
    print(data1.shape)    
    data2 = numpy.append(data2,[data1],axis=0)     

data3=numpy.nanmean(data2,axis=0)
print(data3[0,:,0].T,data3[0,:,1].T,data3[0,:,2].T)
data4 = numpy.nanstd(data2,axis=0)
print(data3[0])
print(data3.shape,data4.shape)
data5 = numpy.asarray((data3[:,:,0].T,data3[:,:,1].T,data3[:,:,2].T,data4[:,:,1].T,data4[:,:,2].T)).T
print(data5)
print(data5.shape)
print(len(data4),len(data3))
#print(data5)
end = time1.perf_counter()

print(abs(begini-end))

filename2 = "density_vs_xbox"+"_Isodiamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".bin"
filename3 = "density_vs_xbox"+"_Isodiamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"

with open(filename2, 'wb+') as f2:                                                                                                                                                                           
    numpy.save(f2, data5)
                                                                                                                                                                                           


with open(filename3, 'w+') as f3:
    for i in range(len(data5)):
        numpy.savetxt(f3, data5[i])
        f3.write('\n')

################################################
