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
from scipy import interpolate
from sympy import symbols, Eq, solve

#######################################################################
##  Equation of a line
######################################################################
def findintercept(slope,xpoint,ypoint):
    c = ypoint-slope*xpoint
    return c
########################################################################
#           Set parameters
#######################################################################
samples=11
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
time_list=[]
period = 8e5

sample=1

filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"



traj=gsd.hoomd.open(name=filename, mode='rb')
print(len(traj))

Lx=traj[0].configuration.box[0]
Ly=traj[0].configuration.box[1]
Lz=traj[0].configuration.box[2]
Lxby2 = Lx/2.0 
Lyby2 = Ly/2.0 
Lzby2 = Lz/2.0 
Ny = int(Ly/sigma[0][0])
poly_len = int(Lz/sigma[0][0])
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

#nbin = int(sys.argv[5])
nbin = float(sys.argv[5])
cutoffz = float(Lzby2/nbin)
lcutoffz =cutoffz
lengthz = int(Lzby2/lcutoffz)
print(lengthz)
lcz = Lzby2/lengthz
shiftz = int(lengthz)
cellnoz= 2*lengthz

cutoffy = cutoffz
lcutoffy =cutoffy
lengthy = int(Lzby2/lcutoffy)

print(lengthy)
lcy = Lzby2/lengthy
shifty = int(lengthy)
cellnoy= 2*lengthy





print("sub_Lx,sub_Ly,sub_Lz",cutoffx,cutoffy,cutoffz)
print("Lx,Ly,Lz",Lx,Ly,Lz)



print("cellno_z,cellno_y,cellno_x",cellnoz,cellnoy,cellnox)

totalcel = cellnox*cellnoy*cellnoz
slcx = shiftx*lcx
slcy = shifty*lcy
slcz = shiftz*lcz



count_A = numpy.empty(cellnox,dtype=int)
count_B = numpy.empty(cellnox,dtype=int)
count_AB = numpy.empty(cellnox,dtype=int)
bin_val = numpy.empty(cellnox,dtype=float)
print(len(bin_val))
data1 = numpy.empty((0,len(bin_val),3),dtype=float)
step=5
start=1400
t=(math.ceil((len(traj)-start)/step))

delV = cutoffy*cutoffz*cutoffx
for x in range(cellnox):
    bin_val[x] = (x+0.5)*lcutoffx-Lxby2
    print(len(traj),Lxby2)
nbin_array = (1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10)

begini = time1.perf_counter()
#for nbin in nbin_array[:]:

data3 = numpy.empty((0,t,cellnoy*cellnoz,len(bin_val),3),dtype=float)
for sample in range(1,samples):
    filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
    print(filename)
    traj=gsd.hoomd.open(name=filename, mode='rb')
    Lx=traj[0].configuration.box[0]
    Ly=traj[0].configuration.box[1]
    Lz=traj[0].configuration.box[2]
    print(Lx,Ly,Lz)
    data2 = numpy.empty((0,cellnoy*cellnoz,len(bin_val),3),dtype=float)
    for frame in range(start,len(traj),step):
        
        time = frame*dt*period
        posA=traj[frame].particles.position[:N_constraint]
        posB=traj[frame].particles.position[N_constraint:N_constraint+N_remainz]
        pos=traj[frame].particles.position
        selA = numpy.where(((-Ly/2<=posA[:,1])&(posA[:,1]<=Ly/2)))[0]
        selB = numpy.where(((-Ly/2<=posB[:,1])&(posB[:,1]<=Ly/2)))[0]

        #selA = numpy.where(((-Lz/2<=posA[:,1])&(posA[:,1]<=Lz/2)))[0]
        #selB = numpy.where(((-Lz/2<=posB[:,1])&(posB[:,1]<=Lz/2)))[0]
        image=traj[frame].particles.image
        posA = posA[selA]
        posB = posB[selB]
        
        yiB = ((slcy +posB[:,1])/lcy).astype(int)
        yiB=numpy.where(yiB<0,yiB+1,yiB)
        yiB=numpy.where(yiB==cellnoy,yiB-1,yiB)
        yiA = ((slcy +posA[:,1])/lcy).astype(int)
        yiA=numpy.where(yiA<0,yiA+1,yiA)
        yiA=numpy.where(yiA==cellnoy,yiA-1,yiA)


        ziA = ((slcz +posA[:,2])/lcz).astype(int)
        ziA=numpy.where(ziA<0,ziA+1,ziA)
        ziA=numpy.where(ziA==cellnoz,ziA-1,ziA)
        ziB = ((slcz +posB[:,2])/lcz).astype(int)
        ziB=numpy.where(ziB<0,ziB+1,ziB)
        ziB=numpy.where(ziB==cellnoz,ziB-1,ziB)
        
        
        subboxidA = ziA+yiA*cellnoz
        subboxidB = ziB+yiB*cellnoz

        
        data1 = numpy.empty((0,len(bin_val),3),dtype=float)
        for subbox in range(cellnoy*cellnoz):
            subposidA = numpy.where(subboxidA==subbox)[0]
            subposidB = numpy.where(subboxidB==subbox)[0]
            #exit()
            #count_A.fill(0)
            #count_B.fill(0)
            subboxposB = numpy.take(posB,subposidB,axis=0)
            subboxposA = numpy.take(posA,subposidA,axis=0)
            xiB = ((slcx +subboxposB[:,0])/lcx).astype(int)
            xiB=numpy.where(xiB<0,xiB+1,xiB)
            xiB=numpy.where(xiB==cellnox,xiB-1,xiB)
        
            xiA = ((slcx +subboxposA[:,0])/lcx).astype(int)
            xiA=numpy.where(xiA<0,xiA+1,xiA)
            xiA=numpy.where(xiA==cellnox,xiA-1,xiA)
        
            mA = xiA[:]
            mB = xiB[:]
                       
            count_A=numpy.bincount(mA,minlength = len(bin_val))
        
            
        
            count_B=numpy.bincount(mB,minlength = len(bin_val))
        
            densityA = count_A[:]/delV
            densityB = count_B[:]/delV
            data = numpy.asarray((bin_val,densityA,densityB)).T
            data1=numpy.append(data1,[data],axis=0)
            
        data2 = numpy.append(data2,[data1],axis=0)     
        
    data3=numpy.append(data3,[data2],axis=0)
print(data3.shape)
tmpdata = numpy.nanmean(data3,axis=1)
print(tmpdata.shape)
avgdensity_time = numpy.nanmean(tmpdata,axis=0)
print(avgdensity_time.shape)

cutoffz = "%.5f" % cutoffz
filename2 = "subboxdifflylz_density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_nbin_"+str(nbin)+"_kT_"+str(kbT)+"_boxwidth_"+str(cutoffz)+".bin"
filename3 = "subboxdifflylz_density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_nbin_"+str(nbin)+"_kT_"+str(kbT)+"_boxwidth_"+str(cutoffz)+".txt"



with open(filename2, 'wb+') as f2:                                                                                                                              
    numpy.save(f2, avgdensity_time)
                                                                                                                                                                                               
with open(filename3, 'w+') as f3:
    for i in range(len(avgdensity_time)):
        numpy.savetxt(f3, avgdensity_time[i])
        f3.write('\n')

################################################
