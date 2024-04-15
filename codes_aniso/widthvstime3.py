#!/usr/bin/python3

import gsd
import gsd.hoomd
import time
import numpy
import random 
import math
import freud
import sys
import derivative
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
dt = 0.001

Ntot = sys.argv[1]
fraction = sys.argv[2]
densratio = sys.argv[3]
molefrac  = sys.argv[4]



period = 8e5
kbT=1.0
cutoff = 0.05*sigma[0][0]
#samples=10
sample=2
filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
traj=gsd.hoomd.open(name=filename, mode='rb')
print(len(traj))
#exit()
Lx=traj[0].configuration.box[0]
Ly=traj[0].configuration.box[1]
Lz=traj[0].configuration.box[2]
box = freud.box.Box(Lx = Lx, Ly= Ly, Lz =Lz)
Lxby2 = Lx/2.
Lyby2 = Ly/2.
Lzby2 = Lz/2.
Ny = int(Ly/sigma[0][0])
poly_len = int(Lz/sigma[0][0])
typeid =  numpy.copy(traj[0].particles.typeid)
N_remainz = numpy.count_nonzero(typeid == 1)
N_constraint = numpy.count_nonzero(typeid==0)
Nx_remainz = int(N_remainz/2/Ny/poly_len)
shape = int(poly_len*3)
shape1 = int(N_remainz/poly_len)
dimension3=3
step=1
init =350
timeint=(math.ceil((len(traj)-init)/step))
sum_Interface1=0.0
sum_Interface2=0.0

lcutoffx =3.0*sigma[0][0]
lengthx = int(Lxby2/lcutoffx)
lcx = Lxby2/lengthx
    
shiftx = int(lengthx)
cellnox= 2*lengthx

totalcell = cellnox
slcx = shiftx*lcx

box_data = numpy.empty((totalcell),dtype=float)
#for x in range(totalcell):
#    box_data[x] = x*lcutoffx-Lxby2
#    print(box_data[x])


Interface1 =[box_data]
print(len(Interface1))
Interface1 = numpy.tile(Interface1,(N_remainz,1)).T
        

data1 = numpy.empty((0,3))

timeint=(math.ceil((len(traj)-init)/step))
sum_Interface1=0.0
sum_Interface2=0.0
print(timeint)
init1=1
print(len(traj))


for frame in range(init1,len(traj),step):
    traj=gsd.hoomd.open(name=filename, mode='rb')
    time1 = traj[frame].configuration.step*dt
    filename2 = "density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".bin"
    data = numpy.load(filename2)
    box_x=data[int((frame-init1)/step+init1-1),:,0]
    Len = int(len(box_x)/2)
    
    gel_dens=data[int(init1-1+(frame-init1)/step),:,1]
    std_geldens = data[int(init1-1+(frame-init1)/step),:,3]
    poly_dens=data[int(init1-1+(frame-init1)/step),:,2]
    std_polydens = data[int(init1-1+(frame-init1)/step),:,4]
    
    diff=numpy.flip(gel_dens[0:Len]-poly_dens[0:Len]) #absolute value difference between number 
                                                                                                #densities of gel and polymer
    diff1=gel_dens[Len:]-poly_dens[Len:]
    sum0=numpy.flip(gel_dens[0:Len]+poly_dens[0:Len]) 
    sum1=gel_dens[Len:]+poly_dens[Len:]
    m1 = diff1/sum1
    m0 = diff/sum0
    #print( diff.shape,diff1.shape)
    min_val=numpy.min(m0) # Minimum of differences means, the values of densities were close enough
    min_val1=numpy.min(m1) # Minimum of differences means, the values of densities were close enough
    
    max_val=numpy.max(m0) # Minimum of differences means, the values of densities were close enough
    max_val1=numpy.max(m1) # Minimum of differences means, the values of densities were close enough
    
    Diff = 0.5*(m0+m1)
    max_val0 = numpy.max(numpy.asarray([max_val,max_val1]))
    min_val0 = numpy.min(numpy.asarray([min_val,min_val1]))
    max_val2 = numpy.max(Diff)
    min_val2 = numpy.min(Diff)

    sg = derivative.SavitzkyGolay(left=3, right=3, order=3, periodic=False)
    #sg =  derivative.Kalman(alpha=0.02)
    #sg =derivative.TrendFiltered(alpha=1e-3, order=2, max_iter=1e6)
    derivative0  = sg.d(m0,box_x[0:Len])
    derivative1  = sg.d(m1,box_x[Len:])
    Deriv = sg.d(Diff,box_x[Len:])
    maxderiv1 = numpy.max(numpy.abs(derivative0))
    maxderiv2 = numpy.max(numpy.abs(derivative1))
    maxDeriv = numpy.max(numpy.fabs(Deriv))
    w1 = (max_val-min_val)/maxderiv1
    w2 = (max_val1-min_val1)/maxderiv2
    #w1 = (max_val0-min_val0)/maxderiv1
    #w2 = (max_val0-min_val0)/maxderiv2

    w1 = w1**2
    w2 = w2**2
    w =numpy.sqrt(0.5*(w1+w2))
    maxD = numpy.nanmean(numpy.asarray([maxderiv1,maxderiv2]))
    #maxD =maxDeriv
    #w = (max_val2-min_val2)/maxDeriv
    temp=numpy.asarray((time1,w,maxD)).T
    print(max_val1,max_val,w1,w2,w,maxD)
    #print(temp)
    data1=numpy.append(data1,[temp],axis=0)

filename3 = "width_vs_time3"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"


with open(filename3, 'w+') as f2:
        numpy.savetxt(f2, data1)
        


