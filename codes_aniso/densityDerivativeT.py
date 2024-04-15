#!/usr/bin/python3
import hoomd
import hoomd.md
import gsd
import gsd.hoomd
import time
import numpy
import random 
import math
import freud
import derivative
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
dt = 0.001

Ntot = sys.argv[1]
fraction = sys.argv[2]
densratio = sys.argv[3]
molefrac  = sys.argv[4]
def Interface_calc(frame,step):
        filename2 = "density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".bin"
        data = numpy.load(filename2)
        box_x=data[init1-1+int((frame-init1)/step),:,0]
        Len = int(len(box_x)/2)
        gel_dens=data[init1-1+int((frame-init1)/step),:,1]
        poly_dens=data[init1-1+int((frame-init1)/step),:,2]
        abs_diff=numpy.fabs(gel_dens[0:Len]-poly_dens[0:Len]) #absolute value difference between number
                                                                                                #densities of gel and polymer
        abs_diff1=numpy.fabs(gel_dens[Len:]-poly_dens[Len:])

        min_val=numpy.min(abs_diff) # Minimum of differences means, the values of densities were close enough
        min_val1=numpy.min(abs_diff1) # Minimum of differences means, the values of densities were close enough

        index=numpy.where(abs_diff==min_val) ## Index of the min_val in the array, this corrsponds to the box_x value where densities
        index1=numpy.where(abs_diff1==min_val1) ## Index of the min_val in the array, this corrsponds to the box_x value where densities
                                             ##approximately intersect
        Interface1 = box_x[index[0][0]]
        Interface2 = box_x[Len+index1[0][0]]
        return Interface1,Interface2



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
        

data1 = numpy.empty((0,7))

timeint=(math.ceil((len(traj)-init)/step))
sum_Interface1=0.0
sum_Interface2=0.0
print(timeint)
init1=1501
print(len(traj))



filename2 = "density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".bin"
data = numpy.load(filename2)
timepts = numpy.arange(1501,1600)*period*0.001
totlen = len(data[0,:,0])
findata = numpy.empty((0,3),dtype=float)
for binnum in range(totlen):
    print(binnum,totlen,data.shape)
    #print(timepts.shape,poly_dens.shape)
    box_x=data[0,:,0]
    Len = int(len(box_x)/2)
    
    gel_dens=data[int(init1-1):,binnum,1]
    std_geldens = data[int(init1-1):,binnum,3]
    poly_dens=data[int(init1-1):,binnum,2]
    std_polydens = data[int(init1-1):,binnum,4]
    frames=gel_dens.shape
    print(timepts.shape,poly_dens.shape)
        
    sg = derivative.SavitzkyGolay(left=3, right=3, order=3, periodic=False)
    #derivative0  = sg.d(gel_dens,timepts)
    derivative1  = sg.d(poly_dens,timepts)
    
    meanderi = numpy.nanmean(derivative1)
    stdderi = numpy.nanstd(derivative1)
    binval = box_x[binnum]
    avpoly = numpy.nanmean(poly_dens)
    tmp = numpy.asarray((binval,avpoly,stdderi)).reshape(-1,3)
    findata = numpy.append(findata,tmp,axis=0)
   
filename3 = "RhoDerivative_vs_bin1_"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"


with open(filename3, 'w+') as f2:
        numpy.savetxt(f2, findata)
        


