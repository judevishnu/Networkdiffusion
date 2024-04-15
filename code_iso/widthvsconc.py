#!/usr/bin/python3
import gsd
import gsd.hoomd
import time  as time1
import numpy
import random 
import math
import freud
import sys
from mpi4py import MPI
import statsmodels.api as sm
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

sample=int(sys.argv[1])
Ntot = sys.argv[2]
fraction=sys.argv[3]
densratio = sys.argv[4]
molefrac  = sys.argv[5]
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
start=1400
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
widthsample = numpy.zeros(samples-1,dtype=float)
step=1
start =1000
for sample in range(1,samples):
    
    filename = "trajectoryContinueIso"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
    print(filename)
    traj=gsd.hoomd.open(name=filename, mode='rb')
    Lx=traj[0].configuration.box[0]
    Ly=traj[0].configuration.box[1]
    Lz=traj[0].configuration.box[2]
    print(Lx,Ly,Lz) 
    #data1 = numpy.empty((0,len(bin_val),3),dtype=float)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    elements_per_process = (len(traj)-start)//size
    start_index = rank*elements_per_process
    end_index = (rank+1)*elements_per_process
    if rank == size-1:
        end_index = start_index +len(traj) -(start_index+start)
        print("rank :",rank,"size-1 :",size-1,"start_index :",start_index,"end_index :",end_index,flush=True)

    data1 = numpy.zeros((int((end_index-start_index)/step),1),dtype=float)

    #data1 = numpy.zeros((i,len(bin_val),3),dtype=float)
    for frame in range(start+start_index,end_index+start,step):
        
        time = frame*dt*period
        print("sample:",sample,"rank:",rank,"time:",time,"frame:",frame,"int((frame-(start+start_index))/step):",int((frame-(start+start_index))/step),"start_index:",start_index,"end_index:",end_index)
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
        poly1st  = sm.nonparametric.lowess(data[:len(data)//2,2],data[:len(data)//2,0], frac=0.08).T
        gel1st = sm.nonparametric.lowess(data[:len(data)//2,1],data[:len(data)//2,0],frac=0.08).T
        poly2nd = sm.nonparametric.lowess(data[len(data)//2:,2],data[len(data)//2:,0],frac=0.08).T
        gel2nd = sm.nonparametric.lowess(data[len(data)//2:,1],data[len(data)//2:,0],frac=0.08).T
        
        derivativepoly1 = numpy.gradient(poly1st[1],poly1st[0])
        derivativegel1 =  numpy.gradient(gel1st[1],gel1st[0])
        derivativepoly2 = numpy.gradient(poly2nd[1],poly2nd[0])
        derivativegel2 =  numpy.gradient(gel2nd[1],gel2nd[0])

                               
        maxpoly1 = numpy.max(poly1st[1,:])
        minpoly1 = numpy.min(poly1st[1,:])
        maxgel1 = numpy.max(gel1st[1,:])
        mingel1 = numpy.min(gel1st[1,:])
            
        maxpoly2 = numpy.max(poly2nd[1,:])
        minpoly2 = numpy.min(poly2nd[1,:])
        maxgel2 = numpy.max(gel2nd[1,:])
        mingel2 = numpy.min(gel2nd[1,:])
                

        maxpoly1deri = numpy.max(numpy.abs(derivativepoly1))
        maxpoly2deri = numpy.max(numpy.abs(derivativepoly2))
        maxgel1deri = numpy.max(numpy.abs(derivativegel1))
        maxgel2deri = numpy.max(numpy.abs(derivativegel2))
                
        subwpoly1 = (abs(maxpoly1) - abs(minpoly1))/abs(maxpoly1deri)
        subwpoly2 = (abs(maxpoly2) - abs(minpoly2))/abs(maxpoly2deri)
        subwgel1 = (abs(maxgel1) - abs(mingel1))/abs(maxgel1deri)
        subwgel2 = (abs(maxgel2) -abs( mingel2))/abs(maxgel2deri)
            
        width1 = numpy.nanmean(numpy.asarray([abs(subwpoly1),abs(subwgel1)]))
        width2 = numpy.nanmean(numpy.asarray([abs(subwpoly2),abs(subwgel2)]))
        width = numpy.nanmean(numpy.asarray([width1,width2]))
        print(width1,width2) 
        


        data1[int((frame -(start+start_index))/step)]=width**2
        #print(densityA,count_A) 
    print(data1.shape)    
    all_data1 = comm.gather(data1,root=0)
    if rank==0:
        print("sample :",sample,flush=True)

        data_concatmean=numpy.concatenate(all_data1,axis=0)
        widthtimemean=numpy.nanmean(data_concatmean,axis=0)
        print("sample:",sample,"widthtimemean:",widthtimemean)
        widthsample[sample-1]=widthtimemean[0]
    comm.Barrier()
    


#print(abs(begini-end))
if rank==0:
    filename2 = "width2_vs_conc"+"_Isodiamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"
    print(widthsample)
    with open(filename2, 'w+') as f2:                                                                                                                                                                           
        numpy.savetxt(f2, widthsample)
                                                                                                                                                        
MPI.Finalize()
end = time1.perf_counter()

print(abs(begini-end))
################################################
