#!/usr/bin/python3

import gsd
import gsd.hoomd
import time as time1
import numpy
import random
import math
import freud
import sys
import scipy.spatial as spatial
import scipy.spatial.distance as dist
from mpi4py import MPI

############################################
########################################################################
#           Set parameters
#######################################################################
dia=[]
dia.append(1.0)
dia.append(1.0)
sigma = [] # interaction parameter sigma

sigma.append([(dia[0]+dia[0])/2.,(dia[0]+dia[1])/2.])
###############################################################################
dt = 0.001
period = 8e5
kbT=1.0
step=1
init=1
init1=1



lentraj=int((1.28e9)/period)

begining=time1.perf_counter()
sample = sys.argv[1]
Ntot = sys.argv[2]
fraction = sys.argv[3]
densratio = sys.argv[4]
molefrac = sys.argv[5]

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

filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
traj=gsd.hoomd.open(name=filename, mode='rb')
Lx=traj[0].configuration.box[0]
Ly=traj[0].configuration.box[1]
Lz=traj[0].configuration.box[2]
points = traj[0].particles.position
bonds_array = traj[0].bonds.group
bonds_array=numpy.sort(bonds_array,axis=1)
bonds_array=numpy.unique(bonds_array,axis=0)
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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

elements_per_process = (len(traj)-init)//size
start_index = rank*elements_per_process
end_index = (rank+1)*elements_per_process
if rank == size-1:
    end_index = start_index +len(traj) -(start_index+init)
    print("rank :",rank,"size-1 :",size-1,"start_index :",start_index,"end_index :",end_index,flush=True)


data2 = numpy.zeros((int((end_index-start_index)/step),3),dtype=float)
system = freud.AABBQuery.from_system((box,points))
distances = numpy.linalg.norm(box.wrap(points[bonds_array[:, 1]] - points[bonds_array[:, 0]]),axis=1)
neighbors = freud.locality.NeighborList.from_arrays(len(points),len(points),
                    bonds_array[:, 0],
                    bonds_array[:, 1],
                    distances,
                )

cl = freud.cluster.Cluster()
cl1=cl.compute(system=system, neighbors=neighbors)


for frame in range(init+start_index,init+end_index,step):

        
        print("sample :",sample,"rank :",rank,"frame : ",frame,"int((frame-(start+start_index))/step) :",int((frame-(init+start_index))/step),"start_index :",start_index,"end_index :",end_index,flush=True)

        if int((frame-(init+start_index))/step) >=len(data2):
            print( "rank:",rank,int((frame-(init+start_index))/step), ">=", len(data2))
            break

        I1,I2=Interface_calc(frame,step)
        
        print("Interfaces",I1,I2)
        I1arr = numpy.asarray([[I1,0,0]])
        I2arr = numpy.asarray([[I2,0,0]])
        points=traj[frame].particles.position
        #pos_poly=traj[frame].particles.position[N_constraint:]
        timestep=dt*traj[frame].configuration.step

        
        #system = freud.AABBQuery.from_system((box,points))
        distances = numpy.linalg.norm(box.wrap(points[bonds_array[:, 1]] - points[bonds_array[:, 0]]),axis=1)
        neighbors = freud.locality.NeighborList.from_arrays(len(points),len(points),
                    bonds_array[:, 0],
                    bonds_array[:, 1],
                    distances,
                )

        cl = freud.cluster.Cluster()
        cl.compute((box,points), neighbors=neighbors)

        
        cl_props = freud.cluster.ClusterProperties()
        cl_props.compute((box,points), cl.cluster_idx)
        compoly = cl_props.centers_of_mass
        print(compoly.shape)
        compoly = numpy.asarray(compoly[1:])
        #"""
        poly_ids1 = numpy.where(((compoly[:,0]>=I1) &(compoly[:,0]<0)))[0]
        poly_ids2 = numpy.where(((compoly[:,0]<=I2) &(compoly[:,0]>0)))[0]

        com_sel1 = compoly[poly_ids1]
        com_sel2 = compoly[poly_ids2]
        if len(com_sel1)!=0:
            depthx1=numpy.fabs(box.wrap(I1arr - com_sel1)[:,0])
            meandepthx1=numpy.nanmean(depthx1,axis=0)
            NcA1=len(com_sel1)/Ly/Lz
            print(meandepthx1)
        else:
            meandepthx1 =0
            NcA1 = 0
        
        if len(com_sel2)!=0:
            depthx2=numpy.fabs(box.wrap(I2arr - com_sel2)[:,0])
            meandepthx2=numpy.nanmean(depthx2,axis=0)
            NcA2=len(com_sel2)/Ly/Lz
        else:
            meandepthx2 = 0
            NcA2 = 0

        
        data2[int((frame-(init+start_index))/step),0]=timestep
        data2[int((frame-(init+start_index))/step),1]=(meandepthx1+meandepthx2)/2
        data2[int((frame-(init+start_index))/step),2]=(NcA2+NcA1)/2
        #"""
all_data2 = comm.gather(data2,root=0)
if rank==0:
    print("sample :",sample,flush=True)
    datatime_concatmean=numpy.concatenate(all_data2,axis=0)
    
   
    filename1 = "Diffusiondepth"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"


    with open(filename1, 'w+') as f3:
        numpy.savetxt(f3, datatime_concatmean)
    print("saved file",filename1)
