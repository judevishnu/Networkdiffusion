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
rmax=2.2
#rmax=2
#rmax=4.5
#timeint=(math.ceil((len(traj)-init)/step))
sum_Interface1=0.0
sum_Interface2=0.0


#data3 = numpy.empty((0,timeint,totalcell,2),dtype=float)
init=1000
#timeint=(math.ceil((len(traj)-init)/step))
sum_Interface1=0.0
sum_Interface2=0.0
#print(timeint)
init1=1000



lentraj=int((1.28e9)/period)

begining=time1.perf_counter()
sample = sys.argv[1]
Ntot = sys.argv[2]
fraction = sys.argv[3]
densratio = sys.argv[4]
molefrac = sys.argv[5]

def Interface_calc(frame,step):
        filename2 = "density_vs_xbox"+"_Isodiamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".bin"
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

filename = "trajectoryContinueIso"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
traj=gsd.hoomd.open(name=filename, mode='rb')
Lx=traj[0].configuration.box[0]
Ly=traj[0].configuration.box[1]
Lz=traj[0].configuration.box[2]
box = freud.box.Box(Lx = 10*Lx, Ly= Ly, Lz =Lz)
Lxby2 = Lx/2.
Lyby2 = Ly/2.
Lzby2 = Lz/2.
Ny = int(Ly/sigma[0][0])
poly_len = 102
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


data2 = numpy.zeros((int((end_index-start_index)/step),5),dtype=float)

for frame in range(init+start_index,init+end_index,step):

        print("sample :",sample,"rank :",rank,"frame : ",frame,"int((frame-(start+start_index))/step) :",int((frame-(init+start_index))/step),"start_index :",start_index,"end_index :",end_index,flush=True)

        if int((frame-(init+start_index))/step) >=len(data2):
                break

        sum_Interface1,sum_Interface2=Interface_calc(frame,step)

        print("Interfaces",sum_Interface1,sum_Interface2)
        leftregion = sum_Interface1-2.5*sigma[0][0]
        rightregion = sum_Interface2+2.5*sigma[0][0]
        print("Regions",leftregion,rightregion,-Lxby2,Lxby2)

        pos_gel=traj[frame].particles.position[:N_constraint]
        pos_poly=traj[frame].particles.position[N_constraint:]
        timestep=traj[frame].configuration.step
        real_pos =pos_poly
        real_pos_poly = pos_poly[:]+numpy.array([Lx,Ly,Lz])*traj[frame].particles.image[N_constraint:]
        real_pos = real_pos.reshape(-1,shape).reshape(shape1,poly_len,dimension3)
        poly_position =real_pos_poly.reshape(-1,shape).reshape(shape1,poly_len,dimension3) #Reshape position array along lenght of the polymers
        R_mean = numpy.mean(poly_position,axis=1)
        keys = numpy.arange(0,len(R_mean),1,dtype=int)
        keys= numpy.repeat(keys,poly_len,axis=0).reshape(shape1,poly_len)
        keys=keys.reshape(-1)

        R_mean1= numpy.repeat(R_mean,poly_len,axis=0).reshape(shape1,poly_len,dimension3)

        dr = poly_position -R_mean1
        Rg2=numpy.mean(numpy.power(numpy.linalg.norm(dr,axis=2),2),axis=1)


        ids=numpy.where(((pos_poly[:,0]>sum_Interface1) & (pos_poly[:,0]<sum_Interface2)))
        ids=ids[0]
        points=numpy.take(pos_poly,ids,axis=0)
        key = numpy.take(keys,ids,axis=0)
        key=key.reshape(-1)
        points=points.reshape(-1,dimension3)



        if points.size!=0:
            system=freud.AABBQuery(box,points)
            cl=freud.cluster.Cluster()
            cl.compute(system,key,neighbors={"r_max":rmax})
            num_cluster=cl.num_clusters
        else:
            num_cluster=0
        trutharray=numpy.empty((0),dtype=bool)
        trutharray1=numpy.repeat(False,cl.num_clusters)
        numclust=0
        for u in range(num_cluster):
            unique_id=numpy.asarray(cl.cluster_keys[u])
            #if ((len(unique_id)>int(poly_len/2))&(int(poly_len*10)>len(unique_ids))):
            if (len(unique_id)>=int(poly_len)):
                numclust+=1
                trutharray1[u]=True


            unique_id1=numpy.unique(unique_id)
            cluster_points=numpy.take(real_pos,unique_id1,axis=0)
            #cluster_points=numpy.take(poly_position,unique_id1,axis=0)
            temporaryp=cluster_points.reshape(-1,dimension3)
            cluster_pointsx=temporaryp[:,0]
            #print(cluster_pointsx)

            truthtable1=((cluster_pointsx>leftregion)&(cluster_pointsx<sum_Interface1))
            #print(truthtable1)
            truthtable2=((cluster_pointsx<rightregion)&(cluster_pointsx>sum_Interface2))
            truthtable3=((cluster_pointsx<sum_Interface2-50)&(cluster_pointsx>sum_Interface1+50))
            #print(truthtable2)

            #nettruth1=numpy.logical_or(truthtable1)
            nettruth1=truthtable1.any()
            #nettruth2=numpy.logical_or(truthtable2)
            nettruth2=truthtable2.any()
            nettruth3=truthtable3.any()
            #print(nettruth1,nettruth2)

            comptruth=numpy.logical_and(numpy.logical_and(nettruth2,nettruth1),nettruth3)
            #print(nettruth1,nettruth2,comptruth)
            #exit()
            trutharray=numpy.append(trutharray,[comptruth],axis=0)

        #truthval=numpy.logical_or(trutharray)
        cl_props = freud.cluster.ClusterProperties()
        cl_props.compute((box,points),cl.cluster_idx)
        sel_Rg_ids = trutharray1*numpy.logical_not(trutharray)
        #cl.cluster_keys[sel_Rg_ids]
        Rgofcluster=cl_props.gyrations[sel_Rg_ids,0,0]
        RgofCl=cl_props.radii_of_gyration[sel_Rg_ids]
        MeanRg  = numpy.nanmean(Rgofcluster)
        MeanRgCl  = numpy.nanmean(numpy.square(RgofCl))
        # append1=numpy.array((float(molefrac),MeanRg,MeanRgCl)).T
        # append2=numpy.array((float(molefrac),numclust)).T
        data2[int((frame-(init+start_index))/step),0]=timestep

        data2[int((frame-(init+start_index))/step),1]=MeanRg
        data2[int((frame-(init+start_index))/step),2]=MeanRgCl
        data2[int((frame-(init+start_index))/step),3]=numclust


        # data_timeRg = numpy.append(data_timeRg,[append1],axis=0)
        # data_timenum = numpy.append(data_timenum,[append2],axis=0)
        truthval=trutharray.any()
        if truthval==True:
            data2[int((frame-(init+start_index))/step),4] = trutharray.sum()
        else:
            data2[int((frame-(init+start_index))/step),4] = 0

all_data2 = comm.gather(data2,root=0)
if rank==0:
    print("sample :",flush=True)
    datatime_concatmean=numpy.concatenate(all_data2,axis=0)
    filename = "PercolationMPI"+str(sample)+"_Isodiamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_rmax_"+str(rmax)+"_kT_"+str(kbT)+".txt"

    with open(filename, 'w+') as f3:
        numpy.savetxt(f3, datatime_concatmean)
