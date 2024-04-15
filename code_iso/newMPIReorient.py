import gsd
import gsd.hoomd
import time  as time1
import numpy as np
import random
import math
import freud
import sys
from numpy import  linalg as LA
from mpi4py import MPI
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
################################################################################
dt = 0.001
step = 1
#samples=int(sys.argv[1])
Ntot = sys.argv[1]
fraction=sys.argv[2]
densratio = sys.argv[3]
molefrac  = sys.argv[4]
segmentlen = int(sys.argv[5])
segment_length=segmentlen
sample=1
kbT = 1.0
##############################################################################
#################### Gyration Tensor ###########################################
#############################################################################

def cluster_props(snap,value=0):
    system = freud.AABBQuery.from_system(snap)
    typeids=snap.particles.typeid == snap.particles.types.index("B")

    num_query_points = num_points = snap.particles.N
    bonds = snap.bonds.group
    bonds =np.unique(bonds,axis=0)
    query_point_indices = bonds[:, 0]
    point_indices = bonds[:, 1]
    #print(num_query_points,num_points,len(query_point_indices),len(point_indices))
    distances = system.box.compute_distances(
        system.points[query_point_indices], system.points[point_indices]
    )
    nlist = freud.NeighborList.from_arrays(
        num_query_points, num_points, query_point_indices, point_indices, distances
    )
    cluster = freud.cluster.Cluster()

    cluster.compute(system=system, neighbors=nlist)
    if value==0:
        return cluster.num_clusters, cluster.cluster_idx, cluster.cluster_keys
    else:
        if value==1:

        
            clp = freud.cluster.ClusterProperties()
            clp.compute(system,cluster.cluster_idx)
            return clp.gyrations, clp.centers,clp.radii_of_gyration


def Rgfunc(points,box):
    
    #center_of_mass = np.nanmean(points, axis=1,keepdims=True)
    #center_of_mass=box.wrap(center_of_mass.repeat(segment_length,axis=1).reshape(-1,3))
    #center_of_mass=center_of_mass.reshape(-1,segment_length,3)
    #print(center_of_mass)
    
    cluster_id = np.arange(0,len(points)).repeat(segment_length,axis=0).reshape(-1,segment_length)
    #print(cluster_id)
    
    system = (box,points.reshape(-1,3))
    clp = freud.cluster.ClusterProperties()
    clp.compute(system,cluster_id.reshape(-1))
 
    #comwrap = box.wrap(center_of_mass.reshape(-1,3))
    
    #differences = np.apply_along_axis(box.wrap,axis=2,arr=sel_pos - center_of_mass)
    #gyration_tensor = np.nanmean(np.einsum('ijk,ijl->ijkl', differences, differences), axis=1)
    
    #print(gyration_tensor[300],clp.gyrations[300])
    #exit()
    #rg2 =gyration_tensor[:,0,0]+gyration_tensor[:,1,1]+gyration_tensor[:,2,2]

    return clp.gyrations, clp.centers,(clp.radii_of_gyration)**2


############################################################################
init = 1000
overlap=2
filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"

traj=gsd.hoomd.open(name=filename, mode='rb')
snap =traj[0]
Lx=snap.configuration.box[0]
Ly=snap.configuration.box[1]
Lz=snap.configuration.box[2]
box = freud.box.Box(Lx = Lx, Ly= Ly, Lz =Lz)
Lxby2 = Lx/2.
Lyby2 = Ly/2.
Lzby2 = Lz/2.

cutoffx = 2*sigma[0][0]
lcutoffx =cutoffx
lengthx = int(Lxby2/lcutoffx)
lcx = Lxby2/lengthx


shiftx = int(lengthx)
cellnox= 2*lengthx
totalcell = cellnox
slcx = shiftx*lcx
def wrap(arr):
    print(arr.shape)
    return box.wrap(arr)
bin_val = np.empty(totalcell,dtype=float)
for x in range(totalcell):
    bin_val[x] = (x+0.5)*lcutoffx-Lxby2

print(len(bin_val),totalcell)


poly_len = int(Lz/sigma[0][0])
typeid =  np.copy(traj[0].particles.typeid)
N_remainz = np.count_nonzero(typeid == 1)
shape1 = int(N_remainz/poly_len)
sample_orient = np.empty((0,len(bin_val),12))

start = time1.time()
numseg = int(poly_len/segmentlen)
if numseg%2!=0:
    mid  = int(numseg/2)+1
    print("mid odd",mid)
else:
    mid = np.array([int(numseg/2),int(numseg/2)+1])
    print("mid even",mid)
#exit()
print(mid,numseg)
step=1
#start = time1.time()
for sample in range(1,samples):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
    print(filename)
    traj=gsd.hoomd.open(name=filename, mode='rb')
    snap =traj[0]
    position  =snap.particles.position
    COM = np.nanmean(position,axis=0)
    relpos = position-COM
    image  =snap.particles.image
    real_pos = box.unwrap(position,image)
    num_cluster ,cluster_id,cluster_keys = cluster_props(snap,0)
    #print(cluster_keys)
    cluster_keys = np.asarray(cluster_keys[1:])
    
    num_segments = (len(cluster_keys) - overlap) // (segment_length - overlap) 
    # Create a view with overlapping segments 
    overlapping_segments = np.lib.stride_tricks.sliding_window_view(cluster_keys, (segment_length,),axis=1) 
    overlapping_segments = overlapping_segments.reshape(-1,segment_length)
    
    #sel_pos = position[overlapping_segments]
        #print(sel_pos.shape)
    #sel_pos = sel_pos.reshape(-1,segment_length,3)
    #gyration_tensor,comwrap,rg2=Rgfunc(sel_pos,box)
    #print(gyration_tensor[300])
    #sel_pos1 = relpos[overlapping_segments]
    #com=np.nanmean(sel_pos1,axis=1,keepdims=True)
    #diff = sel_pos1-com
    #diff= np.asarray(diff)
    #gyration_tensor1 = np.nanmean(np.einsum('ijk,ijl->ijkl', diff, diff), axis=1)
       
    #print(gyration_tensor1[300])
    #exit()

    if rank==0:
        print(overlapping_segments)
        print(cluster_keys)
    elements_per_process = (len(traj)-init)//size
    start_index = rank*elements_per_process
    end_index = (rank+1)*elements_per_process
    if rank == size-1:
        end_index = start_index +len(traj) -(start_index+init)
        print("rank :",rank,"size-1 :",size-1,"start_index :",start_index,"end_index :",end_index,flush=True)

    time_orient= np.zeros((int((end_index-start_index)/step),len(bin_val),12),dtype=float)


    for frame in range(init+start_index,init+end_index,step):
        print("sample :",sample,"rank :",rank,"segmentlen:",segment_length,"frame : ",frame,"int((frame-(start+start_index))/step) :",int((frame-(init+start_index))/step),"start_index :",start_index,"end_index :",end_index,flush=True)
        if int((frame-(init+start_index))/step) >=len(time_orient):
            break

        snap =traj[frame]
        position = snap.particles.position
        image = snap.particles.image
        real_pos = box.unwrap(position,image)
        sel_pos = position[overlapping_segments]
        #print(sel_pos.shape)
        sel_pos = sel_pos.reshape(-1,segment_length,3)
        gyration_tensor,comwrap,rg2=Rgfunc(sel_pos,box)
        sel_realpos = real_pos[overlapping_segments]
        Re_segment =box.wrap( (sel_pos[:,0]-sel_pos[:,segment_length-1]).reshape(-1,3))
        Re2_segment = Re_segment**2
        Remag_segment = np.sqrt(np.sum(Re2_segment,axis=1)).reshape(-1,1)
        #print(gyration_tensor.shape,comwrap.shape)
        cosangle = np.sum((Re_segment/Remag_segment)*np.asarray([1,0,0]),axis=1)
        cosangle=np.where(cosangle>1,1,cosangle)
        cosangle=np.where(cosangle<-1,-1,cosangle)
        cos2angle=cosangle**2
        #comwrap1 = comwrap.reshape(-1,segment_length,3)
        Re_segment_com1 = box.wrap((sel_pos[:,segment_length-1]-comwrap[:]).reshape(-1,3))
        Re_segment_com0 = box.wrap((sel_pos[:,0]-comwrap[:]).reshape(-1,3))
        Re_segment_com1sq =Re_segment_com1**2
        Re_segment_com0sq =Re_segment_com0**2

        xi = ((slcx +comwrap[:,0])/lcx).astype(int)
        xi=np.where(xi<0 ,xi+1, xi)
        xi=np.where(xi==cellnox,xi-1,xi)

        #print(xi.shape)
        data1 = np.empty((0,12),dtype=float)

        for i in range(totalcell):
            idi=np.where(xi == i)[0]
            
            Rg2x = gyration_tensor[idi[:],0,0]
            Rg2y = gyration_tensor[idi[:],1,1]
            Rg2z =gyration_tensor[idi[:],2,2]
            Rg2s =Rg2x+Rg2y+Rg2z
            Re2x = Re2_segment[idi[:],0]
            Re2y = Re2_segment[idi[:],1]
            Re2z = Re2_segment[idi[:],2]
            Re2s = Re2x+Re2y+Re2z

            Re2xsegcom1 = Re_segment_com1sq[idi[:],0]
            Re2ysegcom1 = Re_segment_com1sq[idi[:],1]
            Re2zsegcom1 = Re_segment_com1sq[idi[:],2]
            Re2segcom1s = Re2xsegcom1+Re2ysegcom1+Re2zsegcom1
            
            Re2xsegcom0 = Re_segment_com0sq[idi[:],0]
            Re2ysegcom0 = Re_segment_com0sq[idi[:],1]
            Re2zsegcom0 = Re_segment_com0sq[idi[:],2]
            Re2segcom0s = Re2xsegcom0+Re2ysegcom0+Re2zsegcom0



            Anumerator = 0.5*(Rg2z-Rg2x)**2+0.5*(Rg2z-Rg2y)**2 + 0.5*(Rg2y-Rg2x)**2 
            selcos2 = cos2angle[idi[:]]
            
            Rg2xav = np.nanmean(Rg2x)
            Rg2yav =  np.nanmean(Rg2y)
            Rg2zav =  np.nanmean(Rg2z)
            Rg2av = np.nanmean(Rg2s)
            
            Re2xav = np.nanmean(Re2x)
            Re2yav =  np.nanmean(Re2y)
            Re2zav =  np.nanmean(Re2z)
            Re2av = np.nanmean(Re2s)
            
            Re2xsegcom0av = np.nanmean(Re2xsegcom0)
            Re2ysegcom0av =  np.nanmean(Re2ysegcom0)
            Re2zsegcom0av =  np.nanmean(Re2zsegcom0)
            Re2segcom0av = np.nanmean(Re2segcom0s)

            Re2xsegcom1av = np.nanmean(Re2xsegcom1)
            Re2ysegcom1av =  np.nanmean(Re2ysegcom1)
            Re2zsegcom1av =  np.nanmean(Re2zsegcom1)
            Re2segcom1av = np.nanmean(Re2segcom1s)

            Anumeratorav = np.nanmean(Anumerator)
            Aav = Anumeratorav/(Rg2av)**2
            selcos2av = np.nanmean(selcos2)
            
            delRex = (3*Re2xav - Re2av)/(2*Re2av)
            delRey = (3*Re2yav - Re2av)/(2*Re2av)
            delRez = (3*Re2zav - Re2av)/(2*Re2av)
            
            delRexsegcom0 = (3*Re2xsegcom0av - Re2segcom0av)/(2*Re2segcom0av)
            delReysegcom0 = (3*Re2ysegcom0av - Re2segcom0av)/(2*Re2segcom0av)
            delRezsegcom0 = (3*Re2zsegcom0av - Re2segcom0av)/(2*Re2segcom0av)

            delRexsegcom1 = (3*Re2xsegcom1av - Re2segcom1av)/(2*Re2segcom1av)
            delReysegcom1 = (3*Re2ysegcom1av - Re2segcom1av)/(2*Re2segcom1av)
            delRezsegcom1 = (3*Re2zsegcom1av - Re2segcom1av)/(2*Re2segcom1av)

            delcosang = (3/2)*selcos2av - 0.5

           
            data = np.asarray((delRex,delRey,delRez,delRexsegcom0, delReysegcom0,delRezsegcom0,delRexsegcom1,delReysegcom1,delRezsegcom1,Re2av,delcosang,Aav)).T
            data1 = np.append(data1,[data],axis=0)

        time_orient[int((frame-(init+start_index))/step)] = data1
        
    all_timeorient  = comm.gather(time_orient,root=0)
    if rank==0:
        concat_timeorient=np.concatenate(all_timeorient,axis=0)
        sample_orient=np.append(sample_orient,[np.nanmean(concat_timeorient,axis=0)],axis=0)
comm.Barrier()
if rank==0:
    samplemean = np.nanmean(sample_orient,axis=0)
    samplestd = np.nanstd(sample_orient,axis=0)
    final = np.concatenate((bin_val.reshape(-1,1),samplemean,samplestd),axis=1)
    filename3 = "LocalReAsphericity_"+"diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+"_segment_"+str(segmentlen)+".txt"
    with open(filename3, 'wb+') as f3:
        np.savetxt(f3, final)
    end = time1.time()
    print(end-start)
MPI.Finalize()
