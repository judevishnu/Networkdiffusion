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

#samples=int(sys.argv[1])
Ntot = sys.argv[1]
fraction=sys.argv[2]
densratio = sys.argv[3]
molefrac  = sys.argv[4]
segmentlen = int(sys.argv[5])
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
            clp.compute(system,cluster_ids)
            return clp.gyrations, clp.centers,clp.radii_of_gyration


def Rgfunc(points,box,cluster_ids):
    clp = freud.cluster.ClusterProperties()
    #print(box)
    #system = (box,points)
    system = freud.AABBQuery(box, points)

    clp.compute(system,cluster_ids)
    return clp.gyrations,clp.centers,clp.radii_of_gyration

############################################################################
init =1000
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
delV = cutoffx*Ly*Lz
def wrap(arr):
    print(arr.shape)
    return box.wrap(arr)
bin_val = np.empty(totalcell,dtype=float)
for x in range(totalcell):
    bin_val[x] = (x+0.5)*lcutoffx-Lxby2

print(len(bin_val),totalcell)

data3 = np.empty((0,totalcell,4),dtype=float)
datavar = np.empty((0,totalcell,4),dtype=float)


poly_len = int(Lz/sigma[0][0])
typeid =  np.copy(traj[0].particles.typeid)
N_remainz = np.count_nonzero(typeid == 1)
shape1 = int(N_remainz/poly_len)
sample_orient = np.empty((0,2),dtype=float)

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
#start = time1.time()
step=1
begini = time1.time()
for sample in range(1,samples):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
        

    filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
    print(filename)
    traj=gsd.hoomd.open(name=filename, mode='rb')
    snap =traj[0]
    num_cluster ,cluster_id,cluster_keys = cluster_props(snap,0)
    #print(cluster_keys)
    cluster_keys = np.asarray(cluster_keys[1:])
    store_shape = cluster_keys.shape
    typid = snap.particles.typeid == 1
    polypartlen = np.sum(typid)
    cluster_id = np.arange(int(polypartlen/segmentlen)).reshape(int(polypartlen/segmentlen),1)
    cluster_id=np.tile(cluster_id,(1,segmentlen))
    print(cluster_id)
    
    # selid = cluster_id[:,1:poly_len-1:2]

    data2 = np.empty((0,totalcell,3),dtype=float)
    st =time1.perf_counter()
    time_orient = np.empty((0,len(bin_val),3))
    
    filename2 = "density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".bin"

    data = np.load(filename2)
    print(data.shape)  
    
    elements_per_process = (len(traj)-init)//size
    start_index = rank*elements_per_process
    end_index = (rank+1)*elements_per_process
    if rank == size-1:
        end_index = start_index +len(traj) -(start_index+init)
        print("rank :",rank,"size-1 :",size-1,"start_index :",start_index,"end_index :",end_index,flush=True)

    data2 = np.zeros((int((end_index-start_index)/step),2),dtype=float)

    for frame in range(init+start_index,end_index+init,step):
        #print(frame,init-1+int((frame-init)/step))
        
        print("sample :",sample,"rank :",rank,"frame : ",frame,"int((frame-(start+start_index))/step) :",int((frame-(init+start_index))/step),"start_index :",start_index,"end_index :",end_index,flush=True)
        if int((frame-(init+start_index))/step) >=len(data2):
            break

        frameid = frame-1
        
        print(frame,frameid)
        box_x=data[frameid,:,0]
        print(box_x.shape)
        #exit()
        Len = int(len(box_x)/2)
        gel_dens=data[frameid,:,1]
        poly_dens=data[frameid,:,2]
        abs_diff=np.fabs(gel_dens[0:Len]-poly_dens[0:Len]) #absolute value difference between number 
                                                                                                #densities of gel and polymer
        abs_diff1=np.fabs(gel_dens[Len:]-poly_dens[Len:]) 
     
        min_val=np.min(abs_diff) # Minimum of differences means, the values of densities were close enough
        min_val1=np.min(abs_diff1) # Minimum of differences means, the values of densities were close enough
    
        index=np.where(abs_diff==min_val) ## Index of the min_val in the array, this corrsponds to the box_x value where densities 
        index1=np.where(abs_diff1==min_val1) ## Index of the min_val in the array, this corrsponds to the box_x value where densities 
                                             ##approximately intersect
        
        #print(index,index)
        Interface1 = box_x[index[0][0]]
        Interface2 = box_x[Len+index1[0][0]]
        print(frame,frameid,Interface1,Interface2)
        Interface1_end = box_x[index[0][0]]-cutoffx
        Interface2_end = box_x[Len+index[0][0]]+cutoffx
        
        binInterf1=((slcx +Interface1)/lcx).astype(int)
        binInterf2=((slcx +Interface2)/lcx).astype(int)
        
        binInterf1_end=((slcx +Interface1_end)/lcx).astype(int)
        binInterf2_end=((slcx +Interface2_end)/lcx).astype(int)

        snap =traj[frame]
        position = snap.particles.position
        image = snap.particles.image
        real_pos = box.unwrap(position,image)
        sel_real_pos = real_pos[cluster_keys].reshape(int(polypartlen/segmentlen),segmentlen,3)
        sel_pos = position[cluster_keys].reshape(int(polypartlen/segmentlen),segmentlen,3)
        
        #print(sel_pos.shape,cluster_id.shape)
        gyrseg,comseg,rgseg = Rgfunc(sel_pos.reshape(-1,3),box,cluster_id.flatten())
        #print(comseg.shape,rgseg.shape,gyrseg.shape)
        comseg = box.wrap(comseg)
        #gyrseg = gyrseg.reshape(int(gyrseg.shape[0]/numseg),numseg,3,3)

        comseg = comseg.reshape(int(comseg.shape[0]/numseg),numseg,3)

        
        comsegend0 = comseg[:,0,0].flatten()
        comsegend1 = comseg[:,numseg-1,0].flatten()
        comsegmid = comseg[:,mid,0].flatten()
        
        if numseg%2==0:
            #gyrsegmid = np.nanmean(np.asarray((gyrseg[:,mid,0,0].T,gyrseg[:,mid,1,1].T,gyrseg[:,mid,2,2].T)).T,axis=1).reshape((-1,3))
            #rg2segmid =  np.nanmean(rg2seg[:,mid],axis=1).flatten()
            comsegmid1 = comseg[:,mid[0],0].flatten()
            comsegmid2 = comseg[:,mid[1],0].flatten()
            
            xiend0 = ((slcx +comsegend0)/lcx).astype(int)
            xiend0=np.where(xiend0<0 ,xiend0+1, xiend0)
            xiend0=np.where(xiend0==cellnox,xiend0-1,xiend0)
        
            ximid1 = ((slcx +comsegmid1)/lcx).astype(int)
            ximid1=np.where(ximid1<0 ,ximid1+1, ximid1)
            ximid1=np.where(ximid1==cellnox,ximid1-1,ximid1)
            
            ximid2 = ((slcx +comsegmid2)/lcx).astype(int)
            ximid2=np.where(ximid2<0 ,ximid2+1, ximid2)
            ximid2=np.where(ximid2==cellnox,ximid2-1,ximid2)

            
            xiend1 = ((slcx +comsegend1)/lcx).astype(int)
            xiend1=np.where(xiend1<0 ,xiend1+1, xiend1)
            xiend1=np.where(xiend1==cellnox,xiend1-1,xiend1)
            
            numxend1=len(np.where(((xiend0==(binInterf1_end|binInterf1))|(xiend0==(binInterf2_end|binInterf2))))[0])
            numxend2=len(np.where(((xiend1==(binInterf1_end|binInterf1))|(xiend1==(binInterf2_end|binInterf2))))[0])
            numxmid1=len(np.where(((ximid1==(binInterf1_end|binInterf1))|(ximid1==(binInterf2_end|binInterf2))))[0])
            numxmid2=len(np.where(((ximid2==(binInterf1_end|binInterf1))|(ximid2==(binInterf2_end|binInterf2))))[0])
            numxmid = np.nanmean(np.asarray([numxmid1,numxmid2]))
            numxend =  np.nanmean(np.asarray([numxend1,numxend2]))
            print(numxend1,numxend2,numxmid1,numxmid2)
            
            


        
        else:
            print(comsegmid.shape)
            xiend0 = ((slcx +comsegend0)/lcx).astype(int)
            xiend0=np.where(xiend0<0 ,xiend0+1, xiend0)
            xiend0=np.where(xiend0==cellnox,xiend0-1,xiend0)
        
            ximid = ((slcx +comsegmid)/lcx).astype(int)
            ximid=np.where(ximid<0 ,ximid+1, ximid)
            ximid=np.where(ximid==cellnox,ximid-1,ximid)

            xiend1 = ((slcx +comsegend1)/lcx).astype(int)
            xiend1=np.where(xiend1<0 ,xiend1+1, xiend1)
            xiend1=np.where(xiend1==cellnox,xiend1-1,xiend1)
                
            numxend1=len(np.where(((xiend0==(binInterf1_end|binInterf1))|(xiend0==(binInterf2_end|binInterf2))))[0])
            numxend2=len(np.where(((xiend1==(binInterf1_end|binInterf1))|(xiend1==(binInterf2_end|binInterf2))))[0])
            numxmid=len(np.where(((ximid==(binInterf1_end|binInterf1))|(ximid==(binInterf2_end|binInterf2))))[0])
            numxend =  np.nanmean(np.asarray([numxend1,numxend2]))
            
            print(numxend1,numxend2,numxmid)

            
        
        data1 = np.asarray((numxend,numxmid)).T
        data2[int((frame-(init+start_index))/step)] = data1
    
    collect_data2 = comm.gather(data2,root=0)
    if rank==0:
        data_concate=np.concatenate(collect_data2,axis=0)
        sample_orient = np.append(sample_orient,data_concate,axis=0)
        print(sample_orient,sample_orient.shape)
comm.Barrier()
if rank==0:        
    filename3 = "SegProbability_"+"diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+"_segment_"+str(segmentlen)+".txt"
    with open(filename3, 'wb+') as f3:
        np.savetxt(f3, sample_orient)
end = time1.time()
print(end-start)
MPI.Finalize()
