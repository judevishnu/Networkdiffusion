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
segment_length = int(sys.argv[5])
segmentlen =segment_length
sample=1
kbT = 1.0
overlap = 2
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
    system = (box,points)
    clp.compute(system,cluster_ids)
    return clp.centers

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

cutoffx = 4*sigma[0][0]
lcutoffx =cutoffx
lengthx = int(Lxby2/lcutoffx)
lcx = Lxby2/lengthx


shiftx = int(lengthx)
cellnox= 2*lengthx
totalcell = cellnox
slcx = shiftx*lcx
delV = cutoffx*Ly*Lz



poly_len = int(Lz/sigma[0][0])
num_cluster ,cluster_id,cluster_keys = cluster_props(snap,0)
numgelpoint = len(cluster_keys[0])
cluster_keys = np.asarray(cluster_keys[1:])
store_shape = cluster_keys.shape
num_segments = (len(cluster_keys) - overlap) // (segment_length - overlap)
# Create a view with overlapping segments 
overlapping_segments = np.lib.stride_tricks.sliding_window_view(cluster_keys, (segment_length,),axis=1)
overlapping_segments = overlapping_segments.reshape(-1,segment_length)
newmask_segmentid = (overlapping_segments-numgelpoint)%poly_len
#print(newmask_segmentid[0,5])
mid_segmentindex= newmask_segmentid[:,segment_length//2]
#print("ref",mid_segmentindex)
if segment_length%2==0:
    mid_segmentindex = (newmask_segmentid[:,0]+newmask_segmentid[:,segment_length-1])//2+1

else:
    mid_segmentindex = (newmask_segmentid[:,segment_length-1]+newmask_segmentid[:,0])//2

print(mid_segmentindex)
#exit()
unique_midsegmentindex = np.unique(mid_segmentindex,axis=0)
print(len(unique_midsegmentindex))
Dbin = unique_midsegmentindex[1]- unique_midsegmentindex[0] 
#print(unique_midsegmentindex)
binval = np.zeros(len(unique_midsegmentindex),dtype=int)
binval =unique_midsegmentindex
binrange = np.arange(binval[0]-0.5,1+binval[len(binval)-1]+0.5,1)
#print(len(binval))
#exit()
#newbinrange =[[newmask_segmentid[k,0],newmask_segmentid[k,segment_length-1]] for  k in range(0,len(newmask_segmentid))]
#newbinrange = np.unique(np.asarray(newbinrange),axis=0)

#print(newbinrange.flatten())
#exit()
cluster_ids = np.arange(0,len(overlapping_segments)).repeat(segment_length).reshape(-1)

typeid =  np.copy(traj[0].particles.typeid)
N_remainz = np.count_nonzero(typeid == 1)
shape1 = int(N_remainz/poly_len)
sample_orient = np.empty((0,len(binval),4),dtype=float)

start = time1.time()
#start = time1.time()
step=1
begini = time1.time()
for sample in range(1,samples):
            

    filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
    print(filename)
    traj=gsd.hoomd.open(name=filename, mode='rb')
        
    st =time1.perf_counter()
    time_orient = np.empty((0,len(binval),3))
    
    filename2 = "density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".bin"

    data = np.load(filename2)
        
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    
    elements_per_process = (len(traj)-init)//size
    start_index = rank*elements_per_process
    end_index = (rank+1)*elements_per_process
    if rank == size-1:
        end_index = start_index +len(traj) -(start_index+init)
        print("rank :",rank,"size-1 :",size-1,"start_index :",start_index,"end_index :",end_index,flush=True)

    data2 = np.zeros((int((end_index-start_index)/step),len(binval),4),dtype=float)

    for frame in range(init+start_index,end_index+init,step):
        #print(frame,init-1+int((frame-init)/step))
        
        print("sample :",sample,"segmentlen :",segment_length,"rank :",rank,"frame : ",frame,"int((frame-(start+start_index))/step) :",int((frame-(init+start_index))/step),"start_index :",start_index,"end_index :",end_index,flush=True)
        if int((frame-(init+start_index))/step) >=len(data2):
            break

        frameid = frame-1
        
        print(frame,frameid)
        box_x=data[frameid,:,0]
        Len = len(box_x)//2
        gel_dens=data[frameid,:,1]
        poly_dens=data[frameid,:,2]
        abs_diff=np.fabs(gel_dens[0:Len]-poly_dens[0:Len]) #absolute value difference between number 
                                                                                                #densities of gel and polymer
        abs_diff1=np.fabs(gel_dens[Len:]-poly_dens[Len:]) 
        print(len(abs_diff1),len(abs_diff))
        
        min_val=np.min(abs_diff) # Minimum of differences means, the values of densities were close enough
        min_val1=np.min(abs_diff1) # Minimum of differences means, the values of densities were close enough
    
        index=np.where(abs_diff==min_val) ## Index of the min_val in the array, this corrsponds to the box_x value where densities 
        index1=np.where(abs_diff1==min_val1) ## Index of the min_val in the array, this corrsponds to the box_x value where densities 
        midpoint1=index[0][0]//2
        midpoint2=int((len(box_x)-(index1[0][0]+Len))/2)+(index1[0][0]+Len)
                                    ##approximately intersect
        
        midx1 = box_x[midpoint1]
        midx2 = box_x[midpoint2]

        Interface1 = box_x[index[0][0]]
        Interface2 = box_x[Len+index1[0][0]]
        print(frame,frameid,Interface1,Interface2,midpoint1,midpoint2)
        index = ((slcx +Interface1)/lcx).astype(int)
        index = np.where(index<0 ,index+1, index)
        index = np.where(index==cellnox,index-1,index)
        
        index1 = ((slcx +Interface2)/lcx).astype(int)
        index1 = np.where(index1<0 ,index1+1, index1)
        index1 = np.where(index1==cellnox,index1-1,index1)



               
        midpoint1 = ((slcx +midx1)/lcx).astype(int)
        midpoin1 = np.where(midpoint1<0 ,midpoint1+1, midpoint1)
        midpoint1 = np.where(midpoint1==cellnox,midpoint1-1,midpoint1)
        
        midpoint2 = ((slcx +midx2)/lcx).astype(int)
        midpoin2 = np.where(midpoint2<0 ,midpoint2+1, midpoint2)
        midpoint2 = np.where(midpoint2==cellnox,midpoint2-1,midpoint2)

        print(midpoint1,midpoint2)
        #midpoint2=Len//2 +Len
        
        #print(box_x[midpoint1],box_x[midpoint2])
        #exit()
        
        snap =traj[frame]
        position = snap.particles.position
        sel_pos = position[overlapping_segments]
        #print(sel_pos.shape)
        sel_pos = sel_pos.reshape(-1,3)
        comwrap = Rgfunc(sel_pos,box,cluster_ids)
        print(comwrap.shape,newmask_segmentid.shape,mid_segmentindex.shape,overlapping_segments.shape)
        print(mid_segmentindex)
        #exit()
        
        xi = ((slcx +comwrap[:,0])/lcx).astype(int)
        xi = np.where(xi<0 ,xi+1, xi)
        xi = np.where(xi==cellnox,xi-1,xi)
        
        #idinterior1=mid_segmentindex[np.where(((midpoint1-1<=xi)&(xi<=midpoint1+1)))[0]]
        #idinterior2=mid_segmentindex[np.where(((midpoint2+1>=xi)&(xi>=midpoint2-1)))[0]]        
        #idinterf1=mid_segmentindex[np.where(((index-1<=xi)&(xi<=index)))[0]]
        #idinterf2=mid_segmentindex[np.where(((index1+1>=xi)&(xi>=index1)))[0]]
                    
        idinterior1=mid_segmentindex[np.where(midpoint1==xi)[0]]
        idinterior2=mid_segmentindex[np.where(midpoint2==xi)[0]]
        idinterf1=mid_segmentindex[np.where(index==xi)[0]]
        idinterf2=mid_segmentindex[np.where(index1==xi)[0]]
    
        #idinterior1=overlapping_segments[np.where(midpoint1==xi)[0]]
        #idinterior2=overlapping_segments[np.where(midpoint2==xi)[0]]        
        #idinterf1=overlapping_segments[np.where(index==xi)[0]]
        #idinterf2=overlapping_segments[np.where(index1==xi)[0]]
        #print(idinterf1)
        
        
        #probmidinterf= np.zeros(len(unique_midsegmentindex),dtype=float)
        #probmidinterior = np.zeros(len(unique_midsegmentindex),dtype=float)
        
        #probmidinterf1= np.zeros(len(unique_midsegmentindex),dtype=float)
        #probmidinterior1 = np.zeros(len(unique_midsegmentindex),dtype=float)
        
       
        count_idinterf1=np.histogram(idinterf1, bins=binrange,density=True)
        count_idinterf2=np.histogram(idinterf2, bins=binrange,density=True)
        count_idinterior1=np.histogram(idinterior1, bins=binrange,density=True)
        count_idinterior2=np.histogram(idinterior2, bins=binrange,density=True)
        probmidinterf1=count_idinterf1[0]
        probmidinterf2=count_idinterf2[0]
        probmidinterior1=count_idinterior1[0]
        probmidinterior2=count_idinterior2[0]
        #print(count_idinterf1[1])
        
       
        #for v in range(len(binval)):
        #    x = binval[v]
            #print(x)
        #    IDIinterior1=np.where(idinterior1==x)[0]
        #    IDIinterior2=np.where(idinterior1==x)[0]
        #    IDIinterf1=np.where(idinterf1==x)[0]
        #    IDIinterf2=np.where(idinterf2==x)[0]
        #    probmidinterior[v]=(len(IDIinterior1))/(len(idinterior1))
        #    probmidinterf[v]=(len(IDIinterf1))/(len(idinterf1))
        #    probmidinterior1[v]=(len(IDIinterior2))/(len(idinterior2))
        #    probmidinterf1[v]=(len(IDIinterf2))/(len(idinterf2))
 
        data1 = np.asarray((probmidinterf1,probmidinterf2,probmidinterior1,probmidinterior2)).T
        data2[int((frame-(init+start_index))/step)] = data1
    
    collect_data2 = comm.gather(data2,root=0)
    if rank==0:
        data_concate=np.concatenate(collect_data2,axis=0)
        #data_concat1 = np.asarray((binval,data_concate[:,0],data_concate[:,1])).T
        sample_orient = np.append(sample_orient,[np.nanmean(data_concate,axis=0)],axis=0)
        print(sample_orient,sample_orient.shape)
comm.Barrier()
if rank==0:
    finalstd=np.nanstd(sample_orient,axis=0)
    finalmean=np.nanmean(sample_orient,axis=0)
    finaldata = np.asarray((binval,finalmean[:,0],finalmean[:,1],finalmean[:,2],finalmean[:,3],finalstd[:,0],finalstd[:,1],finalstd[:,2],finalstd[:,3])).T
    filename3 = "NsegProbabilitynew_"+"diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+"_segment_"+str(segmentlen)+".txt"
    with open(filename3, 'wb+') as f3:
        np.savetxt(f3, finaldata)
end = time1.time()
print(end-start)
MPI.Finalize()
