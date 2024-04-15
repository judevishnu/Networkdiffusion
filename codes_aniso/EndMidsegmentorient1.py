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
samples=3
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



def Rgfunc(points,box):
    
    cluster_id = np.arange(0,len(points)).repeat(segment_length,axis=0).reshape(-1,segment_length)
    print(cluster_id)
    print(cluster_id.shape)
    exit()
    system = (box,points.reshape(-1,3))
    clp = freud.cluster.ClusterProperties()
    clp.compute(system,cluster_id.reshape(-1))
 
    
    return clp.gyrations, clp.centers,(clp.radii_of_gyration)**2



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
bin_val = np.empty(totalcell,dtype=float)
for x in range(totalcell):
    bin_val[x] = (x+0.5)*lcutoffx-Lxby2




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
mid_segmentindex = newmask_segmentid[:,segment_length//2]
print(mid_segmentindex)
#exit()
unique_midsegmentindex = np.unique(mid_segmentindex,axis=0)
print(len(unique_midsegmentindex))
Dbin = unique_midsegmentindex[1]- unique_midsegmentindex[0] 
#print(unique_midsegmentindex)
binval = np.zeros(len(unique_midsegmentindex),dtype=int)
binval =unique_midsegmentindex
binrange = np.arange(binval[0]-0.5,1+binval[len(binval)-1]+0.5,1)
remainder = len(binval)%3
print("end0 rank:",binval[0])
print("end1 rank:",binval[len(binval)-1])
print("mid rank:",binval[len(binval)//2])
#exit()

#if remainder==0:
#    endgroup1 = binval[np.arange(0,len(binval)//3)]
#    midgroup = binval[np.arange(len(binval)//3,int(2*len(binval)//3))]
#    endgroup2 = binval[np.arange(int(2*len(binval)//3),len(binval))]
#else:
#    endgroup1 = binval[np.arange(0,len(binval)//3)]
#    midgroup = binval[np.arange(len(binval)//3,remainder+int(2*len(binval)//3))]
#    endgroup2 = binval[np.arange(remainder+int(2*len(binval)//3),len(binval))]
endgroup1 =np.asarray([binval[0]])
endgroup2 = np.asarray([binval[len(binval)-1]])
midgroup = np.asarray([binval[len(binval)//2]])
#exit()

#newbinrange =[[newmask_segmentid[k,0],newmask_segmentid[k,segment_length-1]] for  k in range(0,len(newmask_segmentid))]
#newbinrange = np.unique(np.asarray(newbinrange),axis=0)

#print(newbinrange.flatten())
#exit()
cluster_ids = np.arange(0,len(overlapping_segments)).repeat(segment_length).reshape(-1)

typeid =  np.copy(traj[0].particles.typeid)
N_remainz = np.count_nonzero(typeid == 1)
shape1 = int(N_remainz/poly_len)
sample_orient = np.empty((0,len(bin_val),12),dtype=float)

start = time1.time()
#start = time1.time()
step=1
begini = time1.time()
for sample in range(1,samples):
            

    filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
    print(filename)
    traj=gsd.hoomd.open(name=filename, mode='rb')
        
    st =time1.perf_counter()
    time_orient = np.empty((0,len(bin_val),3))
    
    #filename2 = "density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".bin"

    #data = np.load(filename2)
        
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    
    elements_per_process = (len(traj)-init)//size
    start_index = rank*elements_per_process
    end_index = (rank+1)*elements_per_process
    if rank == size-1:
        end_index = start_index +len(traj) -(start_index+init)
        print("rank :",rank,"size-1 :",size-1,"start_index :",start_index,"end_index :",end_index,flush=True)

    data2 = np.zeros((int((end_index-start_index)/step),len(bin_val),12),dtype=float)

    for frame in range(init+start_index,end_index+init,step):
        #print(frame,init-1+int((frame-init)/step))
        
        print("sample :",sample,"segmentlen :",segment_length,"rank :",rank,"frame : ",frame,"int((frame-(start+start_index))/step) :",int((frame-(init+start_index))/step),"start_index :",start_index,"end_index :",end_index,flush=True)
        if int((frame-(init+start_index))/step) >=len(data2):
            break

        frameid = frame-1
        
        print(frame,frameid)
        
        snap =traj[frame]
        position = snap.particles.position
        sel_pos = position[overlapping_segments]
        #print(sel_pos.shape)
        sel_pos = sel_pos.reshape(-1,segment_length,3)
        gyration_tensor,comwrap,rg2=Rgfunc(sel_pos,box)

        #print(comwrap.shape,newmask_segmentid.shape,mid_segmentindex.shape,overlapping_segments.shape)
        #print(mid_segmentindex)
        #exit()

        eigval,eigvec=LA.eigh(gyration_tensor)
        maxeigvec = eigvec[:,:,2]
        cosangle = np.sum(maxeigvec*np.asarray([1,0,0]),axis=1)
        cosangle=np.where(cosangle>1,1,cosangle)
        cosangle=np.where(cosangle<-1,-1,cosangle)
        cos2angle=cosangle**2

        
        xi = ((slcx +comwrap[:,0])/lcx).astype(int)
        xi = np.where(xi<0 ,xi+1, xi)
        xi = np.where(xi==cellnox,xi-1,xi)
        
        data1 = np.empty((0,12),dtype=float)

        for v in range(len(bin_val)):
            #x = bin_val[v]
            ids = np.where(xi==v)[0]
            Rg2x = gyration_tensor[ids[:],0,0]
            Rg2y = gyration_tensor[ids[:],1,1]
            Rg2z =gyration_tensor[ids[:],2,2]
            Rg2s =Rg2x+Rg2y+Rg2z
            COS2angle  = cos2angle[ids[:]]

            mid_ids=mid_segmentindex[ids]
            groupend1_sel = np.isin(mid_ids, endgroup1)
            groupend2_sel = np.isin(mid_ids, endgroup2)
            groupmid_sel =  np.isin(mid_ids, midgroup)
            #print(len(groupend1_sel),len(ids),len(gyration_tensor))
            #exit()
            
            Rg2xend1 = Rg2x[ groupend1_sel[:]]
            Rg2yend1 = Rg2y[ groupend1_sel[:]]
            Rg2zend1 = Rg2z[ groupend1_sel[:]]
            Rg2send1 = Rg2s[groupend1_sel[:]]
            
            print(len(groupend1_sel),len(Rg2xend1),len(ids),len(gyration_tensor))
            #exit()
            Rg2xend2 = Rg2x[ groupend2_sel[:]]
            Rg2yend2 = Rg2y[ groupend2_sel[:]]
            Rg2zend2 = Rg2z[ groupend2_sel[:]]
            Rg2send2 = Rg2s[groupend2_sel[:]]
            

            Rg2xmid = Rg2x[ groupmid_sel[:]]
            Rg2ymid = Rg2y[ groupmid_sel[:] ]
            Rg2zmid = Rg2z[ groupmid_sel[:] ]
            Rg2smid = Rg2s[groupmid_sel[:]]
            #print(len(Rg2xmid))
            #exit()
            Rg2xend1av = np.nanmean(Rg2xend1)
            Rg2yend1av =  np.nanmean(Rg2yend1)
            Rg2zend1av =  np.nanmean(Rg2zend1)
            Rg2end1av = np.nanmean(Rg2send1)
            
            Rg2xend2av = np.nanmean(Rg2xend2)
            Rg2yend2av =  np.nanmean(Rg2yend2)
            Rg2zend2av =  np.nanmean(Rg2zend2)
            Rg2end2av = np.nanmean(Rg2send2)
            
            Rg2xmidav = np.nanmean(Rg2xmid)
            Rg2ymidav =  np.nanmean(Rg2ymid)
            Rg2zmidav =  np.nanmean(Rg2zmid)
            Rg2midav = np.nanmean(Rg2smid)

            delRgxend1 = (3*Rg2xend1av - Rg2end1av)/(2*Rg2end1av)
            delRgyend1 = (3*Rg2yend1av - Rg2end1av)/(2*Rg2end1av)
            delRgzend1 = (3*Rg2zend1av - Rg2end1av)/(2*Rg2end1av)

            delRgxend2 = (3*Rg2xend2av - Rg2end2av)/(2*Rg2end2av)
            delRgyend2 = (3*Rg2yend2av - Rg2end2av)/(2*Rg2end2av)
            delRgzend2 = (3*Rg2zend2av - Rg2end2av)/(2*Rg2end2av)
            
            delRgxmid = (3*Rg2xmidav - Rg2midav)/(2*Rg2midav)
            delRgymid = (3*Rg2ymidav - Rg2midav)/(2*Rg2midav)
            delRgzmid = (3*Rg2zmidav - Rg2midav)/(2*Rg2midav)
            
            
            selcos2mid = COS2angle[ groupmid_sel[:]]
            selcos2end1 = COS2angle[ groupend1_sel[:]]
            selcos2end2 = COS2angle[ groupend2_sel[:]]
            selcos2end1av = np.nanmean(selcos2end1) 
            selcos2end2av = np.nanmean(selcos2end2) 
            selcos2midav = np.nanmean(selcos2mid) 
            delcosangend1 = (3/2)*selcos2end1av - 0.5
            delcosangmid = (3/2)*selcos2midav - 0.5
            delcosangend2 = (3/2)*selcos2end2av - 0.5
            
            data = np.asarray((0.5*(delRgxend1+delRgxend2),0.5*(delRgyend1+delRgyend2),0.5*(delRgzend1+delRgzend2),delRgxmid,delRgymid,delRgzmid,Rg2end1av,Rg2midav,Rg2end2av,delcosangend1,delcosangmid,delcosangend2)).T
            data1 = np.append(data1,[data],axis=0)


                        
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
    print(finalmean.shape,finalstd.shape)
    #finaldata = np.asarray((binval,finalmean[:,0],finalmean[:,1],finalmean[:,2],finalmean[:,3],finalstd[:,0],finalstd[:,1],finalstd[:,2],finalstd[:,3])).T
    finaldata = np.concatenate((bin_val.reshape(-1,1),np.concatenate((finalmean,finalstd),axis=1)),axis=1)
    filename3 = "EndMiddleOrientseg1_"+"diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+"_segment_"+str(segmentlen)+".txt"
    with open(filename3, 'wb+') as f3:
        np.savetxt(f3, finaldata)
end = time1.time()
print(end-start)
MPI.Finalize()
