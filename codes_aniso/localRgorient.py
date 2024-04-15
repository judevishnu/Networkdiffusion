import hoomd
import hoomd.md
import gsd
import gsd.hoomd
import time  as time1
import numpy as np
import random
import math
import freud
import sys
from numpy import  linalg as LA

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

############################################################################
init = 1000
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

cutoffx = 1*sigma[0][0]
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

data3 = np.empty((0,totalcell,4),dtype=float)
datavar = np.empty((0,totalcell,4),dtype=float)


poly_len = int(Lz/sigma[0][0])
typeid =  np.copy(traj[0].particles.typeid)
N_remainz = np.count_nonzero(typeid == 1)
shape1 = int(N_remainz/poly_len)
sample_orient = np.empty((0,len(bin_val),9))

start = time1.time()
numseg = int(poly_len/segmentlen)
if numseg%2!=0:
    mid  = int(numseg/2)+1
else:
    mid = int(numseg/2)
for sample in range(1,samples):

    filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
    print(filename)
    traj=gsd.hoomd.open(name=filename, mode='rb')
    snap =traj[0]
    num_cluster ,cluster_id,cluster_keys = cluster_props(snap,0)
    #print(cluster_keys)
    cluster_keys = np.asarray(cluster_keys[1:])
    store_shape = cluster_keys.shape

    # selid = cluster_id[:,1:poly_len-1:2]

    data2 = np.empty((0,totalcell,4),dtype=float)
    st =time1.perf_counter()
    time_orient = np.empty((0,len(bin_val),9))

    for frame in range(init,len(traj),1):
        snap =traj[frame]
        position = snap.particles.position
        image = snap.particles.image
        real_pos = box.unwrap(position,image)
        sel_real_pos = real_pos[cluster_keys]
        sel_pos = position[cluster_keys]
        #print(position.shape,sel_pos.shape,sel_real_pos.shape)

        comseg = sel_pos[:,1:poly_len-1:segmentlen]
        #segment = sel_real_pos[:,0:poly_len-2:segment] 
        segment = sel_real_pos.reshape(-1,numseg,segmentlen,3)
        segment_com = np.nanmean(segment,axis=2)
        SHAPE=segment_com.shape
        #print(np.apply_along_axis(wrap,0,segment_com))

        segment_com1 = segment_com.repeat(segmentlen,axis=1).reshape(-1,numseg,segmentlen,3)
        segment_com =box.wrap(segment_com.reshape(-1,3)).reshape(SHAPE)
        print(segment_com.shape)
        #exit()

        segment_rgvec2 = np.mean((segment - segment_com1)**2,axis=2)
        #print(segment_com)
        print(segment.shape,segment_com.shape,segment_com1.shape,segment_rgvec2.shape)
        #exit()
        #segment_mag = np.sqrt(np.sum(np.square(segment),axis=2))
        #segment_e = segment/segment_mag.repeat(3).reshape(segment_mag.shape[0],segment_mag.shape[1],3)
        xi = ((slcx +segment_com[:,:,0])/lcx).astype(int)
        xi=np.where(xi<0 ,xi+1, xi)
        xi=np.where(xi==cellnox,xi-1,xi)
        print(xi.shape)
        xim = np.asarray((xi[:,0].T,xi[:,mid].T,xi[:,numseg-1].T)).T.reshape(-1,3,1)
        segment_rgvec2m = np.asarray((segment_rgvec2[:,0,:].T,segment_rgvec2[:,mid].T,segment_rgvec2[:,numseg-1].T)).T.reshape(-1,3,3)
        segment_rg2 = np.sum(np.asarray((segment_rgvec2[:,0,:].T,segment_rgvec2[:,mid].T,segment_rgvec2[:,numseg-1].T)).T.reshape(-1,3,3),axis=2)
        segment_rg2rep = segment_rg2#.repeat(3,axis=1).reshape(-1,3,3)
        Rg2alpha = segment_rgvec2m
        Rg2 = segment_rg2rep
        
    
        xi_end0_flat = xim[:,0,:].flatten()
        xi_end1_flat = xim[:,2,:].flatten()
        xi_mid_flat = xim[:,1,:].flatten()
        #print(xi_end0_flat.shape,len(bin_val))
        #print(xi_end0_flat[xi_end0_flat>len(bin_val)])
        #exit()
        Rg2x_end0_flat = Rg2alpha[:,0,0].flatten()
        Rg2y_end0_flat = Rg2alpha[:,0,1].flatten()
        Rg2z_end0_flat = Rg2alpha[:,0,2].flatten()
        
        Rg2x_end1_flat = Rg2alpha[:,2,0].flatten()
        Rg2y_end1_flat = Rg2alpha[:,2,1].flatten()
        Rg2z_end1_flat = Rg2alpha[:,2,2].flatten()
        
        Rg2x_mid_flat = Rg2alpha[:,1,0].flatten()
        Rg2y_mid_flat = Rg2alpha[:,1,1].flatten()
        Rg2z_mid_flat = Rg2alpha[:,1,2].flatten()
        #print(Rg2)
        #exit()
        Rg2_end0_flat = Rg2[:,0]
        Rg2_end1_flat = Rg2[:,1]
        Rg2_mid_flat = Rg2[:,2]

        countend0=np.zeros(len(bin_val))
        countmid=np.zeros(len(bin_val))
        countend1=np.zeros(len(bin_val))
        
        DelRg2xend0=np.zeros(len(bin_val))
        DelRg2xmid=np.zeros(len(bin_val))
        DelRg2xend1=np.zeros(len(bin_val))
        
        DelRg2yend0=np.zeros(len(bin_val))
        DelRg2ymid=np.zeros(len(bin_val))
        DelRg2yend1=np.zeros(len(bin_val))
        
        DelRg2zend0=np.zeros(len(bin_val))
        DelRg2zmid=np.zeros(len(bin_val))
        DelRg2zend1=np.zeros(len(bin_val))

        
        Rg2xend0=np.zeros(len(bin_val))
        Rg2xmid=np.zeros(len(bin_val))
        Rg2xend1=np.zeros(len(bin_val))
        
        Rg2yend0=np.zeros(len(bin_val))
        Rg2ymid=np.zeros(len(bin_val))
        Rg2yend1=np.zeros(len(bin_val))
        
        Rg2zend0=np.zeros(len(bin_val))
        Rg2zmid=np.zeros(len(bin_val))
        Rg2zend1=np.zeros(len(bin_val))
        
        Rg2end0=np.zeros(len(bin_val))
        Rg2mid=np.zeros(len(bin_val))
        Rg2end1=np.zeros(len(bin_val))

        
        np.add.at(countend0,xi_end0_flat,1)
        np.add.at(countmid,xi_mid_flat,1)
        np.add.at(countend1,xi_end1_flat,1)
        
        np.add.at(Rg2xend0,xi_end0_flat,Rg2x_end0_flat)
        np.add.at(Rg2xend1,xi_end1_flat,Rg2x_end1_flat)
        np.add.at(Rg2xmid,xi_mid_flat,Rg2x_mid_flat)
        
        np.add.at(Rg2yend0,xi_end0_flat,Rg2y_end0_flat)
        np.add.at(Rg2yend1,xi_end1_flat,Rg2y_end1_flat)
        np.add.at(Rg2ymid,xi_mid_flat,Rg2y_mid_flat)
        
        np.add.at(Rg2zend0,xi_end0_flat,Rg2z_end0_flat)
        np.add.at(Rg2zend1,xi_end1_flat,Rg2z_end1_flat)
        np.add.at(Rg2zmid,xi_mid_flat,Rg2z_mid_flat)
        
        np.add.at(Rg2end0,xi_end0_flat,Rg2_end0_flat)
        np.add.at(Rg2end1,xi_end1_flat,Rg2_end1_flat)
        np.add.at(Rg2mid,xi_mid_flat,Rg2_mid_flat)


        Rg2xend0 = Rg2xend0/countend0
        Rg2yend0 = Rg2yend0/countend0
        Rg2zend0 = Rg2zend0/countend0
        
        Rg2xend1 = Rg2xend1/countend1
        Rg2yend1 = Rg2yend1/countend1
        Rg2zend1 = Rg2zend1/countend1
        
        Rg2xmid = Rg2xmid/countmid
        Rg2ymid = Rg2ymid/countmid
        Rg2zmid = Rg2zmid/countmid
        
        Rg2mid = Rg2mid/countmid
        Rg2end0 = Rg2end0/countend0
        Rg2end1 = Rg2end1/countend1

        DelRg2yend0=((3/2)*Rg2yend0 - 0.5*Rg2end0)/Rg2end0
        DelRg2ymid=((3/2)*Rg2ymid - 0.5*Rg2mid)/Rg2mid
        DelRg2yend1=((3/2)*Rg2yend1 - 0.5*Rg2end1)/Rg2end1

        DelRg2xend0=((3/2)*Rg2xend0 - 0.5*Rg2end0)/Rg2end0
        DelRg2xmid=((3/2)*Rg2xmid - 0.5*Rg2mid)/Rg2mid
        DelRg2xend1=((3/2)*Rg2xend1 - 0.5*Rg2end1)/Rg2end1
        
        DelRg2zend0=((3/2)*Rg2zend0 - 0.5*Rg2end0)/Rg2end0
        DelRg2zmid=((3/2)*Rg2zmid - 0.5*Rg2mid)/Rg2mid
        DelRg2zend1=((3/2)*Rg2zend1 - 0.5*Rg2end1)/Rg2end1


        #a=np.where(Av_P2x <0)
        #print(Av_P2x)
        #print(Av_P2y)
        #print(Av_P2z)
        val=np.array((DelRg2xend0.T,DelRg2yend0.T,DelRg2zend0.T,DelRg2xend1.T,DelRg2yend1.T,DelRg2zend1.T,DelRg2xmid.T,DelRg2ymid.T,DelRg2zmid.T)).T
        time_orient = np.append(time_orient,[val],axis=0)
        
    sample_orient=np.append(sample_orient,[np.nanmean(time_orient,axis=0)],axis=0)

samplemean = np.nanmean(sample_orient,axis=0)
samplestd = np.nanstd(sample_orient,axis=0)
end = time1.time()
print(end-start)
final  = np.asarray((bin_val.T,samplemean[:,0].T,samplemean[:,1].T,samplemean[:,2].T,samplemean[:,3].T,samplemean[:,4].T,samplemean[:,5].T,samplemean[:,6].T,samplemean[:,7].T,samplemean[:,8].T,samplestd[:,0].T,samplestd[:,1].T,samplestd[:,2],samplestd[:,3].T,samplestd[:,4].T,samplestd[:,5],samplestd[:,6].T,samplestd[:,7].T,samplestd[:,8])).T
filename3 = "LocalRg_"+"diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+"_segment_"+str(segmentlen)+".txt"
with open(filename3, 'wb+') as f3:
    np.savetxt(f3, final)
