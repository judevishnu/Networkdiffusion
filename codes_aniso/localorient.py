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

cutoffx = 4*sigma[0][0]
lcutoffx =cutoffx
lengthx = int(Lxby2/lcutoffx)
lcx = Lxby2/lengthx


shiftx = int(lengthx)
cellnox= 2*lengthx
totalcell = cellnox
slcx = shiftx*lcx

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
sample_orient = np.empty((0,len(bin_val),3))

start = time1.time()
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
    time_orient = np.empty((0,len(bin_val),3))

    for frame in range(init,len(traj),1):
        snap =traj[frame]
        position = snap.particles.position
        image = snap.particles.image
        real_pos = box.unwrap(position,image)
        sel_real_pos = real_pos[cluster_keys]
        sel_pos = position[cluster_keys]
        #print(position.shape,sel_pos.shape,sel_real_pos.shape)

        comseg = sel_pos[:,1:poly_len-1:2]
        segment = sel_real_pos[:,0:poly_len-2:2] - sel_real_pos[:,2:poly_len:2]
        segment_mag = np.sqrt(np.sum(np.square(segment),axis=2))
        segment_e = segment/segment_mag.repeat(3).reshape(segment_mag.shape[0],segment_mag.shape[1],3)
        xi = ((slcx +comseg[:,:,0])/lcx).astype(int)
        xi=np.where(xi<0 ,xi+1, xi)
        xi=np.where(xi==cellnox,xi-1,xi)
        #count=np.bincount(xi[:,:],minlength = len(bin_val))
        #count =np.apply_along_axis(lambda x: np.bincount(x, minlength=len(bin_val)), axis=1, arr=xi)
        #count_flat = np.zeros(len(bin_val))
        Av_P2x = np.zeros(len(bin_val))
        Av_P2y = np.zeros(len(bin_val))
        Av_P2z = np.zeros(len(bin_val))
        
        #mag = np.sqrt(np.sum(segment_e**2,axis=2))
        #print(segment_mag)
        #exit()
        dotx = np.sum(segment_e*np.array([1,0,0]),axis=2)
        #dotx= np.where(dotx<-1,-1,dotx)
        #dotx= np.where(dotx>1,1,dotx)
        doty = np.sum(segment_e*np.array([0,1,0]),axis=2)
        #doty= np.where(doty<-1,-1,doty)
        
        #doty= np.where(doty>1,1,doty)
        
        
        dotz = np.sum(segment_e*np.array([0,0,1]),axis=2)
        #dotz= np.where(dotz<-1,-1,dotz)
        #dotz= np.where(dotz>1,1,dotz)
    
        cosinx2=np.square(dotx)
        cosiny2=np.square(doty)
        cosinz2=np.square(dotz)
        #print(cosinx2)
        #exit()
        P2x = (3/2)*cosinx2 - 0.5
        P2y = (3/2)*cosiny2 - 0.5
        P2z = (3/2)*cosinz2 - 0.5
        #print(xi)

    
        xi_flat = xi.flatten()
        P2x_flat = P2x.flatten()
        P2y_flat = P2y.flatten()
        P2z_flat = P2z.flatten()
        countP2=np.zeros(len(bin_val))
        
        np.add.at(countP2,xi_flat,1)
        np.add.at(Av_P2x,xi_flat,P2x_flat)
        np.add.at(Av_P2y,xi_flat,P2y_flat)
        np.add.at(Av_P2z,xi_flat,P2z_flat)
        Av_P2x =Av_P2x/countP2
        Av_P2y =Av_P2y/countP2
        Av_P2z =Av_P2z/countP2

        #a=np.where(Av_P2x <0)
        #print(Av_P2x)
        #print(Av_P2y)
        #print(Av_P2z)
        val=np.array((Av_P2x.T,Av_P2y.T,Av_P2z.T)).T
        time_orient = np.append(time_orient,[val],axis=0)
        
    sample_orient=np.append(sample_orient,[np.nanmean(time_orient,axis=0)],axis=0)

samplemean = np.nanmean(sample_orient,axis=0)
samplestd = np.nanstd(sample_orient,axis=0)
end = time1.time()
print(end-start)
final  = np.asarray((bin_val.T,samplemean[:,0].T,samplemean[:,1].T,samplemean[:,2].T,samplestd[:,0].T,samplestd[:,1].T,samplestd[:,2])).T
filename3 = "Localorient1_"+"diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+"_cutoff_"+str(cutoffx)+".txt"
with open(filename3, 'wb+') as f3:
    np.savetxt(f3, final)
