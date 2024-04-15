#!/usr/bin/python3
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
        return cluster.num_clusters, cluster.cluster_idx
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
for sample in range(1,samples):

    filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
    print(filename)
    traj=gsd.hoomd.open(name=filename, mode='rb')
    snap =traj[0]
    num_cluster ,cluster_id = cluster_props(snap,0)
    #print(cluster_id)

    data2 = np.empty((0,totalcell,4),dtype=float)
    st =time1.perf_counter()
    for frame in range(init,len(traj),1):
        snap =traj[frame]
        gyra_tensor,com,rg= cluster_props(snap,1)
        #print(rg)
        diag,eigv = LA.eig(gyra_tensor)
        ident = np.tile(np.identity(3),(len(diag),1)).reshape((len(diag),3,3))
        k = diag.repeat(3).reshape((len(gyra_tensor),3,3))
        #print(ident)
        #print(k)
        tmp_gyr=k*ident

        #print(gyra_tensor.shape,com.shape)
        bcom = box.wrap(com[1:])
        Rgx=gyra_tensor[1:,0,0]
        Rgy=gyra_tensor[1:,1,1]
        Rgz=gyra_tensor[1:,2,2]
        Rg2 =Rgx+Rgy+Rgz

        tmp_Rgx=tmp_gyr[1:,0,0]
        tmp_Rgy=tmp_gyr[1:,1,1]
        tmp_Rgz=tmp_gyr[1:,2,2]
        tmp_Rg2 =tmp_Rgx+tmp_Rgy+tmp_Rgz
        #print(Rg2,tmp_Rg2)

        #print(gyra_tensor[0])


        xi = ((slcx +bcom[:,0])/lcx).astype(int)
        xi=np.where(xi<0 ,xi+1, xi)
        xi=np.where(xi==cellnox,xi-1,xi)
        count=np.bincount(xi,minlength = len(bin_val))
        data1 = np.empty((0,4),dtype=float)
        for i in range(totalcell):
            ids=np.where(xi == i)[0]
            #print(ids)
            Rgxs = Rgx[ids[:]]
            Rgys = Rgy[ids[:]]
            Rgzs = Rgz[ids[:]]
            Rg2s = Rg2[ids[:]]
            # avgRg2x=np.nanmean(Rgxs,axis=0)
            # avgRg2y=np.nanmean(Rgys,axis=0)
            # avgRg2z=np.nanmean(Rgzs,axis=0)
            # avgRg2=np.nanmean(Rg2s,axis=0)
            #Rgperp = (3*avgRg2x - avgRg2)/(2*(avgRg2))
            #Rgpara = (3*(avgRg2z+avgRg2y)/2 - avgRg2)/(2*(avgRg2))
            Rgperp = (3*Rgxs - Rg2s)/(2*(Rg2s))
            Rgycnf = (3*(Rgys) - Rg2s)/(2*(Rg2s))
            Rgzcnf = (3*(Rgzs) - Rg2s)/(2*(Rg2s))
            data = np.asarray((bin_val[i],np.nanmean(Rgperp,axis=0),np.nanmean(Rgycnf,axis=0),np.nanmean(Rgzcnf,axis=0))).T
            data1 = np.append(data1,[data],axis=0)

        data2 = np.append(data2,[data1],axis=0)

    en =time1.perf_counter()
    print(en-st)
    data3 = np.append(data3,[np.nanmean(data2,axis=0)],axis=0)
    datavar = np.append(datavar,[np.nanstd(data2,axis=0)],axis=0)

print(datavar.shape,data3.shape)
datastd = np.nanstd(data3[:,:,1:4],axis=0)/np.sqrt(len(data3))
fin_data = np.concatenate((np.nanmean(data3[:,:,:],axis=0),datastd),axis=1)
print(fin_data.shape)
filename3 = "RgxxRgyyRgzz1_"+"diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+"_cutoff_"+str(cutoffx)+".txt"
with open(filename3, 'wb+') as f3:
    np.savetxt(f3, fin_data)
