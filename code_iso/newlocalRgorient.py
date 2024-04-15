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
    print("mid odd",mid)
else:
    mid = np.array([int(numseg/2),int(numseg/2)+1])
    print("mid even",mid)
#exit()
print(mid,numseg)
#start = time1.time()
for sample in range(1,samples):

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

    data2 = np.empty((0,totalcell,4),dtype=float)
    st =time1.perf_counter()
    time_orient = np.empty((0,len(bin_val),9))

    for frame in range(init,len(traj),1):
        snap =traj[frame]
        position = snap.particles.position
        image = snap.particles.image
        real_pos = box.unwrap(position,image)
        sel_real_pos = real_pos[cluster_keys].reshape(int(polypartlen/segmentlen),segmentlen,3)
        sel_pos = position[cluster_keys].reshape(int(polypartlen/segmentlen),segmentlen,3)
        print(sel_pos.shape,cluster_id.shape)
        gyrseg,comseg,rgseg = Rgfunc(sel_pos.reshape(-1,3),box,cluster_id.flatten())
        print(comseg.shape,rgseg.shape,gyrseg.shape)
        comseg = box.wrap(comseg)
        gyrseg = gyrseg.reshape(int(gyrseg.shape[0]/numseg),numseg,3,3)

        comseg = comseg.reshape(int(comseg.shape[0]/numseg),numseg,3)

        rg2seg = np.square(rgseg).reshape(int(rgseg.shape[0]/numseg),numseg)
        
        gyrsegend0 = np.asarray((gyrseg[:,0,0,0].T,gyrseg[:,0,1,1].T,gyrseg[:,0,2,2].T)).T.reshape((-1,3))
        gyrsegend1 = np.asarray((gyrseg[:,numseg-1,0,0].T,gyrseg[:,numseg-1,1,1].T,gyrseg[:,numseg-1,2,2].T)).T.reshape((-1,3))
        gyrsegmid = np.asarray((gyrseg[:,mid,0,0].T,gyrseg[:,mid,1,1].T,gyrseg[:,mid,2,2].T)).T.reshape((-1,3))
        #print(gyrsegmid.shape)
        #exit()
        rg2segend0 = rg2seg[:,0].flatten()
        rg2segend1 = rg2seg[:,numseg-1].flatten()
        rg2segmid =  rg2seg[:,mid].flatten()

        comsegend0 = comseg[:,0,0].flatten()
        comsegend1 = comseg[:,numseg-1,0].flatten()
        comsegmid = comseg[:,mid,0].flatten()
        
        if numseg%2==0:
            gyrsegmid2 =  np.asarray((gyrseg[:,mid[1],0,0].T,gyrseg[:,mid[1],1,1].T,gyrseg[:,mid[1],2,2].T)).T.reshape((-1,3))
            gyrsegmid1 = np.asarray((gyrseg[:,mid[0],0,0].T,gyrseg[:,mid[0],1,1].T,gyrseg[:,mid[0],2,2].T)).T.reshape((-1,3))
            rg2segmid1 =  rg2seg[:,mid[0]].flatten()
            rg2segmid2 =  rg2seg[:,mid[1]].flatten()
            comsegmid1 = comseg[:,mid[0],0].flatten()
            comsegmid2 = comseg[:,mid[1],0].flatten()

            print(gyrsegmid.shape,rg2segmid.shape,comsegmid1.shape,comsegmid2.shape)
            
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
        
            data1 = np.empty((0,9),dtype=float)

            for i in range(totalcell):
                idsend1=np.where(xiend1 == i)[0]
                idsmid1=np.where(ximid1 == i)[0]
                idsmid2=np.where(ximid2 == i)[0]
                idsend0=np.where(xiend0 == i)[0]
                #print(ids)
                Rgxsend0 = gyrsegend0[idsend0[:],0]
                Rgysend0 = gyrsegend0[idsend0[:],1]
                Rgzsend0 = gyrsegend0[idsend0[:],2]
                Rg2send0 = rg2segend0[idsend0[:]]
            
                Rgxsend1 = gyrsegend1[idsend1[:],0]
                Rgysend1 = gyrsegend1[idsend1[:],1]
                Rgzsend1 = gyrsegend1[idsend1[:],2]
                Rg2send1 = rg2segend1[idsend1[:]]
            
                Rgxsmid1 = gyrsegmid[idsmid1[:],0]
                Rgysmid1 = gyrsegmid[idsmid1[:],1]
                Rgzsmid1 = gyrsegmid[idsmid1[:],2]
                Rg2smid1 = rg2segmid[idsmid1[:]]
            
                Rgxsmid2 = gyrsegmid[idsmid2[:],0]
                Rgysmid2 = gyrsegmid[idsmid2[:],1]
                Rgzsmid2 = gyrsegmid[idsmid2[:],2]
                Rg2smid2 = rg2segmid[idsmid2[:]]

                
                
                Rgxsend0av = np.nanmean(Rgxsend0)
                Rgysend0av = np.nanmean(Rgysend0)
                Rgzsend0av = np.nanmean(Rgzsend0)
                Rg2send0av = np.nanmean(Rg2send0)

                Rgxsend1av = np.nanmean(Rgxsend1)
                Rgysend1av = np.nanmean(Rgysend1)
                Rgzsend1av = np.nanmean(Rgzsend1)
                Rg2send1av = np.nanmean(Rg2send1)

                Rgxsmid1av = np.nanmean(Rgxsmid1)
                Rgysmid1av = np.nanmean(Rgysmid1)
                Rgzsmid1av = np.nanmean(Rgzsmid1)
                Rg2smid1av = np.nanmean(Rg2smid1)
                
                Rgxsmid2av = np.nanmean(Rgxsmid2)
                Rgysmid2av = np.nanmean(Rgysmid2)
                Rgzsmid2av = np.nanmean(Rgzsmid2)
                Rg2smid2av = np.nanmean(Rg2smid2)

                Rgxsmidav = np.nanmean(np.asarray([Rgxsmid1av,Rgxsmid2av]))
                Rgysmidav = np.nanmean(np.asarray([Rgysmid1av,Rgysmid2av]))
                Rgzsmidav = np.nanmean(np.asarray([Rgzsmid1av,Rgzsmid2av]))
                Rg2smidav = np.nanmean(np.asarray([Rg2smid1av,Rg2smid2av]))

            
                delRgxend0 = (3*Rgxsend0av - Rg2send0av)/(2*Rg2send0av)
                delRgyend0 = (3*Rgysend0av - Rg2send0av)/(2*Rg2send0av)
                delRgzend0 = (3*Rgzsend0av - Rg2send0av)/(2*Rg2send0av)

                delRgxend1 = (3*Rgxsend1av - Rg2send1av)/(2*Rg2send1av)
                delRgyend1 = (3*Rgysend1av - Rg2send1av)/(2*Rg2send1av)
                delRgzend1 = (3*Rgzsend1av - Rg2send1av)/(2*Rg2send1av)

                delRgxmid = (3*Rgxsmidav -  Rg2smidav)/(2*Rg2smidav)
                delRgymid = (3*Rgysmidav - Rg2smidav)/(2*Rg2smidav)
                delRgzmid = (3*Rgzsmidav - Rg2smidav)/(2*Rg2smidav)


                data = np.asarray((delRgxend0.T,delRgyend0.T,delRgzend0.T,delRgxend1.T,delRgyend1.T,delRgzend1.T,delRgxmid.T,delRgymid.T,delRgzmid.T)).T
                data1 = np.append(data1,[data],axis=0)



        else:
        #exit()
            xiend0 = ((slcx +comsegend0)/lcx).astype(int)
            xiend0=np.where(xiend0<0 ,xiend0+1, xiend0)
            xiend0=np.where(xiend0==cellnox,xiend0-1,xiend0)
        
            ximid = ((slcx +comsegmid)/lcx).astype(int)
            ximid=np.where(ximid<0 ,ximid+1, ximid)
            ximid=np.where(ximid==cellnox,ximid-1,ximid)

            xiend1 = ((slcx +comsegend1)/lcx).astype(int)
            xiend1=np.where(xiend1<0 ,xiend1+1, xiend1)
            xiend1=np.where(xiend1==cellnox,xiend1-1,xiend1)
        
            data1 = np.empty((0,9),dtype=float)
            for i in range(totalcell):
                idsend1=np.where(xiend1 == i)[0]
                idsmid=np.where(ximid == i)[0]
                idsend0=np.where(xiend0 == i)[0]
            #print(ids)
                Rgxsend0 = gyrsegend0[idsend0[:],0]
                Rgysend0 = gyrsegend0[idsend0[:],1]
                Rgzsend0 = gyrsegend0[idsend0[:],2]
                Rg2send0 = rg2segend0[idsend0[:]]
            
                Rgxsend1 = gyrsegend1[idsend1[:],0]
                Rgysend1 = gyrsegend1[idsend1[:],1]
                Rgzsend1 = gyrsegend1[idsend1[:],2]
                Rg2send1 = rg2segend1[idsend1[:]]
            
                Rgxsmid = gyrsegmid[idsmid[:],0]
                Rgysmid = gyrsegmid[idsmid[:],1]
                Rgzsmid = gyrsegmid[idsmid[:],2]
                Rg2smid = rg2segmid[idsmid[:]]
            
                Rgxsend0av = np.nanmean(Rgxsend0)
                Rgysend0av = np.nanmean(Rgysend0)
                Rgzsend0av = np.nanmean(Rgzsend0)
                Rg2send0av = np.nanmean(Rg2send0)

                Rgxsend1av = np.nanmean(Rgxsend1)
                Rgysend1av = np.nanmean(Rgysend1)
                Rgzsend1av = np.nanmean(Rgzsend1)
                Rg2send1av = np.nanmean(Rg2send1)

                Rgxsmidav = np.nanmean(Rgxsmid)
                Rgysmidav = np.nanmean(Rgysmid)
                Rgzsmidav = np.nanmean(Rgzsmid)
                Rg2smidav = np.nanmean(Rg2smid)

            
                delRgxend0 = (3*Rgxsend0av - Rg2send0av)/(2*Rg2send0av)
                delRgyend0 = (3*Rgysend0av - Rg2send0av)/(2*Rg2send0av)
                delRgzend0 = (3*Rgzsend0av - Rg2send0av)/(2*Rg2send0av)

                delRgxend1 = (3*Rgxsend1av - Rg2send1av)/(2*Rg2send1av)
                delRgyend1 = (3*Rgysend1av - Rg2send1av)/(2*Rg2send1av)
                delRgzend1 = (3*Rgzsend1av - Rg2send1av)/(2*Rg2send1av)

                delRgxmid = (3*Rgxsmidav -  Rg2smidav)/(2*Rg2smidav)
                delRgymid = (3*Rgysmidav - Rg2smidav)/(2*Rg2smidav)
                delRgzmid = (3*Rgzsmidav - Rg2smidav)/(2*Rg2smidav)


                data = np.asarray((delRgxend0.T,delRgyend0.T,delRgzend0.T,delRgxend1.T,delRgyend1.T,delRgzend1.T,delRgxmid.T,delRgymid.T,delRgzmid.T)).T
                data1 = np.append(data1,[data],axis=0)

        


        time_orient = np.append(time_orient,[data1],axis=0)
        
    sample_orient=np.append(sample_orient,[np.nanmean(time_orient,axis=0)],axis=0)

samplemean = np.nanmean(sample_orient,axis=0)
samplestd = np.nanstd(sample_orient,axis=0)
end = time1.time()
print(end-start)
final  = np.asarray((bin_val.T,samplemean[:,0].T,samplemean[:,1].T,samplemean[:,2].T,samplemean[:,3].T,samplemean[:,4].T,samplemean[:,5].T,samplemean[:,6].T,samplemean[:,7].T,samplemean[:,8].T,samplestd[:,0].T,samplestd[:,1].T,samplestd[:,2],samplestd[:,3].T,samplestd[:,4].T,samplestd[:,5],samplestd[:,6].T,samplestd[:,7].T,samplestd[:,8])).T
filename3 = "LocalRg_"+"diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+"_segment_"+str(segmentlen)+".txt"
with open(filename3, 'wb+') as f3:
    np.savetxt(f3, final)
