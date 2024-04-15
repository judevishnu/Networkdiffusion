#!/usr/bin/python3
import hoomd
import hoomd.md
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
############################################
########################################################################
#           Set parameters
#######################################################################
samples=11
dia=[]
dia.append(1.0)
dia.append(1.0)
sigma = [] # interaction parameter sigma 

sigma.append([(dia[0]+dia[0])/2.,(dia[0]+dia[1])/2.])
###############################################################################
dt = 0.001
period = 4e5
kbT=1.0
step=1
rmax=4
#rmax=2
#rmax=4.5
#timeint=(math.ceil((len(traj)-init)/step))
sum_Interface1=0.0
sum_Interface2=0.0
       

#data3 = numpy.empty((0,timeint,totalcell,2),dtype=float)
init=750
#timeint=(math.ceil((len(traj)-init)/step))
sum_Interface1=0.0
sum_Interface2=0.0
#print(timeint)
init1=750


ptlarray=(1155108,  1073100, 990888, 949884, 908880, 826872, 744660, 703656, 621648, 605328, 580644,  539640, 531480, 498636, 465792)
fracarray=("0.84991","0.74998", "0.64980", "0.59984" ,"0.54987" ,"0.44994", "0.34976" ,"0.29979", "0.19986", "0.17998", "0.14990" ,"0.09993" ,"0.08999" ,"0.04997", "0.00994")
densarray=("1.52410", "1.34490", "1.16525", "1.07565", "0.98605", "0.80685", "0.62720", "0.53760", "0.35840", "0.32274" ,"0.26880" ,"0.17920" ,"0.16137", "0.08960", "0.01783")
molearray=("0.60382", "0.57354", "0.53816", "0.51822", "0.49649", "0.44655", "0.38545", "0.34964", "0.26384", "0.24399", "0.21185", "0.15197", "0.13895", "0.08223", "0.01752")


ptlarray=(2426844, 2098608, 1934592, 1852584, 1770372, 1688364, 1442136, 1278120, 1196112)
fracarray=("2.39981", "1.99980", "1.79992", "1.69998", "1.59979", "1.49985", "1.19978", "0.99990", "0.89996") 
densarray=("4.30305", "3.58580", "3.22740", "3.04820", "2.86855",  "2.68935", "2.15130", "1.79290", "1.61370") 
molearray=("0.81143", "0.78194", "0.76345", "0.75298", "0.74151", "0.72895", "0.68267", "0.64195", "0.61740") 

#rmaxarr=("5.85","6.375","5.625","5.875","5.375","5.625","4.875","5.125","3.375","3.875","3.125","3.875","0.875","0.875","0.875")

data2 = numpy.empty((0,3),dtype=float)
data2Rg = numpy.empty((0,5),dtype=float)
data2num = numpy.empty((0,3),dtype=float)
lentraj=int((32e7)/period)

begining=time1.perf_counter()

def Interface_calc(frame):
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
            

for Ntot, fraction,densratio,molefrac in zip(ptlarray, fracarray,densarray,molearray):
#for Ntot, fraction,densratio,molefrac,rmax in zip(ptlarray, fracarray,densarray,molearray,rmaxarr):
    print(Ntot,fraction)
    sum_Interface1=0.0
    sum_Interface2=0.0
    stsample=1
    #if  Ntot==429912:
     #   stsample=4
     #   print(stsample)
    #rmax=float(rmax)
    
    
    data_sample=numpy.empty((0,2),dtype=float)
    data_sampleRg=numpy.empty((0,3),dtype=float)
    data_samplenum=numpy.empty((0,2),dtype=float)
    for sample in range(1,samples):
        filename = "trajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
        traj=gsd.hoomd.open(name=filename, mode='rb')
        Lx=traj[0].configuration.box[0]
        Ly=traj[0].configuration.box[1]
        Lz=traj[0].configuration.box[2]
        box = freud.box.Box(Lx = 10*Lx, Ly= Ly, Lz =Lz)
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

        #data=numpy.empty((0),dtype=float)
        timearray=numpy.empty((0),dtype=float)
        data_time  = numpy.empty((0,2),dtype=float)
        data_timeRg  = numpy.empty((0,3),dtype=float)
        data_timenum  = numpy.empty((0,2),dtype=float)
        for frame in range(init,len(traj),step):
            sum_Interface1,sum_Interface2=Interface_calc(frame)
            
            print("Interfaces",sum_Interface1,sum_Interface2)
            leftregion = sum_Interface1-2.5*sigma[0][0]
            rightregion = sum_Interface2+2.5*sigma[0][0]
            print("Regions",leftregion,rightregion,-Lxby2,Lxby2)

            pos_gel=traj[frame].particles.position[:N_constraint]
            pos_poly=traj[frame].particles.position[N_constraint:]
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
                truthtable3=((cluster_pointsx<sum_Interface2-60)&(cluster_pointsx>sum_Interface1+60))
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
            append1=numpy.array((float(molefrac),MeanRg,MeanRgCl)).T
            append2=numpy.array((float(molefrac),numclust)).T
            data_timeRg = numpy.append(data_timeRg,[append1],axis=0)
            data_timenum = numpy.append(data_timenum,[append2],axis=0)
            truthval=trutharray.any()
            if truthval==True:
                append=numpy.array((float(molefrac),trutharray.sum())).T
                data_time=numpy.append(data_time,[append],axis=0)
            else:
                append=numpy.array((float(molefrac),0)).T
                data_time=numpy.append(data_time,[append],axis=0)
        #print(data_time)
        data_time_mean=numpy.nanmean(data_time,axis=0)
        data_time_meanRg=numpy.nanmean(data_timeRg,axis=0)
        data_time_meannum=numpy.nanmean(data_timenum,axis=0)
        #print(data_time_mean)
        data_sample= numpy.append(data_sample,[data_time_mean],axis=0)
        data_sampleRg= numpy.append(data_sampleRg,[data_time_meanRg],axis=0)
        data_samplenum= numpy.append(data_samplenum,[data_time_meannum],axis=0)
    data_sample_mean = numpy.nanmean(data_sample,axis=0)
    data_sample_std = numpy.nanstd(data_sample,axis=0)
    data_sample_meanRg = numpy.nanmean(data_sampleRg,axis=0)
    data_sample_meannum = numpy.nanmean(data_samplenum,axis=0)
    data_sample_stdRg = numpy.nanstd(data_sampleRg,axis=0)
    data_sample_stdnum = numpy.nanstd(data_samplenum,axis=0)

    print(data_sample_mean)
    print(data_sample_std.shape,data_sample_mean.shape)
    
    #data2 = numpy.append(data2,[data_sample_mean],axis=0)
    data2 = numpy.append(data2,[numpy.asarray((data_sample_mean[0],data_sample_mean[1],data_sample_std[1])).T],axis=0)
    data2Rg = numpy.append(data2Rg,[numpy.asarray((data_sample_meanRg[0],data_sample_meanRg[1],data_sample_meanRg[2],data_sample_stdRg[1],data_sample_stdRg[2])).T],axis=0)
    data2num = numpy.append(data2num,[numpy.asarray((data_sample_meannum[0],data_sample_meannum[1],data_sample_stdnum[1])).T],axis=0)
    end=time1.perf_counter()
print(abs(begining-end))

print(data2)
    
#filename3 = "Percolation_points_vs_concentration"+"_diamond_network_diffusive"+"_kT_"+str(kbT)+".txt"
filename3 = "Percolation8_points_vs_concentration"+"_diamond_network_diffusive"+"_rmax_"+str(rmax)+"_kT_"+str(kbT)+".txt"
filename4 = "ClusterRg8_vs_concentration"+"_diamond_network_diffusive"+"_rmax_"+str(rmax)+"_kT_"+str(kbT)+".txt"
filename5 = "AvgClusternumber8_vs_concentration"+"_diamond_network_diffusive"+"_rmax_"+str(rmax)+"_kT_"+str(kbT)+".txt"

with open(filename3, 'w+') as f3:
        numpy.savetxt(f3, data2)
 
with open(filename4, 'w+') as f4:
        numpy.savetxt(f4, data2Rg)
 
with open(filename5, 'w+') as f5:
        numpy.savetxt(f5, data2num)
 



