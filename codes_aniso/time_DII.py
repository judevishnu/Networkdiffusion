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
Ntot = sys.argv[1]
fraction = sys.argv[2]
densratio = sys.argv[3]
molefrac  = sys.argv[4]

period = 2e5
kbT=4.3
cutoff = sigma[0][0]*1e-08
rtol=sigma[0][0]*1e-05
sample=1
#filename = "trajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_kT_"+str(kbT)+".gsd"

filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
traj=gsd.hoomd.open(name=filename, mode='rb')
Lx=traj[0].configuration.box[0]
Ly=traj[0].configuration.box[1]
Lz=traj[0].configuration.box[2]
box = freud.box.Box(Lx = Lx, Ly= Ly, Lz =Lz)
Ny = int(Ly/sigma[0][0])
poly_len = int(Lz/sigma[0][0])
typeid =  numpy.copy(traj[0].particles.typeid)
N_remainz = numpy.count_nonzero(typeid == 1)
N_constraint = numpy.count_nonzero(typeid==0)
Nx_remainz = int(N_remainz/2/Ny/poly_len)
shape = int(poly_len*3)
shape1 = int(N_remainz/poly_len)

dimension3=3
step=1
#timeint = 1+int((len(traj))/step)
init =1
timeint=(math.ceil((len(traj)-init)/step))
sum_Interface1=0.0
sum_Interface2=0.0
print(timeint,len(traj))
init1=750
data2 = numpy.empty((0,timeint,8),dtype=float)
print(data2.shape)
#exit()
begining = time1.perf_counter()
for frame in range(init1,len(traj),step):
    traj=gsd.hoomd.open(name=filename, mode='rb')
    
    
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
    print(Interface1,Interface2)
    sum_Interface2 =sum_Interface2+Interface2
    sum_Interface1 =sum_Interface1+Interface1
    max_densgel =  numpy.max(gel_dens)
    max_denspoly = numpy.max(poly_dens)  

avtime=(math.ceil((len(traj)-init1)/step))
sum_Interface2=sum_Interface2/avtime
sum_Interface1=sum_Interface1/avtime
print("Interfaces",sum_Interface1,sum_Interface2)
#exit()
step=1
r0=1.5
Lmax=0
Smax =r0*max_denspoly
if poly_len%2==0:
    for i in range(int(poly_len/2)):
        Lmax = Lmax+(i-0.5)*r0
else:
    if poly_len%2!=0:
        for i in range(int((poly_len-1)/2)):
            Lmax = Lmax+(i)*r0

Inorm = Lmax*Smax
#Inorm=max_denspoly

for sample in range(1,samples):
    #filename = "trajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_kT_"+str(kbT)+".gsd"
    
    filename = "trajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
    print(filename)
    traj=gsd.hoomd.open(name=filename, mode='rb')
    #filename2 = "density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_kT_"+str(kbT)+".bin"
    
    filename2 = "density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".bin"
    data1 = numpy.empty((0,8),dtype=float) 
    #idt =0
    for frame in range(init,len(traj),step):
        pos_gel=traj[frame].particles.position[:N_constraint]
        pos_poly=traj[frame].particles.position[N_constraint:]
        real_pos_poly = pos_poly[:]+numpy.array([Lx,Ly,Lz])*traj[frame].particles.image[N_constraint:]

        poly_position =real_pos_poly.reshape(-1,shape).reshape(shape1,poly_len,dimension3)  #Reshape position array along lenght of the polymers
        
        #print(Ny*2*Nx_remainz,len(poly_position))

        
        points2 = real_pos_poly[:,0]
        #data = numpy.load(filename2)
        #box_x=data[int(frame/step),:,0]
        #gel_dens=data[int(frame/step),:,3]
        #poly_dens=data[int(frame/step),:,4]
        #abs_diff=numpy.fabs(gel_dens-poly_dens) #absolute value difference between number densities of gel and polymer
        
        #min_val=numpy.min(abs_diff) # Minimum of differences means, the values of densities were close enough
        
        #index=numpy.where(abs_diff==min_val) ## Index of the min_val in the array, this corrsponds to the box_x value where densities 
                                             ##approximately intersect
        
        
        #if frame==0:
        #Interface1=-box_x[index[0][0]]
        #Interface2=box_x[index[0][0]]
        #Interface2 = abs(Interface2)
        #print(Interface1,Interface2)
        #"""
        data = numpy.load(filename2)
        box_x=data[math.ceil((frame-init)/step),:,0]
        #box_x=data[int(frame/step),:,0]

        Len = int(len(box_x)/2)
        gel_dens=data[math.ceil((frame-init)/step),:,1]
        #gel_dens=data[int(frame/step),:,3]
        poly_dens=data[math.ceil((frame-init)/step),:,2]
        #poly_dens=data[int(frame/step),:,4]

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
        #print(Interface1, Interface2) 
        #"""
        diff1=Interface1-points2 
        diff2=Interface2-points2 
        
    
        ### Finding polymer indices of polymers which are crossing the interface 1 ###
        #indices1 = numpy.where(numpy.fabs(diff1) <= cutoff )[0]
        indices1 = numpy.where(numpy.fabs(diff1) <= cutoff +rtol*numpy.fabs(Interface1))[0]
        #indices1 = numpy.where(numpy.fabs(diff1) <= cutoff )
        #print(indices1)
        #exit()
        indices1 = numpy.unique(indices1) 
        #print(diff1[indices1[:]])
        remainder1 = indices1%(poly_len)
        poly_indices1=((indices1-remainder1)/poly_len).astype(int)
        poly_indices1=numpy.unique(poly_indices1) 
        print(len(poly_indices1))
        poly_position1=numpy.take(poly_position,poly_indices1,axis=0)  ## position of beads of the polymers near interface 1
        poly_position1x = poly_position1[:,:,0]
        
        diff1=poly_position1x-Interface1
        #print(difference1) 
        d1_R=numpy.empty((0,1),dtype=float)
        d1_L = numpy.empty((0,1),dtype=float)
        d1 = numpy.empty((0),dtype=float)
        
        

        for i in range(len(diff1)):
            temp=diff1[i,numpy.where(diff1[i,:]>=0)]
            #print(temp)
            
            #print(len(temp[0,:]))
            d1_R=numpy.append(d1_R,[numpy.sum(temp,axis=1)],axis=0)
            
            temp=diff1[i,numpy.where(diff1[i,:]<0)]
            
            #print(len(temp[0,:]))
            d1_L=numpy.append(d1_L,[numpy.sum(numpy.fabs(temp),axis=1)],axis=0)
            #d1_L=numpy.append(d1_L,[abs(numpy.sum(temp,axis=1))],axis=0)
            
            #min_d1 = min(d1_R[i][0],d1_L[i][0])
            min_d1 = d1_R[i][0]
            d1 =  numpy.append(d1,[min_d1],axis=0)
        
     
        
        
       ########################################################

       
        ### Finding polymer indices of polymers which are crossing the interface 2 ###
        #indices2 = numpy.where(numpy.fabs(diff2) <= cutoff)[0]
        indices2 = numpy.where(numpy.fabs(diff2) <= cutoff +rtol*numpy.fabs(Interface2))[0]
        indices2 = numpy.unique(indices2)
        remainder2 = indices2%(poly_len)
        poly_indices2=((indices2-remainder2)/(poly_len)).astype(int)
        poly_indices2=numpy.unique(poly_indices2)
        #print(len(poly_indices2))
        poly_position2=numpy.take(poly_position,poly_indices2,axis=0) ## position of beads of the polymers near interface 2
        poly_position2x = poly_position2[:,:,0]
        diff2=poly_position2x-Interface2
        
        d2_R=numpy.empty((0,1),dtype=float)
        d2_L = numpy.empty((0,1),dtype=float)
        d2 = numpy.empty((0),dtype=float)
        
        for i in range(len(diff2)):
            temp=diff2[i,numpy.where(diff2[i,:]>0)]
            
            d2_R=numpy.append(d2_R,[numpy.sum(temp,axis=1)],axis=0)
            temp=diff2[i,numpy.where(diff2[i,:]<=0)]
            
            
            d2_L=numpy.append(d2_L,[numpy.sum(numpy.fabs(temp),axis=1)],axis=0)
            #d2_L=numpy.append(d2_L,[abs(numpy.sum(temp,axis=1))],axis=0)
            #min_d2 = min(d2_R[i][0],d2_L[i][0])
            min_d2 = d2_L[i][0]
            
            d2 =  numpy.append(d2,[min_d2],axis=0)
            #print(len(d2)) 
        
        #######################################################
        #  Length of polymer crossing interface
        ######################################################
        #print(d1[0])
        if len(d1) !=0: 
            average_L1=numpy.nanmean(d1,axis=0)
        else: 
            average_L1=0

        if len(d2)!=0:
            average_L2=numpy.nanmean(d2,axis=0)
        else:
            average_L2=0
        #print(average_L1,average_L2)
        #print(average_L1)
        #################################################
        Nc1 = len(poly_indices1)
        Nc2 = len(poly_indices2)
        A = Ly*Lz
        S1 = Nc1/A
        S2 = Nc2/A

        I1 = average_L1*S1  
        I2 = average_L2 *S2 
        I =(I1+I2)/2.
        time =dt*period*frame
        #idt=idt+1
        print(time,I1,I2) 
        data0=numpy.array((time,I1,I2,I,I1/Inorm,I2/Inorm,I/Inorm,(Nc1+Nc2)/2)).T
        data1= numpy.append(data1,[data0],axis=0)
        #print(data1)

    data2 = numpy.append(data2,[data1],axis=0)
    
    
data3=numpy.nanmean(data2,axis=0)
data4 = numpy.nanstd(data2,axis=0)
print(data4.shape)
data5 = numpy.concatenate((data3,data4[:,1:]),axis=1)

print(data5)
      

filename = "tDIInorm3"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"
with open(filename, 'w+') as f3:
    numpy.savetxt(f3, data5)
        

end = time1.perf_counter()
print(abs(end-begining))


