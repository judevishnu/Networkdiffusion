import numpy as np
import gsd.hoomd
import freud
#import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import sys
from mpi4py import MPI

sample = sys.argv[1]
Ntot = sys.argv[2]
fraction = sys.argv[3]
densratio = sys.argv[4]
molefrac  = sys.argv[5]


n_bin = 200
data_jump = 1
dt = 0.001
kbT=1.0
init=1
####################################################################################################################
## loading in csv dictionary
####################################################################################################################

descriptorDict = defaultdict(list)

#with open('description_file.csv', 'r') as data:

#    csv_reader = csv.reader(data)
#    rows = list(csv_reader)
#for row in rows:
#    descriptorDict[int(row[0])] = row[1]
#a = descriptorDict[Ntot]
#v=a.strip('][').split(', ')

####################################################################################################################
## parameters
####################################################################################################################

#volfrac = v[0]
#molefrac = v[1]
#denseratio = v[2]


####################################################################################################################
## functions
####################################################################################################################

def truncate(n):
    n = float(int(n * 100))
    n/= 100
    return n

def Interface(i):
    data=data_den[i-init]
    data_1 = data[0:int(len(data)/2)]
    data_2 = data[int(len(data)/2):]

    gel_den_1 = data_1[:,1]
    poly_den_1 = data_1[:,2]
    polydif_1 = np.argmin(np.absolute(np.subtract(gel_den_1,poly_den_1)))

    gel_den_2 = data_2[:,1]
    poly_den_2 = data_2[:,2]
    polydif_2 = np.argmin(np.absolute(np.subtract(gel_den_2,poly_den_2)))
    I1=data_1[polydif_1,0]
    I2=data_2[polydif_2,0]
    return I1,I2

def Interface_calc(frame,step):
        filename2 = "density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".bin"
        data = np.load(filename2)
        box_x=data[init-1+int((frame-init)/step),:,0]
        Len = int(len(box_x)/2)
        gel_dens=data[init-1+int((frame-init)/step),:,1]
        poly_dens=data[init-1+int((frame-init)/step),:,2]
        abs_diff=np.fabs(gel_dens[0:Len]-poly_dens[0:Len]) #absolute value difference between number
                                                                                                #densities of gel and polymer
        abs_diff1=np.fabs(gel_dens[Len:]-poly_dens[Len:])

        min_val=np.min(abs_diff) # Minimum of differences means, the values of densities were close enough
        min_val1=np.min(abs_diff1) # Minimum of differences means, the values of densities were close enough

        index=np.where(abs_diff==min_val) ## Index of the min_val in the array, this corrsponds to the box_x value where densities
        index1=np.where(abs_diff1==min_val1) ## Index of the min_val in the array, this corrsponds to the box_x value where densities
                                             ##approximately intersect
        Interface1 = box_x[index[0][0]]
        Interface2 = box_x[Len+index1[0][0]]
        return Interface1,Interface2



######################################################################################################################



standLen = 102

filename2 = "density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".bin"
data_den = np.load(filename2)
print (len(data_den))
max_denspoly=np.max(data_den[1000:,:,2])

filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
 


datadummy = gsd.hoomd.open(name=filename, mode='rb')
time_1=int((len(datadummy)-1)/data_jump)
DII_sample = np.empty((0,time_1,3),dtype=float)

bonds_array = datadummy[0].bonds.group
bonds_array=np.sort(bonds_array,axis=1)
bonds_array=np.unique(bonds_array,axis=0)
points  = datadummy[0].particles.position
Lx = datadummy[0].configuration.box[0]
Ly = datadummy[0].configuration.box[1]
Lz = datadummy[0].configuration.box[2]
box = freud.box.Box(Lx,Ly,Lz)
step=1
    
data_par = gsd.hoomd.open(name=filename, mode='rb')
simly=data_par[0].configuration.box[1]
simlz=data_par[0].configuration.box[2]
Area = simly*simlz
DII_time = np.empty((0,3),dtype=float)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


elements_per_process = (len(datadummy)-init)//size
start_index = rank*elements_per_process
end_index = (rank+1)*elements_per_process
if rank == size-1:
    end_index = start_index +len(datadummy) -(start_index+init)
    print("rank :",rank,"size-1 :",size-1,"start_index :",start_index,"end_index :",end_index,flush=True)


data2 = np.zeros((int((end_index-start_index)/step),3),dtype=float)
system = freud.AABBQuery.from_system((box,points))
distances = np.linalg.norm(box.wrap(points[bonds_array[:, 1]] - points[bonds_array[:, 0]]),axis=1)
neighbors = freud.locality.NeighborList.from_arrays(len(points),len(points),
                    bonds_array[:, 0],
                    bonds_array[:, 1],
                    distances,
                )

cl = freud.cluster.Cluster()
cl1=cl.compute(system=system, neighbors=neighbors)



#for frame in range(1,len(data_par),data_jump):
for frame in range(init+start_index,init+end_index,step):
        
        print("sample :",sample,"rank :",rank,"frame : ",frame,"int((frame-(start+start_index))/step) :",int((frame-(init+start_index))/step),"start_index :",start_index,"end_index :",end_index,flush=True)

        if int((frame-(init+start_index))/step) >=len(data2):
                break

        i = int((frame-0)/step)
        I1,I2=Interface_calc(frame,step)

        I1arr = np.asarray([[I1,0,0]])
        I2arr = np.asarray([[I2,0,0]])
        print(I1,I2)
        snap=data_par[frame]

        pos=snap.particles.position
        arr_polymer=snap.particles.typeid==1
        polypos=pos[arr_polymer]
        polystrand=polypos.reshape(int(len(polypos)/standLen),standLen,3)
        stran_x=polystrand[:,:,0]
        
        system = freud.AABBQuery.from_system((box,pos))
        cl_props = freud.cluster.ClusterProperties()
        cl_props.compute(system, cl.cluster_idx)
        compoly = cl_props.centers
        compoly = np.asarray(compoly[1:])
        #print(compoly.shape,polystrand.shape)

        #exit()
####################################################################################################################
        #poly_ids1 = np.where(((compoly[:,0]>=I1) &(compoly[:,0]<0)))[0]
        #poly_ids2 = np.where(((compoly[:,0]<=I2) &(compoly[:,0]>0)))[0]

        #com_sel1 = compoly[poly_ids1]
        #com_sel2 = compoly[poly_ids2]
        #if len(com_sel1)!=0:
        #    depthx1=np.fabs(box.wrap(I1arr - com_sel1)[:,0])
        #    meandepthx1=np.nanmean(depthx1,axis=0)
        #    NcA1=len(com_sel1)/Ly/Lz
        #    print(meandepthx1)
        #else:
        #    meandepthx1 =0
        #    NcA1 = 0
        
        #if len(com_sel2)!=0:
        #    depthx2=np.fabs(box.wrap(I2arr - com_sel2)[:,0])
        #    meandepthx2=np.nanmean(depthx2,axis=0)
        #    NcA2=len(com_sel2)/Ly/Lz
        #else:
        #    meandepthx2 = 0
        #    NcA2 = 0
        
    

        #avdepth = (meandepthx2+meandepthx1)/2
        #NcA = (NcA1+NcA2)/2
        #time = dt*snap.configuration.step

        #data2[int((frame-(init+start_index))/step),0] = time
        #data2[int((frame-(init+start_index))/step),1] = avdepth
        #data2[int((frame-(init+start_index))/step),2] = NcA

all_data2 = comm.gather(data2,root=0)
if rank==0:
    print("sample :",sample,flush=True)
    datatime_concatmean=np.concatenate(all_data2,axis=0)

    filename = "Diffusiondepth"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"

    with open(filename, 'w+') as f3:
        np.savetxt(f3, datatime_concatmean)
    print("saved file",filename)





