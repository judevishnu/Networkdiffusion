#!/usr/bin/python3
from scipy.spatial import KDTree
import hoomd
import hoomd.md
import sys
#import gsd
#import gsd.hoomd
import time
import numpy
import random 
import copy
import freud
import math
################################################
#  Random number seed
###############################################
seed1 = random.randint(1,999999999)
seed2 = random.randint(1,999999999)
seed3 = random.randint(1,999999999)



########################################################################
#           Set parameters
#######################################################################
#sample=sys.argv[1]
dia=[]
dia.append(1.0)
dia.append(1.0)
sigma = [] # interaction parameter sigma 
#fraction=sys.argv[2]
sigma.append([(dia[0]+dia[0])/2.,(dia[0]+dia[1])/2.])
sigma.append([(dia[0]+dia[1])/2.,(dia[1]+dia[1])/2.])
#spacing = float(fraction)*(sigma[0][0])
tauP=1.0
tauT=0.5
a=2
epsilon=1
rcut = (2/a)**(1.0/6.)
dt=0.0001
r0=1.5
k=30
kbT=4.3
coords = []
coords1=numpy.empty((0,3),dtype='float')
coords2=numpy.empty((0,3),dtype='float')
nlat=[4,3,3]
alat=100
Lx = alat*nlat[0]*2
Ly = alat*nlat[1]*2
Lz = alat*nlat[2]*2
print(Lx,Ly,Lz)
No_crosslinks = 8*8*nlat[0]*nlat[1]*nlat[2]
print(No_crosslinks)
Lx1=1.5*Lx
Lx1by2 = Lx1/2.0
Lxby2 = Lx/2.0
Lyby2 = Ly/2.0
Lzby2 = Lz/2.0
WallPos = Lxby2+sigma[0][0]

rmax = math.sqrt(0.25*2)*alat
############################################################################
##          Initialising functions
############################################################################
def initialize_coordinates():
    basis =[[0.0,0.0,0.0],[0.5,0.5,0.0],[0.5,0.0,0.5],[0,0.5,0.5],[0.25,0.25,0.25],[0.75,0.75,0.25],[0.75,0.25,0.75],[0.25,0.75,0.75]]
    
    
    for i in range(-nlat[0],nlat[0]):
        for j in range(-nlat[1],nlat[1]):
            for k in range(-nlat[2],nlat[2]):
                for m in range(8):
                    x = (basis[m][0]+i)*alat
                    y = (basis[m][1]+j)*alat
                    z = (basis[m][2]+k)*alat
                    
                    coords.append([x,y,z])
    
def add_polymers():
    zcoords = numpy.linspace(-Lzby2+sigma[0][0], Lzby2-sigma[0][0], Nz)
    ycoords = numpy.linspace(-Lyby2+sigma[0][0], Lyby2- sigma[0][0], Ny)
    xcoords = numpy.linspace(-WallPos-sigma[0][0]*0.1,-Lxby2+sigma[0][0],Nx_remainz)
    xcoords1 = numpy.linspace(WallPos+sigma[0][0]*0.1,Lxby2-sigma[0][0],Nx_remainz)
    global coords1
    global coords2
    for i in range(Nx_remainz):
        for j in range(Ny):
            for k in range(Nz):
                x1 = xcoords1[i]
                x = xcoords[i]
                y = ycoords[j]
                z = zcoords[k]
                
                coords1=numpy.append(coords1,[[x1,y,z]],axis=0)
                coords2=numpy.append(coords2,[[x,y,z]],axis=0)
                




###################################################################################################   
initialize_coordinates()

coordinates = numpy.asarray(coords)
bond_list=[]


print("Number of crosslinks ", len(coordinates))

box = freud.box.Box(Lx =Lx1, Ly= Ly, Lz =Lz)
aq = freud.locality.AABBQuery(box,coordinates)


query_result = aq.query(coordinates, dict(r_max=rmax,exclude_ii=True))
nlist = query_result.toNeighborList()
for (i, j) in nlist:
    bond_list.append([i,j])


print(len(bond_list))
for x in bond_list:
    z = copy.copy(x)
    z.reverse()
    if z in bond_list and x in bond_list:
        bond_list.remove(z)

for i in range(len(bond_list)):
    print(bond_list[i])



    


bond_array = numpy.asarray(bond_list)
bonds = numpy.empty((0,2),dtype=int)
box1  = numpy.empty((0,3),dtype=float)
box1=numpy.append(box1,[[Lx1,Ly,Lz]],axis=0)
print(box1[0,0])
avg_beads=0
print(len(coordinates),len(bond_array))
for x in bond_array:
    dist = numpy.linalg.norm(box.wrap(coordinates[x[1]]-coordinates[x[0]]))
    print(dist)
    dr = box.wrap(coordinates[x[1]]-coordinates[x[0]])
    division = int(dist/dia[0])
    const = dr/division
    b0 = dist/division
    no_beads = division-1
    #print(no_beads)
    avg_beads+=no_beads+1
    bonds=numpy.append(bonds,[[x[0],len(coordinates)]],axis=0)
    
    for k in range(1,no_beads+1):
        pos = box.wrap(k*const+coordinates[x[0]])
        if k == no_beads:
            bonds = numpy.append(bonds,[[len(coordinates),x[1]]],axis=0)
        else:
            bonds=numpy.append(bonds,[[len(coordinates),len(coordinates)+1]],axis=0)
        coordinates=numpy.append(coordinates,[pos],axis=0)

#add_polymers()
print("Number of particles", len(coordinates))
#print(avg_beads/len(bond_array))
#exit()
N = len(coordinates)
#coordinates=numpy.append(coordinates,[coords1],axis=0)
#print(coordinates)
poss = coordinates[:,0]
max_pos_x = numpy.amax(poss)
WallPos = Lxby2+0.6*sigma[0][0] 
context = hoomd.context.initialize("--mode=gpu")

snapshot = hoomd.data.make_snapshot(N=len(coordinates),box=hoomd.data.boxdim(Lx=Lx, Ly=Ly, Lz=Lz),particle_types=['A'],bond_types=['polymer'])

snapshot.bonds.resize( len(bonds) )
snapshot.particles.position[0:len(coordinates)]=coordinates[0:len(coordinates)]

snapshot.particles.typeid[0:len(coordinates)] = 0
snapshot.particles.diameter[0:len(coordinates)] = dia[0]
snapshot.bonds.group[0:len(bonds)]=bonds[0:len(bonds)]
snapshot.bonds.typeid[0:len(bonds)]=0
           
system = hoomd.init.read_snapshot(snapshot);
all = hoomd.group.all()


filename5= "initial_config"+"_diamond_network_diffusive"+"_Ntot_"+str(len(coordinates))+"_kT_"+str(kbT)+".gsd"
#hoomd.dump.gsd(filename5, group=all,period=None ,time_step=1300000, overwrite =True);  #dump initial cofiguration:

#####################################################################################

nl = hoomd.md.nlist.tree(r_buff = 0.5, check_period = 1)
wca = hoomd.md.pair.lj(r_cut = rcut,nlist = nl)
wca.set_params(mode='shift')

wca.pair_coeff.set('A','A', epsilon = epsilon, sigma =sigma[0][0],r_cut = sigma[0][0]*rcut,alpha =a)
hoomd.md.integrate.mode_standard(dt= dt)
fene = hoomd.md.bond.fene()
fene.bond_coeff.set('polymer', k =k, r0 = r0, sigma = sigma[0][0], epsilon = epsilon)
dt = 0.001

hoomd.md.integrate.mode_standard(dt= dt)

#integrator =  hoomd.md.integrate.langevin(group = all, kT = kbT,seed =seed3)
#integrator =  hoomd.md.integrate.npt(group = all, kT =kbT, tau=tauT,tauP=tauP,S=[0.0,0.0,0.0,0.0,0.0,0.0],z=False,couple="none",rescale_all=True)
integrator = hoomd.md.integrate.berendsen(group=all, kT=kbT, tau=tauT)
integrator.randomize_velocities(seed=42)
###################################################################

hoomd.run(3e5);

hoomd.dump.gsd(filename5, group=all,period=3e5,dynamic=['momentum'] , overwrite =True);  #dump initial cofiguration:
#####################################################################################
integrator.disable()
#integrator =  hoomd.md.integrate.npt(group = all, kT =kbT, tau=tauT,tauP=tauP,S=[0.0,0.0,0.0,0.0,0.0,0.0],x=False,couple="none",rescale_all=True)
integrator =  hoomd.md.integrate.npt(group = all, kT =kbT, tau=tauT,tauP=tauP,P=0.001,couple="none",rescale_all=True)
#integrator =  hoomd.md.integrate.npt(group = all, kT =kbT, tau=tauT,tauP=tauP,S=[0.0,0.0,0.0,0.0,0.0,0.0],couple="none",rescale_all=True)
hoomd.run(1e6)



