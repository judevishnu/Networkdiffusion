#!/usr/bin/python3
from scipy.spatial import KDTree
import hoomd
import hoomd.md
import sys
import gsd
import gsd.hoomd
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
##############################################
##      functions
##############################################
def group_update(timestep):
    for group in xgroups.values():
        group.force_update()


########################################################################
#           Set parameters
#######################################################################
sample=sys.argv[1]
N = int(sys.argv[2])

kbT=1.0
a=2.
k=30
r0=1.5
dt=0.001
epsilon=1
rcut=math.pow(2./a,1./6.)
dia=[]
dia.append(1.0)
dia.append(1.0)
sigma = [] # interaction parameter sigma 
#fraction=sys.argv[3]
volfracnew=float(sys.argv[3])
tvolfracnew=sys.argv[3]
sigma.append([(dia[0]+dia[0])/2.,(dia[0]+dia[1])/2.])
sigma.append([(dia[0]+dia[1])/2.,(dia[1]+dia[1])/2.])
#spacing = float(fraction)*(sigma[0][0])


filename= "initial_config"+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_kT_"+str(kbT)+".gsd"
#filename= "Reinitial_config"+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_kT_"+str(kbT)+"pressure0.gsd"
traj = gsd.hoomd.open(name=filename, mode='rb')

poss = numpy.copy(traj[len(traj)-1].particles.position)
bonds   = numpy.copy(traj[0].bonds.group)
bondstypeid   = numpy.copy(traj[0].bonds.typeid)
mass     = numpy.copy(traj[0].particles.mass)
typeid   = numpy.copy(traj[0].particles.typeid)
image =  numpy.copy(traj[len(traj)-1].particles.image)
diameter =   numpy.copy(traj[len(traj)-1].particles.diameter)
#real_pos = position + numpy.array([Lx,Ly,Lz])*image
special_pair = numpy.copy(traj[0].pairs.group)
special_pair_type = numpy.copy(traj[0].pairs.typeid)
vel = numpy.copy(traj[len(traj)-1].particles.velocity)

Lx=traj[len(traj)-1].configuration.box[0]
Ly=traj[len(traj)-1].configuration.box[1]
Lz=traj[len(traj)-1].configuration.box[2]
origLx= Lx
origLxby2= Lx/2
real_pos = poss + numpy.array([Lx,Ly,Lz])*image

position =  numpy.array((real_pos[:,0],poss[:,1],poss[:,2])).T
print(image)
pos_x = position[:,0]
max_pos_x = numpy.amax(pos_x)
min_pos_x = numpy.amin(pos_x)
#print(max_pos_x,min_pos_x,traj[0].configuration.box[0])

##Rescale particle coordinates and box###
constant_fac =(max_pos_x+min_pos_x)/2.0
Lxby2 = (max_pos_x-min_pos_x)/2.
Lx = 2*Lxby2
position[:,0] =  position[:,0]- (constant_fac)
Lzby2 = Lz/2.0
Lyby2 = Ly/2.0
poly_len = 102#int(Lz/sigma[0][0])
Ny = int(Ly/sigma[0][0])
comx = numpy.mean(position,axis=0)
position[:,0]=position[:,0]-comx[0]


################################################
# Wall Position and defining diffusion polymers
################################################
WallPos = Lxby2 +sigma[0][0]
delta = 40*sigma[0][0]
#delta1=90*sigma[0][0]
delta1=200*sigma[0][0]
#delta1=46*sigma[0][0]
Lx = Lx+delta1
WallPos = Lxby2 +sigma[0][0]
Lxby2   = Lx/2.0
nLx = 697.497#2*(WallPos-sigma[0][0])+400
nLxby2=nLx/2

Nx_remainz = int((Lxby2-sigma[0][0]-WallPos)/sigma[0][0])

N_remainz= 2*Nx_remainz*poly_len*Ny
lengthavail=Lxby2-WallPos
#lengthavail=nnLxby2-WallPos
postopoly = lengthavail/2+WallPos
postopoly = lengthavail+WallPos
print(postopoly,WallPos,nLx)
#exit()
vol_frac = N_remainz/(delta*Ly*Lz)
#vol_frac = "%.5f" % vol_frac
print(N_remainz/(Lx*Ly*Lz),vol_frac,Lx,Ly,Lz)


N_remainz_new = int(N_remainz*volfracnew/vol_frac)

lcm = numpy.lcm(2,poly_len)

if N_remainz_new % lcm:
    print(N_remainz_new)
    N_remainz_new = N_remainz_new -(N_remainz_new % lcm)

    print(N_remainz_new)
else:
    N_remainz_new = N_remainz_new


geldens = len(poss)/(Lx*Ly*Lz)
#geldens = "%.5f" % geldens

no_polymersto_remove=int( (-N_remainz_new + N_remainz)/poly_len)

remaining_polymers = int(N_remainz_new/poly_len/2)
remaining_polymers = N_remainz_new/poly_len
if remaining_polymers%2:
    remaining_polymers = int((remaining_polymers -(remaining_polymers%2))/2)
else:
    remaining_polymers = int((remaining_polymers)/2 )
N_remainz_new = 2*remaining_polymers*poly_len


polydens=N_remainz_new/(Lx*Ly*Lz)
#polydens = "%.5f" % polydens
densratio = polydens/geldens
densratio = "%.5f" % densratio
print(N_remainz_new,remaining_polymers)
print(N_remainz_new/(delta*Ly*Lz))

print(Nx_remainz,Ny,poly_len,vol_frac,volfracnew)

print(WallPos,N_remainz_new/(delta*Ly*Lz))

vol_frac=N_remainz_new/(delta*Ly*Lz)
vol_frac = "%.5f" % vol_frac

print(vol_frac,remaining_polymers,N_remainz_new,2*remaining_polymers*(poly_len),Nx_remainz*Ny,2*(Nx_remainz*Ny),2*(remaining_polymers)*(poly_len-1)+len(bonds))
molefrac = polydens/(geldens+polydens)

molefrac = "%.5f" % molefrac
#############################################
##  Coordinates for diffusive polymers
############################################
zcoords = numpy.linspace(-int(poly_len/2)+sigma[0][0], int(poly_len/2)-sigma[0][0],poly_len)
ycoords = numpy.linspace(-Lyby2+sigma[0][0], Lyby2-sigma[0][0],Ny)

#xcoords = numpy.linspace(-WallPos-sigma[0][0],-Lxby2+sigma[0][0],Nx_remainz)

#xcoords1 = numpy.linspace(WallPos+sigma[0][0],Lxby2-sigma[0][0],Nx_remainz)
print(postopoly,Lxby2,WallPos,Nx_remainz)

nLxby2 = nLx/2
xcoords = numpy.linspace(-postopoly -sigma[0][0],-nLxby2+sigma[0][0],Nx_remainz)

xcoords1 = numpy.linspace(postopoly +sigma[0][0],nLxby2-sigma[0][0],Nx_remainz)

#xcoords = numpy.linspace(-WallPos -sigma[0][0],-nLxby2+sigma[0][0],Nx_remainz)

#xcoords1 = numpy.linspace(WallPos +sigma[0][0],nLxby2-sigma[0][0],Nx_remainz)
print(zcoords)
print(ycoords)
print(xcoords)
#exit()
#############################################
#############################################################
# Particle Initialisation in the box
#############################################################
#resize = len(bonds)+(poly_len-1)*2*Nx_remainz*Ny ## New bond length
resize = len(bonds)+2*(poly_len-1)*remaining_polymers
#count2= len(bonds)+(Nx_remainz)*Ny*(poly_len-1)
count2= len(bonds)+(remaining_polymers)*(poly_len-1)
count=len(bonds)
print(len(xcoords),count,count2)


context = hoomd.context.initialize("--mode=gpu")
#nLx = 2*(WallPos-sigma[0][0])+400
#snapshot = hoomd.data.make_snapshot(N=N,box=hoomd.data.boxdim(Lx=Lx, Ly=Ly, Lz=Lz),particle_types=['A'],bond_types=['polymer'])
#resize = len(bonds)
snapshot = hoomd.data.make_snapshot(N=N+N_remainz_new,box=hoomd.data.boxdim(Lx=nLx, Ly=Ly, Lz=Lz),particle_types=['A','B'],bond_types=['polymer'])

snapshot.bonds.resize(resize)
snapshot.particles.position[0:len(position)]=position[0:len(position)]



snapshot.particles.typeid[0:len(position)] = typeid[0:len(position)]
snapshot.particles.diameter[0:len(position)] = diameter[0:len(position)]
snapshot.bonds.group[0:len(bonds)]=bonds[:]
snapshot.bonds.typeid[0:len(bondstypeid)]=bondstypeid[:]
snapshot.particles.velocity[0:len(position)]=vel[0:len(position)]           


### Diffusive polymer particle intialization ##
flag =False
poly_id=0
for i in range(Nx_remainz):

    for j in range(Ny):
        
        #print(poly_id)
        poly_id = (i*Ny+j)
        if poly_id >= remaining_polymers:
            flag = True
            break
        for k in range(poly_len):

            index = N+Ny*poly_len*i+j*poly_len+k
            index2 = N+int(N_remainz_new/2)+Ny*poly_len*i+j*poly_len+k
            
            snapshot.particles.position[index,0]=xcoords[i]
            snapshot.particles.position[index,1]=ycoords[j]
            snapshot.particles.position[index,2]=zcoords[k]
            snapshot.particles.typeid[index] = 1
            snapshot.particles.diameter[index] = dia[0]
            snapshot.particles.position[index2,0]=xcoords1[i]
            snapshot.particles.position[index2,1]=ycoords[j]
            snapshot.particles.position[index2,2]=zcoords[k]
            snapshot.particles.typeid[index2] = 1
            snapshot.particles.diameter[index2] = dia[0]
            snapshot.particles.mass[index2] = 1.0
            snapshot.particles.mass[index] = 1.0

            
            index1=N+Ny*poly_len*i+j*poly_len+k+1
            index21=N+Ny*poly_len*i+j*poly_len+k+int(N_remainz_new/2)+1
            #poly_id=poly_id+2
            if k<poly_len-1 and count< len(bonds)+remaining_polymers*(poly_len-1):
                snapshot.bonds.group[count,0] = index
                snapshot.bonds.group[count,1] = index1
                snapshot.bonds.typeid[count]=0
                print(snapshot.bonds.group[count]) 
                count+=1

            if k<poly_len-1 and count2< resize:
                 snapshot.bonds.group[count2,0] = index2
                 snapshot.bonds.group[count2,1] = index21
                 snapshot.bonds.typeid[count2]=0
                 print(snapshot.bonds.group[count2]) 
                 count2+=1
    if flag==True:
        #print(N_remainz_new+N,len(snapshot.particles.position))
        break

print(snapshot.particles.position)
print(snapshot.bonds.group)
print(N+N_remainz_new,len(snapshot.particles.position))

#################################################################3


system = hoomd.init.read_snapshot(snapshot);
all = hoomd.group.all()

#hoomd.dump.gsd("initial4.gsd", group=all ,period=None, overwrite =True);  #dump initial cofiguration:


##############################################################################################
#nl = hoomd.md.nlist.tree(r_buff = 0.5, check_period = 1)
#nl = hoomd.md.nlist.cell(r_buff = 0.5, check_period = 1)
nl = hoomd.md.nlist.stencil(r_buff = 0.5, check_period = 1)

wca = hoomd.md.pair.lj(r_cut = rcut,nlist = nl)
wca.set_params(mode='shift')

wca.pair_coeff.set('A','A', epsilon = epsilon, sigma =sigma[0][0],r_cut = sigma[0][0]*rcut,alpha =a)
wca.pair_coeff.set('B','B', epsilon = epsilon, sigma =sigma[0][0],r_cut = sigma[0][0]*rcut,alpha =a)
wca.pair_coeff.set('A','B', epsilon = epsilon, sigma =sigma[0][0],r_cut = sigma[0][0]*rcut,alpha =a)
hoomd.md.integrate.mode_standard(dt= dt)
fene = hoomd.md.bond.fene()
fene.bond_coeff.set('polymer', k =k, r0 = r0, sigma = sigma[0][0], epsilon = epsilon)



hoomd.md.integrate.mode_standard(dt= 0.001)


integrator  = hoomd.md.integrate.langevin(group = all, kT = kbT, seed = seed3,dscale=1.0)
#################################################################################################
##      Adding walls
###############################################################################################
#define wall surfaces and group them
wall1 = hoomd.md.wall.plane(origin = (WallPos,0,0),normal =(-1,0,0),inside =False)
wall3 = hoomd.md.wall.plane(origin = (WallPos,0,0),normal =(1,0,0),inside =False)
wall2 = hoomd.md.wall.plane(origin = (-WallPos,0,0),normal =(1,0,0), inside =False)
wall4 = hoomd.md.wall.plane(origin = (-WallPos,0,0),normal =(-1,0,0), inside =False)
walls = hoomd.md.wall.group([wall1,wall2,wall3,wall4])
#add walls
walllj = hoomd.md.wall.lj(walls, r_cut = rcut)
walllj.force_coeff.set('A', sigma = 0.1*sigma[0][0], epsilon= epsilon)
walllj.force_coeff.set('B', sigma = 0.1*sigma[0][0], epsilon= epsilon)
###############################################################################################
#filename3 = "stress_logfile_diamond_wca_fene_network"+str(sample)+"_uni_deform_Ntot_"+str(N_constraint)+"_Nx_"+str(N_poly1)+"_Ny_"+str(N_poly2)+"_Nz_"+str(poly_len)+"_kT_"+str(kbT)+"_rho_"+str(density)+"_D_"+str(D)+"_alpha_"+str(alpha)+".log"
#log1=hoomd.analyze.log(filename = filename3,quantities = ['temperature','pressure_xx','pressure_yy','pressure_zz','volume','lz','lx','ly'], period=10000,overwrite=True)
#filename2 = "temporary"+str(tvolfracnew)+".gsd"

filename2 = "temporaryinit"+"_diamond_network_diffusive"+"_Ntot_"+str(N+N_remainz_new)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
#filename2 = "temporaryinit"+"_diamond_network_diffusive"+"_Ntot_"+str(N+N_remainz_new)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+"pressure0.gsd"

hoomd.dump.gsd(filename2, group=all ,period=None, overwrite =True,dynamic=['momentum']);  #dump initial cofiguration:
#hoomd.dump.gsd(filename2, group=all ,period=99999999, overwrite =True,dynamic=['momentum']);  #dump initial cofiguration:
#hoomd.dump.gsd(filename2, group=all ,period=89999999, overwrite =True,dynamic=['momentum']);  #dump initial cofiguration:


#hoomd.run(10e7)
hoomd.run(16e7)

