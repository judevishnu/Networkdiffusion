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
vol_frac = str(sys.argv[3])
densratio = str(sys.argv[4])
molefrac = str(sys.argv[5])

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
sigma.append([(dia[0]+dia[0])/2.,(dia[0]+dia[1])/2.])
sigma.append([(dia[0]+dia[1])/2.,(dia[1]+dia[1])/2.])
#spacing = float(fraction)*(sigma[0][0])
period =16e7
#period =800000
phase =800000

#filename2 = "temporaryinit"+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"

filename2 = "/lustre/project/nhr-trr146/jvishnu/Regnetbigger/isotropic/temporaryinit"+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"



traj = gsd.hoomd.open(name=filename2, mode='rb')
frame=len(traj)-1
print(len(traj))
Lx=traj[len(traj)-1].configuration.box[0]
print(Lx)
Ly=traj[len(traj)-1].configuration.box[1]
Lz=traj[len(traj)-1].configuration.box[2]
Lxby2 =Lx/2
Lyby2 =Ly/2
Lzby2 =Lz/2
"""
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
N = len(poss)
Lxby2 = Lx/2
Lyby2 = Ly/2
Lzby2 = Lz/2
WallPos=149.74859619140625
context = hoomd.context.initialize("--mode=gpu")

snapshot = hoomd.data.make_snapshot(N=N,box=hoomd.data.boxdim(Lx=Lx, Ly=Ly, Lz=Lz),particle_types=['A','B'],bond_types=['polymer'])
resize = len(bonds)
snapshot.bonds.resize(resize)
snapshot.particles.position[0:len(poss)]=poss[0:len(poss)]
snapshot.particles.typeid[0:len(poss)] = typeid[0:len(poss)]
snapshot.particles.diameter[0:len(poss)] = diameter[0:len(poss)]
snapshot.bonds.group[0:len(bonds)]=bonds[:]
snapshot.bonds.typeid[0:len(bondstypeid)]=bondstypeid[:]
snapshot.particles.velocity[0:len(poss)]=vel[0:len(poss)]


##############################################################################################
system = hoomd.init.read_snapshot(snapshot);
"""

WallPos= 134.83063888549805#149.74859619140625
context = hoomd.context.initialize("--mode=gpu")
system=hoomd.init.read_gsd(filename2,frame=frame)

all = hoomd.group.all()

##############################################################################################
rbuf=0.6842105263157895 
chkperiod=40
nl = hoomd.md.nlist.tree(r_buff = rbuf, check_period =chkperiod)
#nl = hoomd.md.nlist.stencil(r_buff = 0.5, check_period = 1)

wca = hoomd.md.pair.lj(r_cut = rcut,nlist = nl)
wca.set_params(mode='shift')

wca.pair_coeff.set('A','A', epsilon = epsilon, sigma =sigma[0][0],r_cut = sigma[0][0]*rcut,alpha =a)
wca.pair_coeff.set('B','B', epsilon = epsilon, sigma =sigma[0][0],r_cut = sigma[0][0]*rcut,alpha =a)
wca.pair_coeff.set('A','B', epsilon = epsilon, sigma =sigma[0][0],r_cut = sigma[0][0]*rcut,alpha =a)
hoomd.md.integrate.mode_standard(dt= dt)
fene = hoomd.md.bond.fene()
fene.bond_coeff.set('polymer', k =k, r0 = r0, sigma = sigma[0][0], epsilon = epsilon)


wall1 = hoomd.md.wall.plane(origin = (WallPos,0,0),normal =(-1,0,0),inside =False)
wall3 = hoomd.md.wall.plane(origin = (WallPos,0,0),normal =(1,0,0),inside =False)
wall2 = hoomd.md.wall.plane(origin = (-WallPos,0,0),normal =(1,0,0), inside =False)
wall4 = hoomd.md.wall.plane(origin = (-WallPos,0,0),normal =(-1,0,0), inside =False)
walls = hoomd.md.wall.group([wall1,wall2,wall3,wall4])
#add walls
walllj = hoomd.md.wall.lj(walls, r_cut = rcut)
walllj.force_coeff.set('A', sigma = 0.1*sigma[0][0], epsilon= epsilon)
walllj.force_coeff.set('B', sigma = 0.1*sigma[0][0], epsilon= epsilon)



hoomd.md.integrate.mode_standard(dt= dt)


integrator  = hoomd.md.integrate.langevin(group = all, kT = kbT, seed = seed3,dscale=1.0)
#rbuff,chckperiod=nl.tune(warmup=200000, r_min=0.5, r_max=1.0, jumps=20, steps=5000, set_max_check_period=True, quiet=False)
#print(rbuff,chckperiod)
#exit()
################################################################################################

#filename2 = "/lustre/project/nhr-trr146/jvishnu/Regnetbigger/isotropic/finaltrajectoryIso"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
filename2 = "/lustre/project/nhr-trr146/jvishnu/Regnetbigger/isotropic/trajectoryInitIso"+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"



#filename2 = "temporaryinit"+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"

print(filename2)
#hoomd.dump.gsd(filename2, group=all ,period=410199999, overwrite =False,dynamic=['momentum']);  #dump initial cofiguration:
#hoomd.dump.gsd(filename2, group=all ,period=period-1,overwrite =False,phase=phase,dynamic=['momentum']);  #dump initial cofiguration:
hoomd.dump.gsd(filename2, group=all ,period=160000000-1,overwrite =False,dynamic=['momentum']);  #dump initial cofiguration:
#hoomd.dump.gsd(filename2, group=all ,period=800000,overwrite =False,dynamic=['momentum']);  #dump initial cofiguration:

###############################################################################################
"""
cutoffx = sigma[0][0]
lcutoffx =cutoffx
lengthx = int(Lxby2/lcutoffx)
lcx = Lxby2/lengthx
shiftx = int(lengthx)
cellnox = 2*lengthx
nbins = cellnox

dx = cutoffx
vbin  = Ly*Lz*cutoffx


xbins = numpy.empty(1+nbins,dtype=float)
for x in range(nbins+1):
    xbins[x] = (x)*lcutoffx-Lxby2
 
xgroups= {}
xcompute= {}
stensorlist = []
for i in range(nbins):
    xname= 'x' + str(i)
    xgroups[xname] = hoomd.group.cuboid(xname,xmin=xbins[i],xmax=xbins[i+1])
    xcompute[xname]= hoomd.compute.thermo(xgroups[xname])
    stensorlist .append('pressure_xx_'+ xname)
    stensorlist .append('pressure_yy_'+ xname)
    stensorlist .append('pressure_zz_'+ xname)

group_update1 = hoomd.analyze.callback(callback =  group_update, period =period)
filename5= "pressure_component_stensorlist"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(N)+"_vol_frac"+str(vol_frac)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".dat"
pressurelog = hoomd.analyze.log(filename = filename5, quantities = stensorlist , period = period, phase=phase, overwrite=False)
"""
hoomd.run_upto(16e7)
