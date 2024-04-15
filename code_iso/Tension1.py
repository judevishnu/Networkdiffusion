import numpy as np
#import matplotlib.pyplot as plt
import gsd.hoomd
import numba
from numba import  float64,int32,int64,float32,prange
import freud
import time
import math
import pandas as pd
import sys
@numba.njit(cache=True)
def WCA(r, rc, eps, sigma):
    """
    WCA potential function
    """
    if r >= rc:
        return 0
    else:
        sig_by_r6 = (sigma / r) ** 6
        sig_by_r12 = sig_by_r6*sig_by_r6
        return 48 * eps * (sig_by_r12  - sig_by_r6 )/r

@numba.njit(cache=True)
def FENE(r,sigma, r0,rcut, k):
    """
    FENE potential function
    """
    if r >= r0:

        fene= np.inf
    else:
        fene= -k * r/(1-(r/r0)**2)
    if r>=rcut:
        fwca=0
    else:
        sig_by_r6 = (sigma / r) ** 6
        sig_by_r12 = sig_by_r6*sig_by_r6
        fwca =24 * eps * (2*sig_by_r12  - sig_by_r6 )/r
    return fene+fwca


@numba.njit(cache=True)
def PBC(x,L):
    if x>L*0.5:
        x=x-L
    if x<=-L*0.5:
        x=x+L
    return x
@numba.njit(cache=True)
def calpres(f,du,dv,r):
    return f*du*dv/r


@numba.njit(fastmath=True,cache=True)
def comparison(bonds,i,j):
    for k in range(bonds.shape[0]):
        if ((i==bonds[k,0]) & (j==bonds[k,1])):
            return True
    return False




# @numba.njit(cache=True)
# def local_pressureFENE(subpress,bonds,pos,dx,box,n_bins,eps,sigma,k,r0,rc_n):
#     for i,j in bonds:
#         bin1 =int((pos[i,0]+box[0]/2)/dx)
#         bin2 = int((pos[j,0]+box[0]/2)/dx)
#
#
#         # bin1=np.mod(bin1,n_bins)
#         # bin2=np.mod(bin2,n_bins)
#
#         dbin =int((bin1-bin2)%n_bins)
#         if dbin>n_bins/2:
#             dbin =dbin-n_bins
#
#         Nintx = abs(dbin)
#         interbins=np.zeros(Nintx+2,dtype=float)
#         interbins[Nintx+1]=1
#         m=1
#
#         xdr=pos[i,0]-pos[j,0]
#         ydr=pos[i,1]-pos[j,1]
#         zdr=pos[i,2]-pos[j,2]
#
#         xdr=PBC(xdr,box[0])
#         ydr=PBC(ydr,box[1])
#         zdr=PBC(zdr,box[2])
#
#         r=np.sqrt(xdr*xdr+ydr*ydr+zdr*zdr)
#         if r>r0:
#              print(r)
#         f=FENE(r,sigma, r0,rc_n, k)
#         dr = np.array([xdr,ydr,zdr])
#         if dbin<0:
#             for k in range(0,Nintx):
#                 bound = (bin2-k)*dx-box[0]/2
#                 diff =(bound -pos[j,0])
#                 #diff=PBC(diff,box[0])
#                 interbins[m] =diff/xdr
#                 m+=1
#         if dbin>=0:
#             for k in range(0,Nintx):
#                 bound = (bin2+1+k)*dx-box[0]/2
#                 diff =(bound -pos[j,0])
#                 #diff=PBC(diff,box[0])
#                 interbins[m] =diff/xdr
#                 m+=1
#         if m>1:
#             interbins=np.sort(interbins)
#
#         for t in range(0,Nintx+1):
#             dweight = (-interbins[t]+interbins[t+1])
#             midxtemp = 0.5*(interbins[t]+interbins[t+1])
#             mid = xdr*midxtemp+pos[j,0]+box[0]/2
#             val = (mid/box[0])
#             mid=(val-int(val))*box[0]
#             if mid<0:
#                 mid+=box[0]
#
#             bin = int(mid/dx)
#             bin=np.mod(bin,n_bins)
#
#             if dweight>1:
#                 print(dweight,bin,bin1,bin2,pos[i,0],pos[j,0])
#             subpress[bin,0,0]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[0],dr[0],r)
#             subpress[bin,1,1]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[1],dr[1],r)
#             subpress[bin,2,2]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[2],dr[2],r)
#             subpress[bin,1,0]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[1],dr[0],r)
#             subpress[bin,0,1]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[0],dr[1],r)
#             subpress[bin,2,0]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[2],dr[0],r)
#             subpress[bin,0,2]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[0],dr[2],r)
#             subpress[bin,2,1]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[2],dr[1],r)
#             subpress[bin,1,2]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[1],dr[2],r)






# @numba.njit(cache=True)
# def local_pressureWCA(subpress,n,pos,dx,box,n_bins,eps,sigma,k,r0,rc_n):
#
#     for i,j in n:
#         #
#         bin1 =int((pos[i,0]+box[0]/2)/dx)
#         bin2 = int((pos[j,0]+box[0]/2)/dx)
#
#         bin1 =int(np.trunc((pos[i,0]+box[0]/2)/dx))
#         bin2 = int(np.trunc((pos[j,0]+box[0]/2)/dx))
#         if (bin1>=n_bins)or(bin1<0):
#             print("bin1",bin1,n_bins)
#         if (bin2>=n_bins)or(bin2<0):
#             print(bin2,n_bins)
#
#         dbin =int((bin1-bin2)%n_bins)
#         if dbin>n_bins/2:
#             dbin =dbin-n_bins
#
#         Nintx = abs(dbin)
#         interbins=np.zeros(Nintx+2,dtype=float)
#         interbins[Nintx+1]=1
#         m=1
#
#         xdr=pos[i,0]-pos[j,0]
#         ydr=pos[i,1]-pos[j,1]
#         zdr=pos[i,2]-pos[j,2]
#
#         xdr=PBC(xdr,box[0])
#         ydr=PBC(ydr,box[1])
#         zdr=PBC(zdr,box[2])
#
#         r=np.sqrt(xdr*xdr+ydr*ydr+zdr*zdr)
#         if r>sigma:
#              print("Iam in WCA",r)
#         f=WCA(r,sigma,eps,sigma)
#         dr = np.array([xdr,ydr,zdr])
#
#         if dbin<0:
#             for k in range(0,Nintx):
#                 bound = (bin2-k)*dx-box[0]/2
#                 diff =(bound -pos[j,0])
#                 #diff=PBC(diff,box[0])
#                 interbins[m] =diff/xdr
#                 m+=1
#         if dbin>=0:
#             for k in range(0,Nintx):
#                 bound = (bin2+1+k)*dx-box[0]/2
#                 diff =(bound -pos[j,0])
#                 #diff=PBC(diff,box[0])
#                 interbins[m] =diff/xdr
#                 m+=1
#         if m>1:
#             interbins=np.sort(interbins)
#
#         for t in range(0,Nintx+1):
#             dweight = (-interbins[t]+interbins[t+1])
#             midxtemp = 0.5*(interbins[t]+interbins[t+1])
#             mid = xdr*midxtemp+pos[j,0]+box[0]/2
#             val = (mid/box[0])
#             mid=(val-int(val))*box[0]
#             if mid<0:
#                 mid+=box[0]
#
#             #bin = int(np.trunc(mid/dx))
#             bin = int(mid/dx)
#             bin=np.mod(bin,n_bins)
#             # print(bin,)
#             #if ((bin>n_bins)or(bin<0)):
#             # if (bin>=n_bins):
#             #     print(dweight,bin,n_bins,bin1,bin2,pos[i,0],pos[j,0])
#             #
#             # if (bin<0):
#             #     print(dweight,bin,n_bins,bin1,bin2,pos[i,0],pos[j,0])
#
#
#             #bin = PBC(bin,n_bins)
#
#             if dweight>1:
#                 print(dweight,bin,bin1,bin2,pos[i,0],pos[j,0])
#             subpress[bin,0,0]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[0],dr[0],r)
#             subpress[bin,1,1]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[1],dr[1],r)
#             subpress[bin,2,2]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[2],dr[2],r)
#             subpress[bin,1,0]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[1],dr[0],r)
#             subpress[bin,0,1]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[0],dr[1],r)
#             subpress[bin,2,0]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[2],dr[0],r)
#             subpress[bin,0,2]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[0],dr[2],r)
#             subpress[bin,2,1]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[2],dr[1],r)
#             subpress[bin,1,2]+=(dweight/box[1]/box[2]/dx)*calpres(f,dr[1],dr[2],r)
#         # c+=1
    # print(c)

@numba.njit(cache=True)
def Planecalc(subpress,xj,dr,f,dx,nbin,r,box):
    xi = dr+xj
    if xj[0]<xi[0]:
        p1 = xj[0]+box[0]/2+box[0]
        p2= xi[0]+box[0]/2+box[0]
        bin1 =int(np.floor(p1/dx))
        bin2 =int(np.floor(p2/dx))
        nint = bin2-bin1
    else:
        if xi[0]<xj[0]:
            p1 = xi[0]+box[0]/2+box[0]
            p2 = xj[0]+box[0]/2+box[0]
            bin1 =int(np.floor(p1/dx))
            bin2 =int(np.floor(p2/dx))
            nint = bin2-bin1
        else:
            nint=0
            bin1=int(np.floor((xj[0]+box[0]/2)/dx))
    cpbin1=bin1
    if nint>0:
        inv_pij = 1.0/(p2-p1)
        binid = int(bin1%nbin)
        frac = ((bin1+1)*dx-p1)*inv_pij
        if frac>=1:
            print("Here1",p1,p2,binid,frac,bin1,bin2,inv_pij,nbin)
        if frac<0:
            print("Here1",p1,p2,binid,frac,bin1,bin2,inv_pij)
        subpress[binid,0,0]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[0],dr[0],r)
        subpress[binid,1,1]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[1],dr[1],r)
        subpress[binid,2,2]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[2],dr[2],r)
        subpress[binid,1,0]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[1],dr[0],r)
        subpress[binid,0,1]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[0],dr[1],r)
        subpress[binid,2,0]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[2],dr[0],r)
        subpress[binid,0,2]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[0],dr[2],r)
        subpress[binid,2,1]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[2],dr[1],r)
        subpress[binid,1,2]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[1],dr[2],r)
        for binid in range(cpbin1+1,nint):
            frac =dx*inv_pij
            if frac>=1:
                print("Here2",p1,p2,binid,frac,bin1,bin2)
            if frac<0:
                print("Here2",p1,p2,binid,frac,bin1,bin2)
            subpress[binid,0,0]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[0],dr[0],r)
            subpress[binid,1,1]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[1],dr[1],r)
            subpress[binid,2,2]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[2],dr[2],r)
            subpress[binid,1,0]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[1],dr[0],r)
            subpress[binid,0,1]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[0],dr[1],r)
            subpress[binid,2,0]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[2],dr[0],r)
            subpress[binid,0,2]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[0],dr[2],r)
            subpress[binid,2,1]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[2],dr[1],r)
            subpress[binid,1,2]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[1],dr[2],r)

        binid = int(bin2%nbin)
        frac = (p2-(bin2)*dx)*inv_pij
        if frac>=1:
            print("Here3",p1,p2,binid,frac,bin1,bin2)
        if frac<0:
            print("Here3",p1,p2,binid,frac,bin1,bin2)
        subpress[binid,0,0]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[0],dr[0],r)
        subpress[binid,1,1]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[1],dr[1],r)
        subpress[binid,2,2]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[2],dr[2],r)
        subpress[binid,1,0]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[1],dr[0],r)
        subpress[binid,0,1]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[0],dr[1],r)
        subpress[binid,2,0]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[2],dr[0],r)
        subpress[binid,0,2]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[0],dr[2],r)
        subpress[binid,2,1]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[2],dr[1],r)
        subpress[binid,1,2]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[1],dr[2],r)
    else:
        if nint==0:
            binid=bin1%nbin
            frac=1
            subpress[binid,0,0]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[0],dr[0],r)
            subpress[binid,1,1]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[1],dr[1],r)
            subpress[binid,2,2]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[2],dr[2],r)
            subpress[binid,1,0]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[1],dr[0],r)
            subpress[binid,0,1]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[0],dr[1],r)
            subpress[binid,2,0]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[2],dr[0],r)
            subpress[binid,0,2]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[0],dr[2],r)
            subpress[binid,2,1]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[2],dr[1],r)
            subpress[binid,1,2]+=(frac/box[1]/box[2]/dx)*calpres(f,dr[1],dr[2],r)



@numba.njit(cache=True)
def local_pressureWCA(subpress,n,pos,com,dx,box,n_bins,eps,sigma,k,r0,rc_n):
         for i,j in n:

             xitmp = pos[i]-com
             xjtmp = pos[j]-com
             delr = xitmp-xjtmp
             delr[0]=PBC(delr[0],box[0])
             delr[1]=PBC(delr[1],box[1])
             delr[2]=PBC(delr[2],box[2])

             r=np.linalg.norm(delr)
             # if r<rc_n:
             f=WCA(r,sigma,eps,sigma)
             Planecalc(subpress,xjtmp,delr,f,dx,n_bins,r,box)



@numba.njit(cache=True)
def local_pressureFENE(subpress,bonds,pos,com,dx,box,n_bins,eps,sigma,k,r0,rc_n):
    for i,j in bonds:
        xitmp = pos[i]-com
        xjtmp = pos[j]-com
        delr = xitmp-xjtmp
        delr[0]=PBC(delr[0],box[0])
        delr[1]=PBC(delr[1],box[1])
        delr[2]=PBC(delr[2],box[2])
        r=np.linalg.norm(delr)
        f=FENE(r,sigma, r0,rc_n, k)
        Planecalc(subpress,xjtmp,delr,f,dx,n_bins,r,box)





@numba.njit(cache=True)
def kinetic_pressure(subpresKin,pos,comx,n_bins,vel,dx,box,m=1):
    for i in range(len(pos)):

        bin1 =int(int(np.floor((pos[i,0]-comx+box[0]+box[0]/2)/dx))%n_bins)
        subpresKin[bin1,0,0]+=(m*vel[i,0]*vel[i,0])/box[1]/box[2]/dx
        subpresKin[bin1,1,0]+=(m*vel[i,1]*vel[i,0])/box[1]/box[2]/dx
        subpresKin[bin1,2,0]+=(m*vel[i,2]*vel[i,0])/box[1]/box[2]/dx
        subpresKin[bin1,1,1]+=(m*vel[i,1]*vel[i,1])/box[2]/box[1]/dx
        subpresKin[bin1,0,1]+=(m*vel[i,0]*vel[i,1])/box[1]/box[2]/dx
        subpresKin[bin1,0,2]+=(m*vel[i,0]*vel[i,2])/box[1]/box[2]/dx
        subpresKin[bin1,2,2]+=(m*vel[i,2]*vel[i,2])/box[1]/box[2]/dx
        subpresKin[bin1,1,2]+=(m*vel[i,1]*vel[i,2])/box[1]/box[2]/dx
        subpresKin[bin1,2,1]+=(m*vel[i,2]*vel[i,1])/box[1]/box[2]/dx

rc = 1
eps = 1
sigma = 1
k=30
rc_n = math.pow(2/2,1/6)
r0=1.5*sigma
dx=1
cutoff=sigma
samples=2
kbT=1.0
Ntot = sys.argv[1]
fraction = sys.argv[2]
densratio = sys.argv[3]
molefrac = sys.argv[4]


with open("pressure_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt",mode='w+') as f:
    for sample in range(1,samples):
        #filename='trajectory_precompression_sample'+str(sample)+'.gsd'
        #filename='trajectory_Press0_precompression_sample'+str(sample)+'.gsd'
        filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"



        start=time.time()
        traj=gsd.hoomd.open(filename,mode='rb')
        box = traj[0].configuration.box
        box = box[:3]
        n_bins =int( (box[0]/dx))
        bin_x = np.linspace(-box[0]/2,box[0]/2,n_bins+1)
        mid_x=0.5*(bin_x[1:]+bin_x[:-1])
        # print(mid_x.shape)
        # exit()
        Lxby2=box[0]/2
        Lyby2=box[1]/2
        Lzby2=box[2]/2
        cutoff=sigma
        lcutoffx=cutoff
        lengthx = int(Lxby2/cutoff)
        lengthx = int(Lxby2/lcutoffx)
        lcx = Lxby2/lengthx
        shiftx=int(lengthx)
        slcx=shiftx*lcx
        n_particles = traj[0].particles.N
        bonds = traj[0].bonds.group
        bonds = np.sort(bonds,axis=1)
        bonds=np.unique(bonds,axis=0)
############################################################################
        box1 = freud.box.Box(box[0],box[1],box[2])
        init = 800
        subpres=np.zeros((len(traj)-init,n_bins,3,3))
        for frame in range(init,len(traj)):
            print(frame,len(traj),flush=True)
            start=time.time()
    ####Selecting Particles position image and velocities##########
            positions = traj[frame].particles.position
            com = np.nanmean(positions,axis=0)
            # bins =(np.floor((positions[:,0]-com[0]+box[0]+box[0]/2)/dx).astype(int)%n_bins).astype(int)
            # print(np.sort(bins))
            # print(np.where(bins<0)[0])
            # print(np.where(bins>n_bins)[0])
            # print(n_bins)
            # exit()




            image =traj[frame].particles.image
            #positions=box1.unwrap(positions,image)

            vel =traj[frame].particles.velocity
    ###############################################################
    ###########Neighborlist calculation for WCA non bondex########
    ###############################################################
            aq = freud.locality.AABBQuery(box1, positions)
            nlist = aq.query(positions, {'r_max':1.0,'r_min':.05}).toNeighborList()
            nlist=np.asarray(nlist,dtype=np.int32)
            nlist=np.sort(nlist,axis=1)
            nlist=np.unique(nlist,axis=0)
    ################################################################3######
        ########Selecting particles which are not bonded out of these#######
    ########################################################################
            av = nlist.view([('', nlist.dtype)] * nlist.shape[1])
            bv = bonds.view([('', bonds.dtype)] * bonds.shape[1])
            #print(av.dtype)
            #print(bv.dtype)
            # Find the set difference using np.setdiff1d
            result = np.setdiff1d(av, bv).view(nlist.dtype).reshape(-1, nlist.shape[1])


    ##########################################################################


            subpressWCA=np.zeros((n_bins,3,3),dtype=float)#####Nonbonded interactions WCA###########
            subpressFENE=np.zeros((n_bins,3,3),dtype=float)###########FENE+WCA part of pressure contributed by bonds#
            subpresKin=np.zeros((n_bins,3,3),dtype=float) #########Kintetic Part of pressure###########



            kinetic_pressure(subpresKin,positions,com[0],n_bins,vel,dx,box)####Call to kinetic part of prssure calc##########

            local_pressureWCA(subpressWCA,result,positions,com,dx,box,n_bins,eps,sigma,k,r0,rc_n)####Call to Non bonded part (WCA) of pressure calc##########

            local_pressureFENE(subpressFENE,bonds,positions,com,dx,box,n_bins,eps,sigma,k,r0,rc_n)####Call to bonded part (FENE)of pressure calc##########
            subpres[frame-init]=subpressFENE+subpressWCA+subpresKin
            # for i in range(len(subpressFENE)):
            # #        #print(subpressWCA[i]+subpressFENE[i])
            #         if i>300 and i<600:
            #             print(subpressWCA[i]+subpressFENE[i])

            end=time.time()
            print(end-start,flush=True)

        traj.close()
        subP = np.nanmean(subpres,axis=0)
        print(subP.shape,mid_x.shape,flush=True)
        finalsubP=np.asarray([mid_x.T,subP[:,0,0].T,subP[:,1,1].T,subP[:,2,2].T,subP[:,0,1].T,subP[:,0,2].T,subP[:,1,2].T]).T
        print(finalsubP.shape,flush=True)
        np.savetxt(f,finalsubP)
        f.write('\n')


#exit()
