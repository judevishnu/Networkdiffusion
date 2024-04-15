#!/usr/bin/python3
import hoomd
import hoomd.md
import gsd
import gsd.hoomd
import time  as time1
import numpy
import random 
import math
import freud
import sys
from scipy import interpolate
from sympy import symbols, Eq, solve
from scipy.signal import savgol_filter
#from scipy.interpolate import CubicSpline
import statsmodels.api as sm
#######################################################################
##  Equation of a line
######################################################################
def findintercept(slope,xpoint,ypoint):
    c = ypoint-slope*xpoint
    return c
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
###############################################################################
dt = 0.001

#samples=int(sys.argv[1])
Ntot = sys.argv[1]
fraction=sys.argv[2]
densratio = sys.argv[3]
molefrac  = sys.argv[4]
kbT = 1.0
time_list=[]
period = 8e5

sample=1

filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"



traj=gsd.hoomd.open(name=filename, mode='rb')
print(len(traj))

Lx=traj[0].configuration.box[0]
Ly=traj[0].configuration.box[1]
Lz=traj[0].configuration.box[2]
Lxby2 = Lx/2.0 
Lyby2 = Ly/2.0 
Lzby2 = Lz/2.0 
Ny = int(Ly/sigma[0][0])
poly_len = int(Lz/sigma[0][0])
print(poly_len)
typeid =  numpy.copy(traj[0].particles.typeid)
N_remainz = numpy.count_nonzero(typeid == 1)
N_constraint = numpy.count_nonzero(typeid==0)
Nx_remainz = int(N_remainz/2/Ny/poly_len)

cutoffx = sigma[0][0]
lcutoffx =cutoffx
lengthx = int(Lxby2/lcutoffx)
lcx = Lxby2/lengthx
shiftx = int(lengthx)
cellnox= 2*lengthx


step=5
start=1100
bin_val =numpy.zeros(cellnox,dtype=float)
for x in range(cellnox):
    bin_val[x] = (x+0.5)*lcutoffx-Lxby2
    print(len(traj),Lxby2)

nbin_array = (1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
#nbin_array = (1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 )

begini = time1.perf_counter()
#for nbin in nbin_array[:]:

data_box = numpy.empty((0,4),dtype=float)
for nbins in nbin_array:
    
    nbin = nbins
    cutoffz = float(Lzby2/nbin)
    lcutoffz =cutoffz
    lengthz = int(Lzby2/lcutoffz)
    print(lengthz)
    lcz = Lzby2/lengthz
    shiftz = int(lengthz)
    cellnoz= 2*lengthz

    cutoffy = cutoffz
    lcutoffy =cutoffy
    lengthy = int(Lyby2/lcutoffy)

    print(lengthy)
    lcy = Lyby2/lengthy
    shifty = int(lengthy)
    cellnoy= 2*lengthy

    totalcel = cellnox*cellnoy*cellnoz
    slcx = shiftx*lcx
    slcy = shifty*lcy
    slcz = shiftz*lcz
    print("sub_Lx,sub_Ly,sub_Lz",cutoffx,cutoffy,cutoffz)
    print("Lx,Ly,Lz",Lx,Ly,Lz)

    print("cellno_z,cellno_y,cellno_x",cellnoz,cellnoy,cellnox)
    delV = cutoffy*cutoffz*cutoffx
    datasample = numpy.empty((0,1),dtype=float)
    for sample in range(1,samples):
        filename = "Combinedtrajectory"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
        print(filename)
        traj=gsd.hoomd.open(name=filename, mode='rb')
        Lx=traj[0].configuration.box[0]
        Ly=traj[0].configuration.box[1]
        Lz=traj[0].configuration.box[2]
        print(Lx,Ly,Lz)
        data2 = numpy.empty((0,1),dtype=float)
        for frame in range(start,len(traj),step):
        
            time = frame*dt*period
            posA=traj[frame].particles.position[:N_constraint]
            posB=traj[frame].particles.position[N_constraint:N_constraint+N_remainz]
            pos=traj[frame].particles.position
            #selA = numpy.where(((-Lz/2<=posA[:,1])&(posA[:,1]<=Lz/2)))[0]
            #selB = numpy.where(((-Lz/2<=posB[:,1])&(posB[:,1]<=Lz/2)))[0]

            selA = numpy.where(((-Ly/2<=posA[:,1])&(posA[:,1]<=Ly/2)))[0]
            selB = numpy.where(((-Ly/2<=posB[:,1])&(posB[:,1]<=Ly/2)))[0]

            image=traj[frame].particles.image
            posA = posA[selA]
            posB = posB[selB]
        
            yiB = ((slcy +posB[:,1])/lcy).astype(int)
            yiB=numpy.where(yiB<0,yiB+1,yiB)
            yiB=numpy.where(yiB==cellnoy,yiB-1,yiB)
            yiA = ((slcy +posA[:,1])/lcy).astype(int)
            yiA=numpy.where(yiA<0,yiA+1,yiA)
            yiA=numpy.where(yiA==cellnoy,yiA-1,yiA)


            ziA = ((slcz +posA[:,2])/lcz).astype(int)
            ziA=numpy.where(ziA<0,ziA+1,ziA)
            ziA=numpy.where(ziA==cellnoz,ziA-1,ziA)
            ziB = ((slcz +posB[:,2])/lcz).astype(int)
            ziB=numpy.where(ziB<0,ziB+1,ziB)
            ziB=numpy.where(ziB==cellnoz,ziB-1,ziB)
        
            subboxidA = ziA+yiA*cellnoz
            subboxidB = ziB+yiB*cellnoz

        
            data1 = numpy.empty(0,dtype=float)
            for subbox in range(cellnoy*cellnoz):
                subposidA = numpy.where(subboxidA==subbox)[0]
                subposidB = numpy.where(subboxidB==subbox)[0]
                subboxposB = numpy.take(posB,subposidB,axis=0)
                subboxposA = numpy.take(posA,subposidA,axis=0)
                xiB = ((slcx +subboxposB[:,0])/lcx).astype(int)
                xiB=numpy.where(xiB<0,xiB+1,xiB)
                xiB=numpy.where(xiB==cellnox,xiB-1,xiB)
        
                xiA = ((slcx +subboxposA[:,0])/lcx).astype(int)
                xiA=numpy.where(xiA<0,xiA+1,xiA)
                xiA=numpy.where(xiA==cellnox,xiA-1,xiA)
        
                mA = xiA[:]
                mB = xiB[:]
                       
                count_A=numpy.bincount(mA,minlength = len(bin_val))
                count_B=numpy.bincount(mB,minlength = len(bin_val))
        
                densityA = count_A[:]/delV
                densityB = count_B[:]/delV
                data = numpy.asarray((bin_val,densityA,densityB)).T
                
                poly1st  = sm.nonparametric.lowess(data[:len(data)//2,2],data[:len(data)//2,0], frac=0.1).T
                gel1st = sm.nonparametric.lowess(data[:len(data)//2,1],data[:len(data)//2,0],frac=0.1).T
                poly2nd = sm.nonparametric.lowess(data[len(data)//2:,2],data[len(data)//2:,0],frac=0.1).T
                gel2nd = sm.nonparametric.lowess(data[len(data)//2:,1],data[len(data)//2:,0],frac=0.1).T
                
                
                derivativepoly1 = numpy.gradient(poly1st[1],poly1st[0])
                derivativegel1 =  numpy.gradient(gel1st[1],gel1st[0])
                derivativepoly2 = numpy.gradient(poly2nd[1],poly2nd[0])

                derivativegel2 =  numpy.gradient(gel2nd[1],gel2nd[0])

                
                
                maxpoly1 = numpy.max(poly1st[1,:len(data)//2])
                minpoly1 = numpy.min(poly1st[1,:len(data)//2])
                maxgel1 = numpy.max(gel1st[1,:len(data)//2])
                mingel1 = numpy.min(gel1st[1,:len(data)//2])
            
                maxpoly2 = numpy.max(poly2nd[1,:len(data)//2])
                minpoly2 = numpy.min(poly2nd[1,:len(data)//2])
                maxgel2 = numpy.max(gel2nd[1,:len(data)//2])
                mingel2 = numpy.min(gel2nd[1,:len(data)//2])
                
                #print(maxpoly1,maxpoly2,maxgel1,maxgel2)
                #exit()

                maxpoly1deri = numpy.max(numpy.abs(derivativepoly1))
                maxpoly2deri = numpy.max(numpy.abs(derivativepoly2))
                maxgel1deri = numpy.max(numpy.abs(derivativegel1))
                maxgel2deri = numpy.max(numpy.abs(derivativegel2))
                #print(abs(maxpoly1) , abs(minpoly1),abs(maxpoly2) , abs(minpoly2))
                subwpoly1 = (abs(maxpoly1) - abs(minpoly1))/maxpoly1deri
                subwpoly2 = (abs(maxpoly2) - abs(minpoly2))/maxpoly2deri
                subwgel1 = (abs(maxgel1) - abs(mingel1))/maxgel1deri
                subwgel2 = (abs(maxgel2) -abs( mingel2))/maxgel2deri
                #print(subwgel1,subwgel2,subwpoly1,subwpoly2)
                width = ((abs(subwpoly1)+abs(subwgel1))/2+(abs(subwpoly2)+abs(subwgel2))/2)/2
                data1 = numpy.append(data1,width)

            #print(data1)
            widthtime =numpy.nanmean(data1*data1,axis=0)
            #print(data1.shape)
            
            data2 = numpy.append(data2,[[widthtime]],axis=0)     
            #print(data2)
        datasample=numpy.append(datasample,[numpy.nanmean(data2,axis=0)],axis=0)
        print(datasample)

    data_box = numpy.append(data_box ,numpy.asarray([[nbins,lcz,numpy.nanmean(datasample,axis=0)[0],numpy.nanstd(datasample,axis=0)[0]]]),axis=0)
    print(data_box)

filename3 = "Subbox_density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"



with open(filename3, 'w+') as f3:                                                                                                                              
    numpy.savetxt(f3, data_box)
                                                                                                                                                                                               
end = time1.perf_counter()
print(end-begini)
################################################
