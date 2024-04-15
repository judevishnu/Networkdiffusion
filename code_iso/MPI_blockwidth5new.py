#!/usr/bin/python3
#import hoomd
#import hoomd.md
import gsd
import gsd.hoomd
import time  as time1
import numpy
import random 
import math
import freud
import sys
from scipy import interpolate
#from sympy import symbols, Eq, solve
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import derivative
#import mpi4py
#mpi4py.rc.initialize = False
from mpi4py import MPI
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
print(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
#samples=int(sys.argv[1])
sample = int(sys.argv[1])
Ntot = sys.argv[2]
fraction=sys.argv[3]
densratio = sys.argv[4]
molefrac  = sys.argv[5]
kbT = 1.0
time_list=[]
period = 8e5

#sample=1

filename = "trajectoryContinueIso"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"



traj=gsd.hoomd.open(name=filename, mode='rb')
print(len(traj))

Lx=traj[0].configuration.box[0]
Ly=traj[0].configuration.box[1]
Lz=traj[0].configuration.box[2]
Lxby2 = Lx/2.0 
Lyby2 = Ly/2.0 
Lzby2 = Lz/2.0 
Ny = int(Ly/sigma[0][0])
poly_len = 102
print(poly_len)
typeid =  numpy.copy(traj[0].particles.typeid)
N_remainz = numpy.count_nonzero(typeid == 1)
N_constraint = numpy.count_nonzero(typeid==0)
Nx_remainz = int(N_remainz/2/Ny/poly_len)

cutoffx = 1*sigma[0][0]
lcutoffx =cutoffx
lengthx = int(Lxby2/lcutoffx)
lcx = Lxby2/lengthx
shiftx = int(lengthx)
cellnox= 2*lengthx


step=2
start=1000
bin_val =numpy.zeros(cellnox,dtype=float)
for x in range(cellnox):
    bin_val[x] = (x+0.5)*lcutoffx-Lxby2
    #print(len(traj),Lxby2)

nbin_array = (1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
#nbin_array = (1 , 2 , 3  )

#begini = time1.perf_counter()
#for nbin in nbin_array[:]:


#subp=np.zeros((1+int((end_index-start_index)/step),n_bins,3,3))
filename3 = "Subbox_density_vs_xboxnewer"+str(sample)+"_Isodiamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"
with open(filename3, 'a+') as f3:                                                                                                                              
        #numpy.savetxt(f3, data_box)

    filename = "trajectoryContinueIso"+str(sample)+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".gsd"
    
    traj=gsd.hoomd.open(name=filename, mode='rb')
    Lx=traj[0].configuration.box[0]
    Ly=traj[0].configuration.box[1]
    Lz=traj[0].configuration.box[2]

    data_nbin = numpy.zeros((len(nbin_array),3),dtype=float)
    #MPI.Init()

    begini = time1.perf_counter()
    for k in range(0,len(nbin_array)):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        nbin = nbin_array[k]
        cutoffz = float(Lzby2/nbin)
        lcutoffz =cutoffz
        lengthz = int(Lzby2/lcutoffz)
        #print(lengthz)
        lcz = Lzby2/lengthz
        shiftz = int(lengthz)
        cellnoz= 2*lengthz

        cutoffy = cutoffz
        lcutoffy =cutoffy
        lengthy = int(Lyby2/lcutoffy)

        #print(lengthy)
        lcy = Lyby2/lengthy
        shifty = int(lengthy)
        cellnoy= 2*lengthy

        totalcel = cellnox*cellnoy*cellnoz
        slcx = shiftx*lcx
        slcy = shifty*lcy
        slcz = shiftz*lcz

        #print("sample :",sample,"nbin :",nbin,"rank:",rank,filename)
        #print("rank : sub_Lx,sub_Ly,sub_Lz",rank,cutoffx,cutoffy,cutoffz)
        #print("rank : Lx,Ly,Lz",rank,Lx,Ly,Lz)
        print("rank : cellno_z,cellno_y,cellno_x",rank,cellnoz,cellnoy,cellnox,flush=True)
        delV = cutoffy*cutoffz*cutoffx
        #data_box = numpy.empty((len(nbin_array),4),dtype=float)
        elements_per_process = (len(traj)-start)//size
        start_index = rank*elements_per_process
        end_index = (rank+1)*elements_per_process
        if rank == size-1:
            end_index = start_index +len(traj) -(start_index+start)
            print("rank :",rank,"size-1 :",size-1,"start_index :",start_index,"end_index :",end_index,flush=True)

        data2 = numpy.zeros((int((end_index-start_index)/step),1),dtype=float)
        for frame in range(start+start_index,start+end_index,step):
            print("sample :",sample,"rank :",rank,"nbin :",nbin,"frame : ",frame,"int((frame-(start+start_index))/step) :",int((frame-(start+start_index))/step),"start_index            :",start_index,"end_index :",end_index,flush=True)
            if int((frame-(start+start_index))/step) >=len(data2):
                break
            time = frame*dt*period
            posA=traj[frame].particles.position[:N_constraint]
            posB=traj[frame].particles.position[N_constraint:N_constraint+N_remainz]
            #pos=traj[frame].particles.position
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

            #print("cellnoz cellnoy",cellnoz,cellnoy)
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
                
                poly1st  = sm.nonparametric.lowess(data[:len(data)//2,2],data[:len(data)//2,0], frac=0.08).T
                gel1st = sm.nonparametric.lowess(data[:len(data)//2,1],data[:len(data)//2,0],frac=0.08).T
                poly2nd = sm.nonparametric.lowess(data[len(data)//2:,2],data[len(data)//2:,0],frac=0.08).T
                gel2nd = sm.nonparametric.lowess(data[len(data)//2:,1],data[len(data)//2:,0],frac=0.08).T
                
                                
                derivativepoly1 = numpy.gradient(poly1st[1],poly1st[0])
                derivativegel1 =  numpy.gradient(gel1st[1],gel1st[0])
                derivativepoly2 = numpy.gradient(poly2nd[1],poly2nd[0])
                derivativegel2 =  numpy.gradient(gel2nd[1],gel2nd[0])
                sg = derivative.SavitzkyGolay(left=3, right=3, order=3, periodic=False)
                #sg = derivative.FiniteDifference(5)
                derivativepoly1  = sg.d(poly1st[1],poly1st[0])
                derivativegel1  = sg.d(gel1st[1],gel1st[0])
                derivativepoly2  = sg.d(poly2nd[1],poly2nd[0])
                derivativegel2  = sg.d(gel2nd[1],gel2nd[0])
                #print(checkderivpoly1.shape)
                #exit()
                #print(derivativegel2)
                #print(derivativepoly2)
                
                maxpoly1 = numpy.max(poly1st[1,:])
                minpoly1 = numpy.min(poly1st[1,:])
                maxgel1 = numpy.max(gel1st[1,:])
                mingel1 = numpy.min(gel1st[1,:])
            
                maxpoly2 = numpy.max(poly2nd[1,:])
                minpoly2 = numpy.min(poly2nd[1,:])
                maxgel2 = numpy.max(gel2nd[1,:])
                mingel2 = numpy.min(gel2nd[1,:])
                #print(maxpoly1,maxpoly2,maxgel1,maxgel2)
                #exit()

                maxpoly1deri = numpy.max(numpy.abs(derivativepoly1))
                maxpoly2deri = numpy.max(numpy.abs(derivativepoly2))
                maxgel1deri = numpy.max(numpy.abs(derivativegel1))
                maxgel2deri = numpy.max(numpy.abs(derivativegel2))
                #print(maxpoly1deri)
                #exit()
                subwpoly1 = (abs(maxpoly1) - abs(minpoly1))/maxpoly1deri
                subwpoly2 = (abs(maxpoly2) - abs(minpoly2))/maxpoly2deri
                subwgel1 = (abs(maxgel1) - abs(mingel1))/maxgel1deri
                subwgel2 = (abs(maxgel2) -abs( mingel2))/maxgel2deri
                #print(subwgel1,subwgel2,subwpoly1,subwpoly2)
                width1 = numpy.nanmean(numpy.asarray([abs(subwpoly1),abs(subwgel1)]))
                width2 = numpy.nanmean(numpy.asarray([abs(subwpoly2),abs(subwgel2)]))
                width = numpy.nanmean(numpy.asarray([width1,width2]))
                #print("width :",width)
                data1 = numpy.append(data1,width)

                #print(data1)
                #comm.Barrier()
            widthtime =numpy.nanmean(data1*data1,axis=0)
                #print(data1.shape)
            
            data2[int((frame -(start+start_index))/step)] = widthtime

                #print(data2)
        all_data2 = comm.gather(data2,root=0)
        if rank==0:
            print("Iam here sample :",sample,"nbin :",nbin,flush=True)
            datatime_concatmean=numpy.concatenate(all_data2,axis=0)
                #print(datatime_concatmean)
            tmp = numpy.nanmean(datatime_concatmean,axis=0)[0]
            print(tmp)
            data_nbin[k,0] = nbin
            data_nbin[k,1] = cutoffz
            data_nbin[k,2] = tmp
            findata  = data_nbin[k]
            numpy.savetxt(f3,findata.reshape(-1,3))

    #comm.Barrier()
    #if rank==0:
        #    print("sample :",sample,"rank : ",rank)
    #    print(data_nbin,flush=True)
    #    numpy.savetxt(f3, data_nbin)
    #    f3.write("\n")
    #    end = time1.time()
    #     print("time: ",end-begini)
if rank==0:
    end = time1.time()
    print("time: ",end-begini)

MPI.Finalize()
################################################
