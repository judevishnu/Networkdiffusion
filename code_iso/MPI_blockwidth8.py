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
#import mpi4py
#mpi4py.rc.initialize = False
from mpi4py import MPI
#######################################################################
##  Equation of a line
######################################################################
def findintercept(slope,xpoint,ypoint):
    c = ypoint-slope*xpoint
    return c

def open_filetoarray(filen):
    with open(filen, "r") as file:
        data_arrays = []  # Initialize a list to store the data arrays
        current_array_lines = []  # Initialize a list to store lines of the current array

    # Read the file line by line
        for line in file:
        # Check if the line is not empty (contains data)
            if line.strip():
                current_array_lines.append(line)
            else:
            # If the line is empty (blank line), it's the end of the current array
                if current_array_lines:
                    current_array = numpy.genfromtxt(current_array_lines)
                    data_arrays.append(current_array)
                    current_array_lines = []

    # Append the last array (if any)
        if current_array_lines:
            current_array = numpy.genfromtxt(current_array_lines)
            data_arrays.append(current_array)
    dataArray = numpy.asarray(data_arrays)

    return dataArray
########################################################################
#           Set parameters
#######################################################################
samples=11
#samples=101
sample=1
dia=[]
dia.append(1.0)
dia.append(1.0)
sigma = [] # interaction parameter sigma 

sigma.append([(dia[0]+dia[0])/2.,(dia[0]+dia[1])/2.])
###############################################################################
dt = 0.001
print(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
#samples=int(sys.argv[1])
#sample = int(sys.argv[1])
Ntot = sys.argv[1]
fraction=sys.argv[2]
densratio = sys.argv[3]
molefrac  = sys.argv[4]
kbT = 1.0
step=1
nbin_array = (1.0 , 2.0 , 3.0 , 4.0 , 5.0 , 6.0 , 7.0 , 8.0 , 9.0 , 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0)#, 19.0, 20.0)
blength1 = ("51.00450", 25.50225, "17.00150", 12.75113, "10.20090",8.50075,7.28636, 6.37556, 5.66717, 5.10045, 4.63677,4.25038, 3.92342, 3.64318, "3.40030", 3.18778, 3.00026,2.83358)#, 2.68445, 2.55023);
#begini = time1.perf_counter()
#for nbin in nbin_array[:]:


#subp=np.zeros((1+int((end_index-start_index)/step),n_bins,3,3))
filename3 = "subbox_density_vs_xboxtryquant"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_kT_"+str(kbT)+".txt"
with open(filename3, 'w+') as f3:                                                                                                                              
    
    
    data_nbin = numpy.zeros((len(nbin_array),4),dtype=float)
    begini = time1.perf_counter()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    
    for k in range(0,len(nbin_array)):

        nbin=nbin_array[k]
        sbox =float(blength1[k])
        
        filename = "subbox_density_vs_xbox"+"_diamond_network_diffusive"+"_Ntot_"+str(Ntot)+"_vol_frac"+str(fraction)+"_densratio_"+str(densratio)+"_molfrac_"+str(molefrac)+"_nbin_"+str(nbin)+"_kT_"+str(kbT)+"_boxwidth_"+str(blength1[k])+".txt"
        dens = open_filetoarray(filename)
        print(dens.shape)
    
        Length = dens.shape[0]
        start =0
        #data_nbin = numpy.empty((len(nbin_array),4),dtype=float)
        elements_per_process = (Length-start)//size
        start_index = rank*elements_per_process
        end_index = (rank+1)*elements_per_process
        if rank == size-1:
            end_index = start_index +Length -(start_index+start)
            print("rank :",rank,"size-1 :",size-1,"start_index :",start_index,"end_index :",end_index,flush=True)

        data2 = numpy.zeros((int((end_index-start_index)/step),1),dtype=float)
        for subbox in range(start+start_index,start+end_index,step):
            print("sample :",sample,"rank :",rank,"nbin :",nbin,"subbox : ",subbox,"int((subbox-(start+start_index))/step) :",int((subbox-(start+start_index))/step),"start_index :",start_index,"end_index :",end_index,flush=True)
            if int((subbox-(start+start_index))/step) >=len(data2):
                break
                    
            densityA = dens[subbox,:,1]
            densityB = dens[subbox,:,2]
            bin_val = dens[subbox,:,0]
            data = numpy.asarray((bin_val,densityA,densityB)).T
                
            poly1st  = sm.nonparametric.lowess(data[:len(data)//2,2],data[:len(data)//2,0], frac=0.01).T
            gel1st = sm.nonparametric.lowess(data[:len(data)//2,1],data[:len(data)//2,0],frac=0.01).T
            poly2nd = sm.nonparametric.lowess(data[len(data)//2:,2],data[len(data)//2:,0],frac=0.01).T
            gel2nd = sm.nonparametric.lowess(data[len(data)//2:,1],data[len(data)//2:,0],frac=0.01).T
            
            #poly1stf  = sm.QuantReg(data[:len(data)//2,2],sm.add_constant(data[:len(data)//2,0])).fit(q=0.05,max_iter=1000)
            #gel1stf = sm.QuantReg(data[:len(data)//2,1],sm.add_constant(data[:len(data)//2,0])).fit(q=0.05,max_iter=1000)
            #poly2ndf = sm.QuantReg(data[len(data)//2:,2],sm.add_constant(data[len(data)//2:,0])).fit(q=0.05,max_iter=1000)
            #gel2ndf = sm.QuantReg(data[len(data)//2:,1],sm.add_constant(data[len(data)//2:,0])).fit(q=0.05,max_iter=1000)
                
                
            #poly1st=numpy.asarray((data[:len(data)//2,0].T,poly1stf.predict(sm.add_constant(data[:len(data)//2,0])).T)).T
            #gel1st=numpy.asarray((data[:len(data)//2,0].T,gel1stf.predict(sm.add_constant(data[:len(data)//2,0])).T)).T
            #poly2nd=numpy.asarray((data[len(data)//2:,0].T,poly2ndf.predict(sm.add_constant(data[len(data)//2:,0])).T)).T
            #gel2nd=numpy.asarray((data[len(data)//2:,0].T,gel2ndf.predict(sm.add_constant(data[len(data)//2:,0])).T)).T

            #poly1st  = numpy.asarray((data[:len(data)//2,0].T,data[:len(data)//2,2].T))
            #gel1st = numpy.asarray((data[:len(data)//2,0].T,data[:len(data)//2,1].T))
            #poly2nd = numpy.asarray((data[len(data)//2:,0].T,data[len(data)//2:,2].T))
            #gel2nd = numpy.asarray((data[len(data)//2:,0].T,data[len(data)//2:,1].T))
   
                
            derivativepoly1 = numpy.gradient(poly1st[1],poly1st[0])
            derivativegel1 =  numpy.gradient(gel1st[1],gel1st[0])
            derivativepoly2 = numpy.gradient(poly2nd[1],poly2nd[0])
            derivativegel2 =  numpy.gradient(gel2nd[1],gel2nd[0])

                               
            maxpoly1 = numpy.max(poly1st[1,:])
            minpoly1 = numpy.min(poly1st[1,:])
            maxgel1 = numpy.max(gel1st[1,:])
            mingel1 = numpy.min(gel1st[1,:])
            
            maxpoly2 = numpy.max(poly2nd[1,:])
            minpoly2 = numpy.min(poly2nd[1,:])
            maxgel2 = numpy.max(gel2nd[1,:])
            mingel2 = numpy.min(gel2nd[1,:])
                

            maxpoly1deri = numpy.max(numpy.abs(derivativepoly1))
            maxpoly2deri = numpy.max(numpy.abs(derivativepoly2))
            maxgel1deri = numpy.max(numpy.abs(derivativegel1))
            maxgel2deri = numpy.max(numpy.abs(derivativegel2))
                
            subwpoly1 = (abs(maxpoly1) - abs(minpoly1))/abs(maxpoly1deri)
            subwpoly2 = (abs(maxpoly2) - abs(minpoly2))/abs(maxpoly2deri)
            subwgel1 = (abs(maxgel1) - abs(mingel1))/abs(maxgel1deri)
            subwgel2 = (abs(maxgel2) -abs( mingel2))/abs(maxgel2deri)
            
            width1 = numpy.nanmean(numpy.asarray([abs(subwpoly1),abs(subwgel1)]))
            width2 = numpy.nanmean(numpy.asarray([abs(subwpoly2),abs(subwgel2)]))
            width = numpy.nanmean(numpy.asarray([width1,width2]))
                       
            data2[int((subbox -(start+start_index))/step)] = width*width

                #print(data2)
        all_data2 = comm.gather(data2,root=0)
        if rank==0:
            print("sample :",sample,"nbin :",nbin,flush=True)
            data_concatmean=numpy.concatenate(all_data2,axis=0)
                #print(datatime_concatmean)
            tmp = numpy.nanmean(data_concatmean,axis=0)[0]
            std = numpy.nanstd(data_concatmean,axis=0)[0]
            print(tmp)
            data_nbin[k,0] = nbin
            data_nbin[k,1] = sbox
            data_nbin[k,2] = tmp
            data_nbin[k,3] = std
                

    comm.Barrier()
    if rank==0:
        #    print("sample :",sample,"rank : ",rank)
        print(data_nbin,flush=True)
        numpy.savetxt(f3, data_nbin)
        end = time1.time()
        print("time: ",end-begini)

MPI.Finalize()
################################################
