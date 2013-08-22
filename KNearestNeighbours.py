####################################################################################################################################################
#Copyright (c) 2013, Josiah Walker
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or #other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED #WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY #DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS #OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING #NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
####################################################################################################################################################
"""
GPU based K nearest neighbours algorithm.
"""

import time
from numpy import array,zeros,amax,amin,sqrt,dot,random
import numpy
from numpy.linalg import eig
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import math


from DataUtils import dataConfig,loadTable,loadSplitTable,loadMatrix,loadIntMatrix,saveTable

KernelLocation = "CudaKernels/KNN/"

# KNN Algorithm --------------------------------------------

def KNNConfig(dataTable,srcDims, k, eps = 1000000000.,gpuMemSize = 512, settings = {}):
    """
    Creates all the memory/data settings to run GPU accelerated KNN.
    """
    settings = dataConfig(dataTable,settings)
    settings["sourceDims"] = min(settings["sourceDims"],srcDims)
    
    #XXX: determine memory and thread sizes from device
    settings["memSize"] = gpuMemSize*1024*1024
    settings["maxThreads"] = 1024
    
    #set up chunk sizes
    memoryPerElement = k*4*2*settings["dataLength"] + (settings["sourceDims"]*4)*2 + 20*4 #this is an estimated memory used per element
    settings["chunkSize"] = min(int(math.ceil(float(settings["memSize"])/memoryPerElement)),settings["dataLength"])
    settings["lastChunkSize"] = ((settings["dataLength"]-1) % settings["chunkSize"]) + 1
    #if settings["lastChunkSize"] == 0:
    #    settings["lastChunkSize"] = settings["chunkSize"]
    
    #create kernel gridsize tuples
    settings["block"] = (settings["maxThreads"],1,1)
    settings["grid"] = (max(int(math.ceil(float(settings["chunkSize"])/settings["maxThreads"])),1),1,1)
    
    #precalculate all constant kernel params
    settings["dimensions"] = numpy.int64(settings["sourceDims"])
    settings["k"] = numpy.int64(k)
    settings["eps"] = numpy.float32(eps)
    settings["dataSize"] = numpy.int64(settings["dataLength"])
    settings["chunkSize"] = numpy.int64(settings["chunkSize"])
    settings["maxThreads"] = numpy.int64(settings["maxThreads"])
    
    
    
    return settings

def KNN(dataTable, k, epsilon, srcDims = 1000000000000000, normData = True):
    """
    Get a k,epsilon version k nearest neighbours
    """
    #load up the configuration
    knnOptions = KNNConfig(dataTable,srcDims,k,epsilon)
    
    
    #load and format the table for use.
    data = loadTable(dataTable,knnOptions)
    
    #check if we should normalise the data (this is really quick and dirty, replace it with something better)
    if normData:
        dmax = max([amax(d) for d in data])
        dmin = max([amin(d) for d in data])
        data = [(d-dmin)/(dmax-dmin+0.00000001) for d in data]
    
    #create the CUDA kernels
    program = SourceModule(open(KernelLocation+"KNN.nvcc").read())
    prg = program.get_function("KNN")
    t0 = time.time()
    
    #make a default distance list
    distances0 = (zeros((knnOptions['chunkSize']*knnOptions['k'])) + knnOptions['eps']).astype(numpy.float32)
    indices0 = zeros((knnOptions['chunkSize']*knnOptions['k'])).astype(numpy.uint32)
    dists = [distances0.copy() for i in xrange(len(data))]
    indices = [indices0.copy() for i in xrange(len(data))]
    
    #calculate KNN
    offset = 0
    source_gpu = drv.mem_alloc(data[0].nbytes)
    indices_gpu = drv.mem_alloc(indices[0].nbytes)
    dists_gpu = drv.mem_alloc(dists[0].nbytes)
    for source in data:
        drv.memcpy_htod(source_gpu, source)
        drv.memcpy_htod(indices_gpu, indices[offset])
        drv.memcpy_htod(dists_gpu, dists[offset])
        for t in xrange(len(data)):
            prg(source_gpu,
                drv.In(data[t]),
                knnOptions["dimensions"],
                indices_gpu,
                dists_gpu,
                knnOptions['k'],
                knnOptions['eps'],
                knnOptions['dataSize'],
                knnOptions['chunkSize'],
                numpy.int64(offset),
                numpy.int64(t),
                knnOptions['maxThreads'],
                block=knnOptions['block'],
                grid=knnOptions['grid'])
        drv.memcpy_dtoh(indices[offset], indices_gpu)
        drv.memcpy_dtoh(dists[offset], dists_gpu)
        offset += 1
    del source_gpu
    del indices_gpu
    del dists_gpu
    
    #organise data and add neighbours
    alldists = numpy.concatenate(dists).reshape((-1,knnOptions['k']))[:(knnOptions['dataSize'])].tolist()
    allindices = numpy.concatenate(indices).reshape((-1,knnOptions['k']))[:(knnOptions['dataSize'])].tolist()
    for i in xrange(len(alldists)): #remove excess entries
        if knnOptions['eps'] in alldists[i]:
            ind = alldists[i].index(knnOptions['eps'])
            alldists[i] = alldists[i][:ind]
            allindices[i] = allindices[i][:ind]
    for i in xrange(len(alldists)):
            j = 0
            for ind in allindices[i]: #add mirrored entries
                if not (i in allindices[ind]):
                    allindices[ind].append(i)
                    alldists[ind].append(alldists[i][j])
                j += 1
    maxKValues = max([len(p) for p in allindices]) 
    #print maxKValues
    for i in xrange(len(alldists)): #pad all entries to the same length for the next algorithm
        if len(alldists[i]) < maxKValues:
            alldists[i].extend( [knnOptions['eps']]*(maxKValues-len(alldists[i])) )
            allindices[i].extend( [0]*(maxKValues-len(allindices[i])) )
    
    print time.time()-t0, " seconds to process KNN"
    """
    #save the link matrix to compare to the matlab results
    f = open("linkmatrix.csv",'w')
    mat = [[0]*len(allindices) for i in xrange(len(allindices))]
    for i in xrange(len(allindices)):
        for j in xrange(len(allindices[i])):
            if alldists[i][j] < knnOptions['eps']:
                mat[i][allindices[i][j]] = mat[allindices[i][j]][i] = 1
    for m in mat:
        f.write(str(m).strip('[]')+'\n')
    f.close()
    """

    return [allindices[i]+alldists[i] for i in xrange(len(alldists))]


