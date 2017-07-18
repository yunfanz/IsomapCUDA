import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq,kmeans,whiten,kmeans2
from sklearn import cluster as skcluster
import sys,os
import matplotlib.cm as cmx
import matplotlib.colors as colors
from bisect import bisect_left
from heapq import nsmallest
from decimal import *
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from KNearestNeighbours import *
from KMeans import KMeans
import matplotlib
import seaborn as sns
import pandas as pd
from function_utils import savitzky_golay
from astropy.coordinates import SkyCoord
def get_cmap(N):
        '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
        RGB color.'''
        color_norm  = colors.Normalize(vmin=0, vmax=N-1)
        scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
        def map_index_to_rgb_color(index):
            return scalar_map.to_rgba(index)
        return map_index_to_rgb_color

def tC(myList, myNumber):
        """
        Assumes myList is sorted. Returns closest value to myNumber.

        If two numbers are equally close, return the smallest number.
        """
        #print myList.tolist().index(myNumber)
        myList = np.delete(myList,myList.tolist().index(myNumber))
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return myList[0]
        if pos == len(myList):
            return myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        return min(after-myNumber,myNumber-before)

        '''
        if after - myNumber < myNumber - before:
            return after
        else:
            return before
        '''


from scipy.spatial.distance import cdist, pdist

def _skelbow(Xd, k):
    kmeansvar = skcluster.KMeans(n_clusters=k, max_iter=1000).fit(Xd)
    k_euclid = cdist(Xd, kmeansvar.cluster_centers_)
    dist = np.min(k_euclid, 1)
    wcss = sum(dist**2)
    return wcss

def derivative(arr):
    return arr[1:]-arr[:-1]
def sec_derivative(arr):
    return arr[:-2]+arr[2:]-2*arr[1:-1]
def curvature(arr):
    yp = derivative(arr)[:-1]
    ypp = sec_derivative(arr)
    return np.abs(ypp)/(1+yp**2)**1.5

def elbow(Xd, max_clusters, slope):
    wcss = Parallel(n_jobs=4)(delayed(_skelbow)(Xd, k) for k in range(1, max_clusters))
    bss = np.array(wcss)
    #bss = np.array(tss) - wcss
    plt.figure()
    plt.plot(bss)
    # plt.show()
    #ncluster = 1
    #import IPython; IPython.embed()
    ncluster = 1
    while bss[ncluster] < slope * bss[ncluster-1]:
        ncluster += 1
        if ncluster == max_clusters:
            break
    return ncluster
    #plt.plot(bss)
    #plt.show()

if __name__ == "__main__":

    #data_dir = '/data1/SETI/SERENDIP/vishal/'
    data_dir = '/home/yunfanz/Projects/SETI/serendip/Data/'
    fname = "20170604_172322.dbase.drfi.clean.exp_time"
    infile = data_dir + fname
    #infile = data_dir + "20170325_092539.dbase.drfi.clean.exp_time"
    #infile = data_dir + "20170604_172322.dbase.drfi.clean.exp_time"
    #infile = sys.argv[1]
    #data1 = np.loadtxt(infile,dtype={'names': ('freq','time','ra','dec','pow'),'formats':('f16','f8','f8','f8','f8')})
    #plt.scatter(data1['freq'],data1['time'],marker='.',s=0.3)
    data1 = np.loadtxt(infile)
    data1 = pd.DataFrame(data=data1, columns=['freq','time','ra','dec','pow'])
    #plt.scatter(data1['freq'],data1['time'],marker='.',s=0.3)
    #plt.show()
    sortfreq = np.unique(np.sort(data1['freq']))
    sorttime = np.unique(np.sort(data1['time']))
    print min(sortfreq),max(sortfreq)
    X = whiten(zip(data1['freq'], data1['time']))

    allindices, alldists, meandists, klist = KNN(X, 8, srcDims=2)
    binmax  = np.amax(meandists)/2
    counts, bins = np.histogram(meandists, range=(0, binmax), bins=100)
    cutoff = bins[30]
    water_flags = meandists<cutoff
    #import IPython; IPython.embed()

    allindices, alldists, meandists, klist = KNN(X, 8, srcDims=1)
    binmax  = np.amax(meandists)/4
    counts, bins = np.histogram(meandists, range=(0, binmax), bins=100)
    cutoff = bins[10]
    freq_flags = meandists<cutoff

    # allindices, alldists, meandists, klist = KNN(X[:,::-1], 8, srcDims=1)
    # binmax  = np.amax(meandists)/16
    # counts, bins = np.histogram(meandists, range=(0, binmax), bins=100)
    # cutoff = bins[3]
    # time_flags = meandists<cutoff

    #broadband_flags = (water_flags & time_flags) ^ freq_flags #won't be exhaustive
    broadband_flags = water_flags ^ freq_flags
    
    flags = np.where(water_flags | freq_flags)[0]
    flags_ = np.where(~(water_flags | freq_flags))[0]
    cflags = np.where((water_flags | freq_flags) ^ broadband_flags)[0]
    
    # plt.plot(counts)
    # plt.show()

    #flags_ = np.argwhere(meandists>cutoff).squeeze()
    #flags = np.argwhere(meandists<=cutoff).squeeze()

    plt.figure()
    plt.scatter(data1['freq'][flags_],data1['time'][flags_],color='r',marker='.',s=2)
    plt.scatter(data1['freq'][flags],data1['time'][flags],color='b',marker='.',s=2)
    plt.scatter(data1['freq'][cflags],data1['time'][cflags],color='g',marker='.',s=2)
    #plt.show()

    #import IPython; IPython.embed()

    data1['cluster'] = 0
    data1['remove'] = 0
    # data1.loc[broadband_flags]['remove'] = 1
    data_dense = data1.loc[flags]
    #data_dense.to_csv(data_dir+fname.split('.')[0]+".flagged")
    #Xd = whiten(zip(data_dense['freq'], data_dense['time']))
    Xd = data_dense['freq']
    max_clusters = 40
    #tss = sum(pdist(Xd)**2)/Xd.shape[0]
    ncluster = elbow(Xd[:,np.newaxis], max_clusters, 0.85)
    #import IPython; IPython.embed()
    kmeans_ = skcluster.KMeans(n_clusters=ncluster, max_iter=1000).fit(Xd[:,np.newaxis])
    data_dense['cluster'] = kmeans_.labels_ + 1
    #data1.set_value('cluster', np.where(flags)[0], kmeans_.labels_)
    data1.loc[flags, 'cluster'] = kmeans_.labels_ + 1
    

    print "generating pairplot"
    #plt.figure()
    g = sns.pairplot(data_dense, hue='cluster', vars=['freq','time','ra','dec'],
        plot_kws={"s":10})
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)
    
    #import IPython; IPython.embed()
    for i in xrange(1, ncluster):
        cluster = data1.loc[data1['cluster'] == i]
        loc = SkyCoord(cluster['ra'],cluster['dec'],unit='deg',frame='icrs')
        sep = loc[0].separation(loc[:]) 
        maxsep = np.amax(sep.deg)
        print i, maxsep
        if maxsep > 16.:
            data1.loc[data1['cluster'] == i, 'remove'] = 1
        elif np.amax(cluster['freq'])-np.amin(cluster['freq']) > 2:
            data1.loc[data1['cluster'] == i, 'remove'] = 2
        else:
            print 'Candidate cluster {}'.format(i)
            #plt.figure()
            #plt.scatter(cluster['freq'], cluster['time'], marker='.', s=2)

    data_candidate = data1.loc[(data1['remove'] == 0) & (data1['cluster']>0)]
    data_clean = data1.loc[data1['cluster']==0]

    plt.figure()
    plt.scatter(data_clean['freq'],data_clean['time'],color='r',marker='.',s=2)
    plt.scatter(data_candidate['freq'],data_candidate['time'],color='g',marker='.',s=2)
    plt.show()


    import IPython; IPython.embed()

    #nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)


    #print len(sortfreq),len(data1['freq'])
    #sys.exit(0)
    #sortfreq = list(set(data1['freq']))
    #sortfreq = np.sort(sortfreq)
    #sorttime = np.sort(data1['time'])
    #cfreq = np.zeros(len(data1['freq']))
    #ctime = np.zeros(len(data1['time']))

    #print "Doing sorting and getting nearest hits"
    #cfreq = Parallel(n_jobs=12)(delayed(tC)(sortfreq, d) for d in data1['freq'])
    #ctime = Parallel(n_jobs=12)(delayed(tC)(sorttime, d) for d in data1['time'])
    
    # for i,d in enumerate(data1['freq']):
    #     #print i
    #     cfreq[i] = tC(sortfreq,d)
    #     ctime[i] = tC(sorttime,data1['time'][i])*(10E-7)
    #     #print cfreq[i],ctime[i]
    #     #print close[i]
    #     #close[i] = np.sort(nsmallest(2, data1['freq'], key=lambda x: abs(x-float(i))))[1]
    #print ctime,min(cfreq)
    #sys.exit(0)
    #print close
    #water = zip(data1['freq'],data1['time'],data1['ra'],data1['dec'],data1['pow'],close)
    #water = zip(data1['ra'],data1['dec'],close)
    #water = zip(cfreq,ctime)
    # water1 = np.array(cfreq)
    # #water2 = np.array(data1['pow'])
    # water2 = np.array(ctime)*(10E-7)
    # w1 = whiten(zip(water1,water2)) #divide by std for each axis
    # import IPython; IPython.embed()

    # ncluster = 20
    # clusterdiff = 15
    # kw1,dist = kmeans(w1,ncluster,iter=1000)
    # idx2,test = vq(w1,kw1)
    # #print idx2
    # plt.xlim(1235,1515)
    # plt.ylim(0,1100)
    # for i in range(1,clusterdiff):
    #     plt.scatter(data1['freq'][idx2==i],data1['time'][idx2==i],color='r',marker='.',s=0.3)
    # #plt.scatter(data1['freq'][idx2==2],data1['time'][idx2==2],color='r',marker='.',s=0.3)
    # for i in range(clusterdiff,ncluster+1):
    #     plt.scatter(data1['freq'][idx2==i],data1['time'][idx2==i],color='blue',marker='.',s=0.3)
    
    # plt.show()
