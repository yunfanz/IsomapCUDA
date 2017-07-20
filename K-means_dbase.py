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
from skimage.measure import label, regionprops
import matplotlib
carr = np.random.rand(256, 3); carr[0,:] = 0
cmap = matplotlib.colors.ListedColormap(carr)
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

# def derivative(arr):
#     return arr[1:]-arr[:-1]
# def sec_derivative(arr):
#     return arr[:-2]+arr[2:]-2*arr[1:-1]
# def curvature(arr):
#     yp = derivative(arr)[:-1]
#     ypp = sec_derivative(arr)
#     return np.abs(ypp)/(1+yp**2)**1.5

def elbow(Xd, max_clusters, slope):
    wcss = Parallel(n_jobs=4)(delayed(_skelbow)(Xd, k) for k in range(1, max_clusters))
    bss = np.array(wcss)
    #bss = np.array(tss) - wcss
    # plt.figure()
    # plt.plot(bss)
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
def injectET(data1, freq_start, time_start):
        """
        The input is exp_time file read using python. The array should have five corresponding attributes
        freq,time,ra,dec,pow
        """
        # Find most common RA and DEC
        # There are two ways to do this. i
        # First is to find most frequently occuring pairs  

        #RA = data1['ra']
        #DEC = data1['dec']

        #d = zip(RA,DEC)
    
        # Find most common pairs of RA and DEC 
        #RAmax,DECmax = Counter(d).most_common(1)[0][0] 
        
        #(v1,c1) = np.unique(RA,return_counts=True)
        #(v2,c2) = np.unique(DEC,return_counts=True)
        
        #Extract the data for this block
        #ETdata = data1[((data1['dec']==DECmax) & (data1['ra']==RAmax))]
        # ^ not what we want eventually so drop

        # What we need is the pairs of RA and DEC which lasted longest. 
        # By looking at the data manully, I determind following ET location to inject birdie. 

        
        ETtime_start = time_start 
        ETtime_end = time_start + 50
        if False:
            ETdec = 16.9
            ETra =  19.1 
        else:
            #import IPython; IPython.embed()
            start_exind = np.random.choice(np.where(np.abs(data1[:,1]-ETtime_start)<1)[0])
            #end_exind = np.random.choice(np.where(np.abs(data1[:,0]-ETtime_end)<0.1)[0])
            ETdec = data1[start_exind][3]
            ETra = data1[start_exind][2]
        ETfreq_start = freq_start
        ETfreq_end = freq_start + 0.001
        ETpow = 15 # This is in log scale. 

        #From above values, calculate slope 
        ETslope = (ETfreq_end - ETfreq_start)/(ETtime_end - ETtime_start)
        
        #Fixed, this will overide ETfreq_end frequencies. The value must be in MHz/sec
        ETslope = 0.0001
        
        ETtime = ETtime_end - ETtime_start
        ETdata = np.zeros((ETtime,5))

        for i in range(ETtime):
            ETdata[i,:] = (ETfreq_start+(i)*ETslope,ETtime_start+i,ETra,ETdec,ETpow)    

        data2 = np.concatenate((data1,ETdata),axis=0)
        return data2

if __name__ == "__main__":

    data_dir = '/data1/SETI/SERENDIP/vishal/'
    #data_dir = '/home/yunfanz/Projects/SETI/serendip/Data/'
    fname = "20170325_092539.dbase.drfi.clean.exp_time"
    infile = data_dir + fname
    #infile = data_dir + "20170325_092539.dbase.drfi.clean.exp_time"
    #infile = data_dir + "20170604_172322.dbase.drfi.clean.exp_time"
    #infile = sys.argv[1]
    #data1 = np.loadtxt(infile,dtype={'names': ('freq','time','ra','dec','pow'),'formats':('f16','f8','f8','f8','f8')})
    #plt.scatter(data1['freq'],data1['time'],marker='.',s=0.3)
    data1 = np.loadtxt(infile)
    data1 = injectET(data1, 1322, 500)
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
    binmax  = np.amax(meandists)/8
    counts, bins = np.histogram(meandists, range=(0, binmax), bins=100)
    cutoff = bins[0]
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
    #plt.scatter(data1['freq'][cflags],data1['time'][cflags],color='g',marker='.',s=2)
    #plt.show()

    #import IPython; IPython.embed()


    data1['cluster'] = 0
    data1['remove'] = 0
    # data1.loc[broadband_flags]['remove'] = 1
    
    #data_dense.to_csv(data_dir+fname.split('.')[0]+".flagged")
    #Xd = whiten(zip(data_dense['freq'], data_dense['time']))
    data_dense = data1.loc[flags]
    Xd = data1.loc[flags, 'freq']

    if False:
        max_clusters = 40
        #tss = sum(pdist(Xd)**2)/Xd.shape[0]
        ncluster = elbow(Xd[:,np.newaxis], max_clusters, 0.90)
        #import IPython; IPython.embed()
        kmeans_ = skcluster.KMeans(n_clusters=ncluster, max_iter=1000).fit(Xd[:,np.newaxis])
        data_dense['cluster'] = kmeans_.labels_ + 1
        #data1.set_value('cluster', np.where(flags)[0], kmeans_.labels_)
        data1.loc[flags, 'cluster'] = kmeans_.labels_ + 1
    elif True:
        #TODO use hist2D
        nbins = 200
        counts, bin_edges = np.histogram(Xd, bins=nbins)
        cluster_labels = label(counts>0)
        for i in xrange(nbins):
            if cluster_labels[i] > 0:
                cur_bin = (data1['freq'] >= bin_edges[i]) & (data1['freq'] < bin_edges[i+1])
                data1.loc[cur_bin & (water_flags | freq_flags), 'cluster'] = cluster_labels[i]
        ncluster = np.amax(cluster_labels)
        #import IPython; IPython.embed()
    else:
        nbins = (50,50)
        counts, freq_edges, time_edges = np.histogram2d(data_dense['freq'], data_dense['time'], bins=nbins)
        cluster_labels = label(counts>0)
        plt.subplot(121)
        plt.imshow(cluster_labels.T, cmap=cmap)
        for i in xrange(nbins[0]):
            column = np.unique(cluster_labels[i])
            column = np.sort(column)
            if column.size <= 2: continue
            column = column[1:]
            print column
            fcur_bin = (data_dense['freq'] >= freq_edges[i]) & (data_dense['freq'] < freq_edges[i+1])
            freq_peaks = {}
            for cluster in column:
                time_ind = np.argwhere(cluster_labels[i][:]==cluster)
                start = np.amin(time_ind)
                end = np.amax(time_ind)
                tcur_bin = (data_dense['time'] >= time_edges[start]) & (data_dense['time'] < time_edges[end])
                tempcount, temp_bins = np.histogram(data_dense.loc[tcur_bin, 'freq'], bins=5)
                peak_loc = np.argmax(tempcount)
                #peak_loc = np.mean(data_dense.loc[tcur_bin, 'freq'])
                freq_peaks[peak_loc] = freq_peaks.get(peak_loc,[]) + [cluster]
            for ploc in freq_peaks.keys():
                example_ind = np.amin(freq_peaks[ploc])
                for c in freq_peaks[ploc]:
                    cluster_labels = np.where(cluster_labels==c, example_ind, cluster_labels)
        for i, c in enumerate(np.unique(cluster_labels)):
            cluster_labels = np.where(cluster_labels==c, i, cluster_labels)

        plt.subplot(122)
        plt.imshow(cluster_labels.T, cmap=cmap)

        #import IPython; IPython.embed()
        ncluster = np.amax(cluster_labels)
        for i in xrange(nbins[0]):
            fcur_bin = (data1['freq'] >= freq_edges[i]) & (data1['freq'] < freq_edges[i+1])
            for j in xrange(nbins[1]):
                if cluster_labels[i,j] == 0:
                    continue
                tcur_bin = (data1['time'] >= time_edges[j]) & (data1['time'] < time_edges[j+1])
                cur_bin = tcur_bin & fcur_bin
                data1.loc[cur_bin,'cluster'] = cluster_labels[i,j]
                #data1.loc[cur_bin,'cluster'] = cluster_labels[i,j]

    print "generating pairplot"
    #plt.figure()
    data_dense = data1.loc[flags]
    g = sns.pairplot(data_dense, hue='cluster', vars=['freq','time','ra','dec'],
        plot_kws={"s":10})
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)
    
    #import IPython; IPython.embed()
    for i in xrange(1, ncluster+1):
        cluster = data1.loc[data1['cluster'] == i]
        #import IPython; IPython.embed()
        loc = SkyCoord(cluster['ra'],cluster['dec'],unit='deg',frame='icrs')
        sep = loc[0].separation(loc[:])
        maxsep = np.amax(sep.deg)
        print i, maxsep
        if maxsep > 16.:
            data1.loc[data1['cluster'] == i, 'remove'] = 1
        elif np.amax(cluster['freq'])-np.amin(cluster['freq']) > 1:
            data1.loc[data1['cluster'] == i, 'remove'] = 2
        else:
            print 'Candidate cluster {}'.format(i)
            #plt.figure()
            #plt.scatter(cluster['freq'], cluster['time'], marker='.', s=2)

    data_candidate = data1.loc[(data1['remove'] == 0) & (data1['cluster']>0)]
    data_clean = data1.loc[data1['cluster']==0]
    data_broad = data1.loc[data1['remove']==2]

    plt.figure()
    plt.scatter(data_clean['freq'],data_clean['time'],color='r',marker='.',s=2)
    plt.scatter(data_candidate['freq'],data_candidate['time'],color='g',marker='.',s=2)
    #plt.scatter(data_broad['freq'],data_broad['time'],color='b',marker='.',s=2)
    plt.show()


    import IPython; IPython.embed()


