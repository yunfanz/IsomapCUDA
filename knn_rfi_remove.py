import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq,kmeans,whiten,kmeans2
from sklearn import cluster as skcluster
import sys, os, fnmatch
from decimal import *
from joblib import Parallel, delayed
from KNearestNeighbours import *
from KMeans import KMeans
import matplotlib
import seaborn as sns
import pandas as pd
from function_utils import savitzky_golay
from astropy.coordinates import SkyCoord
from scipy.spatial.distance import cdist, pdist

def find_files(directory, pattern='*.dbase.drfi.clean.exp_time', sortby="auto"):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    files = np.sort(files)
    return files

def _skelbow(Xd, k):
    kmeansvar = skcluster.KMeans(n_clusters=k, max_iter=1000).fit(Xd)
    k_euclid = cdist(Xd, kmeansvar.cluster_centers_)
    dist = np.min(k_euclid, 1)
    wcss = sum(dist**2)
    return wcss

def elbow(Xd, max_clusters, slope=0.85):
    wcss = Parallel(n_jobs=4)(delayed(_skelbow)(Xd, k) for k in range(1, max_clusters))
    bss = np.array(wcss)
    ncluster = 1
    while bss[ncluster] < slope * bss[ncluster-1]:
        ncluster += 1
        if ncluster == max_clusters:
            break
    return ncluster

def get_flags(X, bin_val=30, freqbin_val=10):
    """
    Parameters:
    X: input array, of shape (num_data_points, 2)
    bin_val: adjustable parameter for cutoff
    freqbin_val: another adjustable parameter for cutoff

    Return:
    flags: dense region flags, Boolean array of length X.shape[0] 
    flags_: sparse region flags, Boolean array of length X.shape[0]
    cflags: complementary flags, Boolean array of length X.shape[0]
    """
    allindices, alldists, meandists, klist = KNN(X, 8, srcDims=2)
    binmax  = np.amax(meandists)/2
    counts, bins = np.histogram(meandists, range=(0, binmax), bins=100)
    cutoff = bins[bin_val]
    water_flags = meandists<cutoff

    allindices, alldists, meandists, klist = KNN(X, 8, srcDims=1)
    binmax  = np.amax(meandists)/4
    counts, bins = np.histogram(meandists, range=(0, binmax), bins=100)
    cutoff = bins[freqbin_val]
    freq_flags = meandists<cutoff

    broadband_flags = water_flags ^ freq_flags
    
    flags = np.where(water_flags | freq_flags)[0]
    flags_ = np.where(~(water_flags | freq_flags))[0]
    cflags = np.where((water_flags | freq_flags) ^ broadband_flags)[0]

    return flags, flags_, cflags

def make_plots(data, flags, flags_, figname):

    f, axes = plt.subplots(1,2, figsize=(16, 8))
    axes[0].scatter(data1['freq'][flags_],data1['time'][flags_],color='r',marker='.',s=2)
    axes[0].scatter(data1['freq'][flags],data1['time'][flags],color='b',marker='.',s=2)
    axes[0].set_title('knn selection')

    axes[1].scatter(data_clean['freq'],data_clean['time'],color='r',marker='.',s=2)
    axes[1].scatter(data_candidate['freq'],data_candidate['time'],color='g',marker='.',s=2)
    axes[1].set_title('Clean and Candidate')
    plt.savefig(figname)
    #plt.show()

    if False:
        print "generating pairplot"
        data_dense = data1.loc[flags]
        g = sns.pairplot(data_dense, hue='cluster', vars=['freq','time','ra','dec'],
            plot_kws={"s":10})
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            g.axes[i, j].set_visible(False)



if __name__ == "__main__":

    data_dir = '/data1/SETI/SERENDIP/vishal/'
    files = find_files(data_dir, pattern='*.dbase.drfi.clean.exp_time')
    for infile in files:
        
        file_dir = os.path.dirname(infile)
        fname = infile.split('/')[-1].split('.')[0] #e.g. "20170604_172322"
        print "########## "+ fname+" ###########"

        data1 = np.loadtxt(infile)
        data1 = pd.DataFrame(data=data1, columns=['freq','time','ra','dec','pow'])
        X = whiten(zip(data1['freq'], data1['time']))

        flags, flags_, cflags = get_flags(X, 30, 10) 


        data1['cluster'] = 0
        data1['remove'] = 0

        data_dense = data1.loc[flags]
        Xd = data_dense['freq']
        max_clusters = 40
        ncluster = elbow(Xd[:,np.newaxis], max_clusters, 0.85)
        kmeans_ = skcluster.KMeans(n_clusters=ncluster, max_iter=1000).fit(Xd[:,np.newaxis])
        data_dense['cluster'] = kmeans_.labels_ + 1
        data1.loc[flags, 'cluster'] = kmeans_.labels_ + 1
        

        for i in xrange(1, ncluster+1):
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


        data_candidate = data1.loc[(data1['remove'] == 0) & (data1['cluster']>0)]
        data_clean = data1.loc[data1['cluster']==0]
        data1.to_csv(data_dir+fname.split('.')[0]+".knn")
        figname = file_dir+'/'+fname+".knn.png"
        #import IPython; IPython.embed()
        make_plots(data1, flags, flags_, figname)
