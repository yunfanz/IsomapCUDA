import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import matplotlib as mplt
import sys,os

if __name__ == "__main__":    
    #infile = "20160122_105802.dbase.exp_time"
    infile = sys.argv[1]
    data1 = np.loadtxt(infile,dtype={'names': ('freq','time','ra','dec','pow'),'formats':('f16','f8','f8','f8','f8')})
    #xl = 1259.1868
    #xh = 1259.1871
    xl = float(sys.argv[2])
    xh = float(sys.argv[3])
    data2 = data1[np.where(np.logical_and(data1['freq']>=xl,data1['freq']<=xh))]

    loc = SkyCoord(data2['ra'],data2['dec'],unit='deg',frame='icrs')
    sep = loc[0].separation(loc[:]) 
    sep = sep.deg

    freqmin = min(data1['freq'])
    freqmax = max(data1['freq'])
    
    sepl = min(sep)
    seph = max(sep)

    timel = min(data1['time'])
    timeh = max(data1['time'])

    mplt.rcParams['axes.linewidth'] = 2
    plt.subplots_adjust(hspace = .001)
    plt.subplots_adjust(wspace = .001) 

    ax1 = plt.subplot2grid((5,5), (0,0), rowspan=5,colspan=4)
    #plt.xlim(freqmin,freqmax)
    plt.xlim(xl,xh)
    plt.ylim(timel,timeh)
    ax1.tick_params(length=4, width=2)
    ax1.set_xlabel('Frequency (MHz)',fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (sec)',fontsize=12, fontweight='bold')
    ax1.scatter(data2['freq'],data2['time'],marker='.',s=0.7)
   
    ax2 = plt.subplot2grid((5,5), (0,4), rowspan=5,colspan=1) 
    ax2.set_xlabel('Sky separation (deg)',fontsize=12, fontweight='bold')
    plt.ylim(timel,timeh)
    plt.xlim(sepl-2,seph+2)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.xticks(np.arange(sepl,seph, 10.0))
    ax2.scatter(sep,data2['time'])
   
    plt.savefig("test.pdf")
    plt.show()
