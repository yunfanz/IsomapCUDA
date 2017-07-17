import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray
import pycuda.driver as cu
import pycuda.autoinit
import numpy as np
import matplotlib.pyplot as plt
import os, fnmatch
import pandas as pd
from KNearestNeighbours import KNN
from matplotlib.colors import LogNorm
def find_files(directory, pattern='*.png', sortby="auto"):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    files = np.sort(files)
    return files

def _get_axes(I):
  sortfreq = np.unique(np.sort(I[0]))
  sorttime = np.unique(np.sort(I[1]))
  freqs = np.linspace(min(sortfreq),max(sortfreq),int(np.sqrt(I.size)/32)*32)
  times = np.linspace(min(sorttime),max(sorttime),int(np.sqrt(I.size)/32)*32)
  return freqs, times
def grid_numpy(I):
  freqs, times = _get_axes(I)
  height, width = freqs.size, times.size
  dense = np.histogram2d(I[1], I[0], bins=[height, width])
  return dense

def grid_data(I):

  # Get contiguous image + shape.
  freqs, times = _get_axes(I)
  height, width = freqs.size, times.size
  I = np.float32(I.copy())

  # Get block/grid size for steps 1-3.
  block_size =  (32,32,1)
  grid_size =   (width/(block_size[0]),
                height/(block_size[0]))

  # Initialize variables.
  dense       = np.zeros([height,width], dtype=np.int32) 
  width         = np.int32(width)
  height        = np.int32(height)

  # Transfer labels asynchronously.
  dense_d = gpuarray.to_gpu_async(labeled)
  counter_d = gpuarray.to_gpu_async(count)

  # Bind CUDA textures.
  I_cu = cu.matrix_to_array(I, order='C')
  cu.bind_array_to_texref(I_cu, image_texture)

  # Step 1.
  descent_kernel(labeled_d, width, 
  height, block=block_size, grid=grid_size)
  return



if __name__ == "__main__":
  data_dir = '/data1/SETI/SERENDIP/vishal/'
  files = find_files(data_dir, pattern="*.exp_time")
  fig, axes = plt.subplots(len(files), 2)
  for i, infile in enumerate(files):
    print infile
    data1 = np.loadtxt(infile)
    data1 = pd.DataFrame(data=data1, columns=['freq','time','ra','dec','pow'])
    X = np.array(zip(data1['freq'], data1['time']))
    allindices, alldists, meandists, klist = KNN(X, 8, srcDims=2)
    axes[i,0].hist(meandists, range=(0, np.amax(meandists)/2), bins=100)
    axes[i,1].hist(np.log(meandists+1.e-9), bins=100)
  plt.tight_layout()
  plt.show()

    # dense = grid_numpy(X)[0]
    # import IPython; IPython.embed()
    # plt.imshow(dense, interpolation='nearest', norm=LogNorm(vmin=dense.min(), vmax=dense.max()))
    # plt.title(infile.split('/')[-1])
    # plt.colorbar()
    # import IPython; IPython.embed()