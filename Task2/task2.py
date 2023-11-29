import kornia as K 
import kornia.feature as KF
import numpy as np 
import pandas as pd

import rasterio
from rasterio.plot import reshape_as_image
from rasterio.features import rasterize

# from shapely.geometry import mapping, Point, Polygon
# from shapely.ops import cascaded_union

import matplotlib.pyplot as plt


RASTER_PATH1 = 'T36UYA_20160212T084052_TCI.jp2'
RASTER_PATH2 = 'T36UYA_20160330T082542_TCI.jp2'
with rasterio.open(RASTER_PATH1, "r", driver='JP2OpenJPEG') as src1:
    raster_image1 = src1.read()
    raster_meta1 = src1.meta
with rasterio.open(RASTER_PATH2, "r", driver='JP2OpenJPEG') as src2:
    raster_image2 = src2.read()
    raster_meta2 = src2.meta

import itertools
raster_img1 = reshape_as_image(raster_image1)
raster_img2 = reshape_as_image(raster_image2)
# total count of fragmen frag_count*10
frag_count = 10
frag_size = int(raster_img1.shape[0] / frag_count)
frag_dict1 = {}
frag_dict2 = {}
    
for y, x in itertools.product(range(frag_count), range(frag_count)):
    frag_dict1[(x, y)] = raster_img1[y*frag_size: (y+1)*frag_size,  
                                      x*frag_size: (x+1)*frag_size,:]
    frag_dict2[(x, y)] = raster_img2[y*frag_size: (y+1)*frag_size,  
                                      x*frag_size: (x+1)*frag_size,:]


from PIL import Image as im
for i in range(0,frag_count):
    for j in range(0,frag_count):
        data = im.fromarray(frag_dict1[(i,j)])
        data.save("Task2/IMG1/{}.png".format(i*10+j))
        data = im.fromarray(frag_dict2[(i,j)])
        data.save("Task2/IMG2/{}.png".format(i*10+j))


