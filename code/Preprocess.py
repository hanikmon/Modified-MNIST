# library
# standard library

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy   as np
import pandas as pd
import scipy.misc # to visualize only

from skimage import data, img_as_float, color, feature
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.util import random_noise
from skimage.filters.rank import median
from skimage.morphology import disk
from scipy import ndimage as ndi

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.metrics import accuracy_score

# load da
trainXPath = "../../kaggleDatasets/train_x.csv"
trainYPath = "../../kaggleDatasets/train_y.csv"
testXPath = "../../kaggleDatasets/test_x.csv"
dtype = torch.cuda.FloatTensor
# dtype =  torch.FloatTensor
class kaggleDataset(Dataset):
    def __init__(self, csv_pathX, csv_pathY, transforms=None):
        self.x_data = pd.read_csv(csv_pathX,header=None)
        self.y_data = pd.read_csv(csv_pathY,header=None).as_matrix()
        self.transforms = transforms

    def __getitem__(self, index):
        # label = np.zeros((10))
        # label[self.y_data[index][0]] = 1
        # singleLable = torch.from_numpy(label).type(dtype)

        singleLable = torch.from_numpy(self.y_data[index]).type(torch.FloatTensor)
        singleX = np.asarray(self.x_data.iloc[index]/255.0).reshape(1, 64, 64)
        x_tensor = torch.from_numpy(singleX).type(dtype)
        return x_tensor, singleLable

    def __len__(self):
        return len(self.x_data.index)


class kaggleDatasetNoReshape(Dataset):
    def __init__(self, csv_pathX, csv_pathY, transforms=None):
        self.x_data = pd.read_csv(csv_pathX,header=None)
        self.y_data = pd.read_csv(csv_pathY,header=None).as_matrix()
        self.transforms = transforms

    def __getitem__(self, index):
        # label = np.zeros((10))
        # label[self.y_data[index][0]] = 1
        # singleLable = torch.from_numpy(label).type(dtype)

        singleLable = torch.from_numpy(self.y_data[index]).type(torch.FloatTensor)
        singleX = np.asarray(self.x_data.iloc[index])
        x_tensor = torch.from_numpy(singleX).type(torch.FloatTensor)
        return x_tensor, singleLable

    def __len__(self):
        return len(self.x_data.index)

class testDataset(Dataset):
    def __init__(self, csv_pathX, transforms=None):
        self.x_data = pd.read_csv(csv_pathX,header=None)
        self.transforms = transforms

    def __getitem__(self, index):

        singleX = np.asarray(self.x_data.iloc[index]/255.0).reshape(1, 64, 64)
        x_tensor = torch.from_numpy(singleX).type(dtype)
        return x_tensor

    def __len__(self):
        return len(self.x_data.index)

def preprocessImage(image,th,r,sig):
    # th = 220
    #r=2,3 works well
    # sigma = 1
    #global_thresh = threshold_otsu(image)
    binary_global = image > th
    outputImage = median(binary_global, disk(r))
    output = np.where(outputImage>250,1,0)
    edges = feature.canny(outputImage, sigma=sig)
    outputEdge = np.where(edges>250,1,0)
    return output,outputEdge

#######################
        
trainX = np.loadtxt(trainXPath, delimiter=",") # load from text 
trainY = np.loadtxt(trainYPath, delimiter=",") 
trainX = trainX.reshape(-1, 64, 64) # reshape 
trainY = trainY.reshape(-1, 1) 

n = 300;
plt.imshow(trainX[n],cmap = 'gray') # to visualize only 



#image = data.page()
image = trainX[n]

global_thresh = 200;
#threshold_otsu(image)

binary_global = image > global_thresh

block_size = 35
binary_adaptive = threshold_adaptive(image, block_size, offset=10)

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image')

ax1.imshow(binary_global)
ax1.set_title('Global thresholding')

ax2.imshow(binary_adaptive)
ax2.set_title('Adaptive thresholding')

for ax in axes:
    ax.axis('off')

plt.show()


###################################################################################

original = img_as_float(data.chelsea()[100:250, 50:300])

sigma = 0.155
noisy = random_noise(original, var=sigma**2)
noisy = binary_global
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5),
                       sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, multichannel=False, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
print("Estimated Gaussian noise standard deviation = {}".format(sigma_est))


ax[0, 0].imshow(noisy)
ax[0, 0].axis('off')
ax[0, 0].set_title('Noisy')
ax[0, 1].imshow(denoise_tv_chambolle(noisy, weight=0.1, multichannel=False))
ax[0, 1].axis('off')
ax[0, 1].set_title('TV')
ax[0, 2].imshow(denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15,
                multichannel=False))
ax[0, 2].axis('off')
ax[0, 2].set_title('Bilateral')
ax[0, 3].imshow(denoise_wavelet(noisy, multichannel=False))
ax[0, 3].axis('off')
ax[0, 3].set_title('Wavelet denoising')

ax[1, 1].imshow(denoise_tv_chambolle(noisy, weight=0.2, multichannel=False))
ax[1, 1].axis('off')
ax[1, 1].set_title('(more) TV')
ax[1, 2].imshow(denoise_bilateral(noisy, sigma_color=0.1, sigma_spatial=15,
                multichannel=False))
ax[1, 2].axis('off')
ax[1, 2].set_title('(more) Bilateral')
ax[1, 3].imshow(denoise_wavelet(noisy, multichannel=False, convert2ycbcr=True))
ax[1, 3].axis('off')
ax[1, 3].set_title('Wavelet denoising\nin YCbCr colorspace')
#ax[1, 0].imshow(original)
#ax[1, 0].axis('off')
#ax[1, 0].set_title('Original')

fig.tight_layout()

plt.show()

#############################################################################
noisy_image = binary_global


fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(noisy_image)#, vmin=0, vmax=255, cmap=plt.cm.gray)
ax[0].set_title('Noisy image')

ax[1].imshow(median(noisy_image, disk(1)), vmin=0, vmax=255, cmap=plt.cm.gray)
ax[1].set_title('Median $r=1$')

ax[2].imshow(median(noisy_image, disk(2)), vmin=0, vmax=255, cmap=plt.cm.gray)
ax[2].set_title('Median $r=2$')

ax[3].imshow(median(noisy_image, disk(3)), vmin=0, vmax=255, cmap=plt.cm.gray)
ax[3].set_title('Median $r=3$')

for a in ax:
    a.axis('off')

plt.tight_layout()

#########################################################


# Generate noisy image of a square
#im = np.zeros((128, 128))
#im[32:-32, 32:-32] = 1
#
#im = ndi.rotate(im, 15, mode='constant')
#im = ndi.gaussian_filter(im, 4)
#im += 0.2 * np.random.random(im.shape)
im = median(noisy_image, disk(3))
# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im)
edges2 = feature.canny(im, sigma=3)

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.tight_layout()

plt.show()

################################################################


#image = data.coins()[50:-50, 50:-50]
image = median(noisy_image, disk(2))
# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(3))

# remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)
image_label_overlay = label2rgb(label_image, image=image)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)

for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 100:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()