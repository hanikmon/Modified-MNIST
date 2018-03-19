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
from skimage.morphology import disk,closing, square,erosion
from scipy import ndimage as ndi

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb
# watershed
from scipy import ndimage

from skimage.morphology import watershed
from skimage.feature import peak_local_max


# load da
trainXPath = "../data/og/train_x.csv"
trainYPath = "../data/og/train_y.csv"
testXPath = "../data/og/test_x.csv"

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

def showthreshold(image,th,pltsh):
    binary_global = (image > th).astype(float)
    if pltsh:        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 8))
        ax0, ax1 = axes
        plt.gray()
        
        ax0.imshow(image)
        ax0.set_title('Image')
        
        ax1.imshow(binary_global)
        ax1.set_title('Global thresholding')
        
        for ax in axes:
            ax.axis('off')
        
        plt.show()
    return binary_global
def medianfilter(image,pltsh):
    if pltsh:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        ax = axes.ravel()
        
        ax[0].imshow(image)#, vmin=0, vmax=255, cmap=plt.cm.gray)
        ax[0].set_title('Noisy image')
        
        ax[1].imshow(median(image, disk(1)), vmin=0, vmax=255, cmap=plt.cm.gray)
        ax[1].set_title('Median $r=1$')
        
        ax[2].imshow(median(image, disk(2)), vmin=0, vmax=255, cmap=plt.cm.gray)
        ax[2].set_title('Median $r=2$')
        
        ax[3].imshow(median(image, disk(3)), vmin=0, vmax=255, cmap=plt.cm.gray)
        ax[3].set_title('Median $r=3$')
        
        for a in ax:
            a.axis('off')
        
        plt.tight_layout()
        return median(image, disk(1))
def Cannyfilter(image,pltsh):
    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(image)
    edges2 = feature.canny(image, sigma=3)
    if pltsh:      
        # display results
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                            sharex=True, sharey=True)
        
        ax1.imshow(image, cmap=plt.cm.gray)
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
    return edges1,edges2
def bwlabeling(image,pltsh):
    if pltsh:
        # apply threshold
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(3))
        
        # remove artifacts connected to image border
        #cleared = clear_border(bw)
        
        # label image regions
        label_image = label(bw)
        image_label_overlay = label2rgb(label_image, image=image)
        
        fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols =2, figsize=(5, 2))
        ax1.imshow(image_label_overlay)
        
        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 80:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                ax1.add_patch(rect)
        
        ax1.set_axis_off()
        
        ax2.imshow(bw)
        #ax3.imshow(cleared)
        plt.tight_layout()
        plt.show()
def bigSegment(imageref,imageseg,pltsh,ignore = False):
    #thresh = threshold_otsu(image)
    bw = closing(imageseg, square(1))    
    label_image = label(bw)
    amax0 = -1
    amax = -1
    pmax = -1
    wmax = -1
    for region in regionprops(label_image):
        r0, c0, r1, c1 = region.bbox
        length = np.max([r1-r0,c1-c0])
        width = np.min([r1-r0,c1-c0])
        if length<=30 or ignore:
            a = length**2
            segment = imageref[r0:r1,c0:c1].astype(int)
            if np.sum(segment) >pmax:
                outputImageP = segment
                pmax = np.sum(segment)
            if region.area > amax:
                outputImageA = segment
                amax = region.area
            if a>amax0 or (a== amax0 and width>wmax):
                outputImageAS = segment
                amax0 = a
                wmax = width

            
        else:
            print("Merged Numbers")
            raise Exception('Merged Numbers')
    if pltsh:     
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
        ax0, ax1, ax2 = axes
        plt.gray()
        
        ax0.imshow(outputImageA)
        ax0.set_title('Biggest Number by Area')
        
        ax1.imshow(outputImageAS)
        ax1.set_title('Biggest Number by Square Area')
        
        ax2.imshow(outputImageP)
        ax2.set_title('Biggest Number by Pixel')
        
        for ax in axes:
            ax.axis('off')
        
        plt.show()
    return outputImageA,outputImageAS,outputImageP
            

    
def differentfilters(image):
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5),
                           sharex=True, sharey=True)
    
    plt.gray()
    
    # Estimate the average noise standard deviation across color channels.
    sigma_est = estimate_sigma(image, multichannel=False, average_sigmas=True)
    # Due to clipping in random_noise, the estimate will be a bit smaller than the
    # specified sigma.
    print("Estimated Gaussian noise standard deviation = {}".format(sigma_est))
    
    
    ax[0, 0].imshow(image)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Noisy')
    ax[0, 1].imshow(denoise_tv_chambolle(image, weight=0.1, multichannel=False))
    ax[0, 1].axis('off')
    ax[0, 1].set_title('TV')
    ax[0, 2].imshow(denoise_bilateral(image, sigma_color=0.05, sigma_spatial=15,
                    multichannel=False))
    ax[0, 2].axis('off')
    ax[0, 2].set_title('Bilateral')
    ax[0, 3].imshow(denoise_wavelet(image, multichannel=False))
    ax[0, 3].axis('off')
    ax[0, 3].set_title('Wavelet denoising')
    
    ax[1, 0].imshow(denoise_tv_chambolle(image, weight=0.2, multichannel=False))
    ax[1, 0].axis('off')
    ax[1, 0].set_title('(more) TV')
    ax[1, 1].imshow(denoise_bilateral(image, sigma_color=0.1, sigma_spatial=15,
                    multichannel=False))
    ax[1, 1].axis('off')
    ax[1, 1].set_title('(more) Bilateral')
    ax[1, 2].imshow(denoise_wavelet(image, multichannel=False, convert2ycbcr=True))
    ax[1, 2].axis('off')
    ax[1, 2].set_title('Wavelet denoising\nin YCbCr colorspace')
    #ax[1, 0].imshow(original)
    #ax[1, 0].axis('off')
    #ax[1, 0].set_title('Original')
    
    fig.tight_layout()
    
    plt.show()
    
def preprocess(image,erode = 0):
    global_thresh = 254;
    pltShow = True
    imth = showthreshold(image,global_thresh,pltShow)
    #differentfilters(imageth)
    immed = medianfilter(imth,pltShow)
    Cannyfilter(median(imth, disk(1)),pltShow)
    bwlabeling(median(imth, disk(1)),True)
    imforbig = immed
    if erode == 0:
        imA,imAS,imP = bigSegment(imforbig,median(imth, disk(1)),True)
    elif erode == -1:
        imA,imAS,imP = bigSegment(imforbig,median(imth, disk(1)),True,ignore = True)
    else:
        bwlabeling(erosion(imforbig, disk(erode)),True)
        imA,imAS,imP = bigSegment(imforbig,erosion(imth,disk(erode)),True)
            
    return imA,imAS,imP,imth

def normalizeIm(im,size = 32):
    im = im.astype(int)
    isize = im.shape
    hs = int(size/2) #half size for centering
    w = int(isize[0]/2)
    h = int(isize[1]/2)
    if isize[0]>size:
        im = im[w-hs:w+hs,:]
        isize = im.shape
        w = int(isize[0]/2)
        
    if isize[1]>size:
        im = im[:,h-hs:h+hs]
        isize = im.shape
        h = int(isize[1]/2)
            
    rw = int(isize[0] %2)
    rh = int(isize[1] %2)
    
    out = np.zeros((size,size),dtype=int)
    out[hs-w:hs+w+rw,hs-h:hs+h+rh] = im
    
    return out   
    
#######################
if not 'trainX' in globals():
    trainX = pd.read_csv(trainXPath, header=None).values
    #trainX = np.loadtxt(trainXPath,delimiter=",") # load from text 
    trainX = trainX.reshape(-1, 64, 64) # reshape 
if not 'trainY' in globals():
    trainY = pd.read_csv(trainYPath, header=None).values
    #trainY = np.loadtxt(trainYPath, delimiter=",") 
    trainY = trainY.reshape(-1, 1) 


if __name__ == '__main__':
    smax = 0
    #nvec = [42802,25203,27981,25616,12608,48329,45209,26742,46419]
    nvec = np.random.randint(0,high=trainX.shape[0],size = 1) 
    bigNumOut = np.zeros((len(nvec),32,32),dtype=int)

    for i in range(len(nvec)):
        #n = np.random.randint(0,high=trainX.shape[0])
        n = nvec[i]
        print('For n = ',n)
        image = trainX[n]
        plt.imshow(image)
        try:
            bigNumImA,bigNumImAS,bigNumImP,imageth= preprocess(image)
        except:
            try:
                bigNumImA,bigNumImAS,bigNumImP,imageth= preprocess(image,erode =1)
            except:
                #try:
                    #bigNumImA,bigNumImaAS,bigNumImP,imageth= preprocess(image,erode =2)
                #except:
                bigNumImA,bigNumImAS,bigNumImP,imageth= preprocess(image,erode = -1)
        bigNumOut[i,:,:] = normalizeIm(bigNumImAS)
        sA = np.max([bigNumImA.shape,bigNumImAS.shape,bigNumImP.shape])
        if sA>smax :
            smax = sA
            print ('max Dimension is: ',smax)
        print ('Ouput is: ',trainY[n])
        

