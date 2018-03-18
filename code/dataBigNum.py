import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


from skimage.measure import label, regionprops
from skimage.morphology import disk,closing,square,erosion
from skimage.filters.rank import median


DATA_PATH = '../data/'
TRAIN_PERCENT = 0.98
# Threshold filter
THRESHOLD = 250
THRESHOLD_DIR = 'thresholded/'

# Biggest Number
MERGED = 0
BIGGEST_DIR = 'biggest/'
LENGTHMAX = 30 # Expected MNIST image sizes
DISKSIZE = 1 # for median filter
OUTPUTSIZE = 32 # Size of data

if not os.path.exists(DATA_PATH+THRESHOLD_DIR):
    os.makedirs(DATA_PATH+THRESHOLD_DIR)
if not os.path.exists(DATA_PATH+BIGGEST_DIR):
    os.makedirs(DATA_PATH+BIGGEST_DIR)
#==============================
# Save and Load arrays
def load_array(fname):
    print('Loading {}'.format(fname))
    a = pd.read_csv(fname, header=None).values
    print('Shape: {}'.format(a.shape))
    return a

def save_array(array, fname):
    print('Saving {}'.format(fname))
    print('Shape: {}'.format(array.shape))
    pd.DataFrame(array).to_csv(fname, header=None, index=False)


#==============================
# Split the og dataset into TRAIN/VALIDATION
def train_valid_split(train_perc=0.9):
    print('Reading old data')
    og_x = pd.read_csv(DATA_PATH+'og/train_x.csv', header=None).values
    og_y = pd.read_csv(DATA_PATH+'og/train_y.csv', header=None).values

    # shuffle data
    print('Shuffling')
    rdm_state = np.random.get_state()
    np.random.shuffle(og_x)
    np.random.set_state(rdm_state)
    np.random.shuffle(og_y)

    num_train = int(train_perc*og_x.shape[0])
    
    print('Saving training data')
    x = og_x[:num_train, :]
    y = og_y[:num_train, :]
    save_array(x, DATA_PATH+'train_valid/train_x.csv')
    save_array(y, DATA_PATH+'train_valid/train_y.csv')

    print('Saving validation data')
    x = og_x[num_train:, :]
    y = og_y[num_train:, :]
    save_array(x, DATA_PATH+'train_valid/valid_x.csv')
    save_array(y, DATA_PATH+'train_valid/valid_y.csv')


# ==============================
# Show images
def show_image(img_as_array, dim=64):
    plt.imshow(img_as_array.reshape((dim, dim)), cmap='gray')
    plt.show()

def show_random_image(imgs_as_array, labels, dim=64):
    rdm_idx = np.random.randint(imgs_as_array.shape[0])
    print('Label: {}'.format(labels[rdm_idx]))
    show_image(imgs_as_array[rdm_idx], dim=dim)


#==============================

def threshold_filter(images):
    return (images > THRESHOLD).astype(int)

def create_threshold_dataset():
    print('Reading old data')    
    og_x = pd.read_csv(DATA_PATH+'og/train_x.csv', header=None).values
    og_y = pd.read_csv(DATA_PATH+'og/train_y.csv', header=None).values
    og_tx = pd.read_csv(DATA_PATH+'og/test_x.csv', header=None).values



    num_train = int(TRAIN_PERCENT*og_x.shape[0])

    print('Applying filter')
    og_x = threshold_filter(og_x)
    og_tx = threshold_filter(og_x)

    
    print('Shuffling')
    state = np.random.get_state()
    np.random.shuffle(og_x)
    np.random.set_state(state)
    np.random.shuffle(og_y)

    print('Splitting')
    valid_x = og_x[num_train:, :]
    valid_y = og_y[num_train:, :]

    print('Saving validation set')
    save_array(valid_x, DATA_PATH+THRESHOLD_DIR+'valid_x.csv')
    save_array(valid_y, DATA_PATH+THRESHOLD_DIR+'valid_y.csv')
    
    train_x = og_x[:num_train, :]
    train_y = og_y[:num_train, :]

    print('Saving train set')
    save_array(train_x, DATA_PATH+THRESHOLD_DIR+'train_x.csv')
    save_array(train_y, DATA_PATH+THRESHOLD_DIR+'train_y.csv')

    print('Saving test set')
    save_array(og_tx, DATA_PATH+THRESHOLD_DIR+'test_x.csv')
    
def normalizeIm(im,size = OUTPUTSIZE):
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
def bigSegment(imageref,imageseg,ignore = False):
    
    bw = closing(imageseg, square(1))    
    label_image = label(bw)
    amax = -1
    wmax = -1
    for region in regionprops(label_image):
        r0, c0, r1, c1 = region.bbox
        length = np.max([r1-r0,c1-c0])
        width = np.min([r1-r0,c1-c0])
        if length<=LENGTHMAX or ignore:
            a = length**2
            segment = imageref[r0:r1,c0:c1]
            if a>amax or (a== amax and width>wmax):
                outIm = segment
                amax = a
                wmax = width            
        else:
            raise Exception('Merged Numbers')
    return outIm
            

def biggest(imth,erode = 0):
    if erode == 0:
        im = bigSegment(imth,median(imth, disk(DISKSIZE)))
    elif erode == -1:
        im = bigSegment(imth,median(imth, disk(DISKSIZE)),ignore = True)
    else:
        im = bigSegment(imth,erosion(imth,disk(erode)))  
    return im
def findBiggest(imth):
    #imth = imth.reshape(64,64)
    try:
        out = biggest(imth)
    except:
        try:
            out = biggest(imth,erode =1)
        except:
            #try:
                #bigNumImA,bigNumImaAS,bigNumImP,imageth= preprocess(image,erode =2)
            #except:
            out = biggest(imth,erode = -1)
    return normalizeIm(out)
def create_biggest_dataset():
    print('Reading old data')
    og_x = pd.read_csv(DATA_PATH+'og/train_x.csv', header=None).values
    og_y = pd.read_csv(DATA_PATH+'og/train_y.csv', header=None).values
    og_tx = pd.read_csv(DATA_PATH+'og/test_x.csv', header=None).values

    og_x = og_x.reshape(-1, 64, 64) # reshape 
    og_tx = og_tx.reshape(-1, 64, 64) # reshape
    og_y = og_y.reshape(-1,1)
    num_train = int(TRAIN_PERCENT*og_x.shape[0])

    print('Creating new dataset')
    big_x = np.zeros((og_x.shape[0],OUTPUTSIZE**2),dtype = int)
    big_tx = np.zeros((og_tx.shape[0],OUTPUTSIZE**2),dtype = int)

    print('Applying threshold filter')
    og_x = threshold_filter(og_x)
    og_tx = threshold_filter(og_tx)

    print('Shuffling')
    state = np.random.get_state()
    np.random.shuffle(og_x)
    np.random.set_state(state)
    np.random.shuffle(og_y)
    
    print('Applying biggest number preprocessing to training set')
    for i in range(og_x.shape[0]):
        big_x[i,:] = findBiggest(og_x[i]).flatten()
        if i % 1000 == 0:
            print (i/og_x.shape[0]*100,"% complete for training set")
    
    print('Applying biggest number preprocessing to test set')
    for i in range(og_tx.shape[0]):
        big_tx[i,:] = findBiggest(og_tx[i]).flatten()
        if i % 1000 == 0:
            print (i/og_tx.shape[0]*100,"% complete for test set")

    print('Splitting')
    valid_x = big_x[num_train:, :]
    valid_y = og_y[num_train:, :]

    print('Saving validation set')
    save_array(valid_x, DATA_PATH+BIGGEST_DIR+'valid_x.csv')
    save_array(valid_y, DATA_PATH+BIGGEST_DIR+'valid_y.csv')
    
    train_x = big_x[:num_train, :]
    train_y = og_y[:num_train, :]

    print('Saving train set')
    save_array(train_x, DATA_PATH+BIGGEST_DIR+'train_x.csv')
    save_array(train_y, DATA_PATH+BIGGEST_DIR+'train_y.csv')

    print('Saving test set')
    save_array(big_tx, DATA_PATH+BIGGEST_DIR+'test_x.csv')
 
#==============================
#==============================

if __name__ == '__main__':
    #train_valid_split(train_perc=TRAIN_PERCENT)
    create_threshold_dataset()
    create_biggest_dataset()
