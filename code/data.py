import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

DATA_PATH = '../data/'

DATASET_DICT = {
    'threshold': DATA_PATH + 'thresholdmed/',
    'og': DATA_PATH + 'og/',
    'big': DATA_PATH+'big/'
}

from skimage.measure import label, regionprops
from skimage.morphology import disk, closing, square, erosion
from skimage.filters.rank import median
from skimage.transform import rotate

LENGTHMAX = 30 # Expected MNIST image sizes
DISKSIZE = 1 # for median filter
OUTPUTSIZE = 28 # Size of data

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

def load_dataset(name):
    dataset_path = DATASET_DICT[name]
    x_train = load_array(dataset_path + 'train_x.csv')
    y_train = load_array(dataset_path + 'train_y.csv')
    x_valid = load_array(dataset_path + 'valid_x.csv')
    y_valid = load_array(dataset_path + 'valid_y.csv')
    return x_train, y_train, x_valid, y_valid
    


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
# Threshold filter
THRESHOLD = 230
THRESHOLD_DIR = 'thresholded/'

def threshold_filter(images):
    return (images > THRESHOLD).astype(float)

def create_threshold_dataset():
    print('Reading old data')
    og_x = pd.read_csv(DATA_PATH+'og/train_x.csv', header=None).values
    og_y = pd.read_csv(DATA_PATH+'og/train_y.csv', header=None).values
    
    print('Applying filter')
    og_x = threshold_filter(og_x)
    
    print('Shuffling')
    state = np.random.get_state()
    np.random.shuffle(og_x)
    np.random.set_state(state)
    np.random.shuffle(og_y)

    print('Splitting')
    valid_x = og_x[:1000, :]
    valid_y = og_y[:1000, :]

    print('Saving validation set')
    save_array(valid_x, DATA_PATH+THRESHOLD_DIR+'valid_x.csv')
    save_array(valid_y, DATA_PATH+THRESHOLD_DIR+'valid_y.csv')
    
    train_x = og_x[1000:, :]
    train_y = og_y[1000:, :]

    print('Saving train set')
    save_array(train_x, DATA_PATH+THRESHOLD_DIR+'train_x.csv')
    save_array(train_y, DATA_PATH+THRESHOLD_DIR+'train_y.csv')

    print('Loading og test set')
    og_x = load_array(DATA_PATH+'test_x.csv')
    og_x = threshold_filter(og_x)
    save_array(og_x, DATA_PATH+THRESHOLD_DIR+'test_x.csv')
    


def findBiggest(imth):
    imth = imth.reshape(64,64)
    #immed = medianfilter(imth,1)
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

def biggest(imth,erode = 0):
    if erode == 0:
        im = bigSegment(imth,median(imth, disk(DISKSIZE)))
    elif erode == -1:
        im = bigSegment(imth,median(imth, disk(DISKSIZE)),ignore = True)
    else:
        im = bigSegment(imth,erosion(imth,disk(erode)))  
    return im

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
            a = length*length
            segment = imageref[r0:r1,c0:c1]
            if a>amax or (a== amax and width>wmax):
                outIm = segment
                amax = a
                wmax = width            
        else:
            raise Exception('Merged Numbers')
    return outIm

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


def rotate_img(img):
    rots = []
    rots.append(rotate(img.reshape((64, 64)), 27, preserve_range=True))
    rots.append(rotate(img.reshape((64, 64)), -27, preserve_range=True))

    #return rots
    return [(r > 0.75).astype(int).reshape((1, 64**2)) for r in rots]


BIG_MED_DIR = 'augmented_big/'
AUG_TMED_DIR = 'augmented_tmed/'


def augment_and_biggest():
    dataset_path = DATASET_DICT['og']
    
    print('Loading values')
    x = pd.read_csv(dataset_path+'train_x.csv', header=None).values
    y = pd.read_csv(dataset_path+'train_y.csv', header=None).values
    #x_valid = pd.read_csv(dataset_path+'valid_x.csv', header=None).values
    #y_valid = pd.read_csv(dataset_path+'valid_y.csv', header=None).values

    #x = np.append(x, x_valid)
    #y = np.append(y, y_valid)
    
    #augmented_x = np.zeros((x.shape[0]*3, 64**2))
    #augmented_y = np.zeros((y.shape[0]*3, y.shape[1]))
    x = threshold_filter(x) 
    big_x = np.zeros((x.shape[0], OUTPUTSIZE*OUTPUTSIZE))
    
    for i in range(x.shape[0]):
        if (i+1)%50 == 0:
            print('{:5d} / {:5d}\r'.format(i+1, x.shape[0]), end='')
        
        start_idx = i
        #img = x[i]
        img = (findBiggest(x[i])).reshape((1, OUTPUTSIZE**2))
        #rotations = rotate_img(img)
        big_x[start_idx] = img
        #augmented_x[start_idx+1] = rotations[0]
        #augmented_x[start_idx+2] = rotations[1]
        
        #augmented_y[start_idx] = y[i]
        #augmented_y[start_idx+1] = y[i]
        #augmented_y[start_idx+2] = y[i]
    print('')
    
    
    print('Shuffling')
    state = np.random.get_state()
    np.random.shuffle(big_x)
    np.random.set_state(state)
    np.random.shuffle(y)

    print('Splitting')
    valid_x = big_x[:2000, :]
    valid_y = y[:2000, :]

    print('Saving validation set')
    save_array(valid_x, DATA_PATH+'big/'+'valid_x.csv')
    save_array(valid_y, DATA_PATH+'big/'+'valid_y.csv')
    
    train_x = big_x[2000:, :]
    train_y = y[2000:, :]

    print('Saving train set')
    save_array(train_x, DATA_PATH+'big/'+'train_x.csv')
    save_array(train_y, DATA_PATH+'big/'+'train_y.csv')
    
    # do the same for the test set
    #x = pd.read_csv(dataset_path+'test_x.csv', header=None).values
    #augmented_x = np.zeros((x_test.shape[0]*3), x_test.shape[1])
    #for i in range(x.shape[0]):
    #    if i%50 == 0:
    #        print('{:5d} / {:5d}\r'.format(i, x.shape[0]), end='')
    #    start_idx = i*3
    #    augmented_x[start_idx] = x[i]
    #    img = (findBiggest(x[i])).reshape((1, 64**2))
    #    rotations = rotate_img(img)
    #    augmented_x[start_idx+1] = rotations[0]
    #    augmented_x[start_idx+2] = rotations[1]
        
    #    augmented_y[start_idx] = y[i]
    #    augmented_y[start_idx+1] = y[i]
    #    augmented_y[start_idx+2] = y[i]
    #print('')
    #save_array(augmented_x, DATA_PATH+AUG_TMED_DIR+'test_x.csv')


#==============================
def one_hot(arr):
    enc = OneHotEncoder()
    return enc.fit_transform(arr).toarray()


#==============================

if __name__ == '__main__':
    augment_and_biggest()
    
    #train_x = load_array(DATA_PATH+AUG_TMED_DIR+'train_x.csv')
    #train_y = load_array(DATA_PATH+AUG_TMED_DIR+'train_y.csv')

    #for i in range(50):
    #    show_random_image(train_x, train_y, dim=64)
