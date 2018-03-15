import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = '../data/'

#==============================
# Save and Load arrays
def load_array(fname):
    return pd.read_csv(fname, header=None).values

def save_array(array, fname):
    pd.DataFrame(array).to_csv(fname, header=None, index=False)


#==============================
# Split the og dataset into TRAIN/VALIDATION
def train_valid_split(train_perc=0.8):
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
    save_array(x, DATA_PATH+'train_x.csv')
    save_array(y, DATA_PATH+'train_y.csv')

    print('Saving validation data')
    x = og_x[num_train:, :]
    y = og_y[num_train:, :]
    save_array(x, DATA_PATH+'valid_x.csv')
    save_array(y, DATA_PATH+'valid_y.csv')


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
    return images > THRESHOLD

def create_threshold_dataset():
    for fname in ['valid_x.csv', 'train_x.csv']:
        print('Transforming {}'.format(fname))
        arr = load_array(DATA_PATH+fname)
        arr = threshold_filter(arr)
        save_array(arr, DATA_PATH+THRESHOLD_DIR+fname)



#==============================
#==============================

if __name__ == '__main__':
    create_threshold_dataset()
    
    #train_valid_split()
    #for f in ['train_x.csv', 'train_y.csv', 'valid_x.csv', 'valid_y.csv']:
    #    a = load_array(DATA_PATH+f)
    #    print('{} : {}'.format(f, a.shape))

