import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = '../data/'

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
    

#==============================
#==============================

if __name__ == '__main__':
    create_threshold_dataset()
