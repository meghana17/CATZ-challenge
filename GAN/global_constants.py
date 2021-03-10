import numpy as np
import os
from glob import glob
import shutil
from scipy.ndimage import imread


def get_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def clear_dir(directory):
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)

def get_test_frame_dims():
    img_path = glob(os.path.join(TEST_DIR, '*/*'))[0]
    img = imread(img_path, mode='RGB')
    shape = np.shape(img)

    return shape[0], shape[1]

def get_train_frame_dims():
    img_path = glob(os.path.join(TRAIN_DIR, '*/*'))[0]
    img = imread(img_path, mode='RGB')
    shape = np.shape(img)

    return shape[0], shape[1]

def set_test_dir(directory):
    global TEST_DIR, FULL_HEIGHT, FULL_WIDTH

    TEST_DIR = directory
    FULL_HEIGHT, FULL_WIDTH = get_test_frame_dims()

def set_save_name(name):
    global SAVE_NAME, MODEL_SAVE_DIR, SUMMARY_SAVE_DIR, IMG_SAVE_DIR

    SAVE_NAME = name
    MODEL_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Models/', SAVE_NAME))
    SUMMARY_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Summaries/', SAVE_NAME))
    IMG_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Images/', SAVE_NAME))

def clear_save_name():
    clear_dir(MODEL_SAVE_DIR)
    clear_dir(SUMMARY_SAVE_DIR)
    clear_dir(IMG_SAVE_DIR)


DATA_DIR         = get_dir('../Data/')
TRAIN_DIR        = os.path.join(DATA_DIR, 'catz/train/')
TEST_DIR         = os.path.join(DATA_DIR, 'catz/test/')
SAVE_DIR         = get_dir('../Save/')
SAVE_NAME        = 'Default/'
MODEL_SAVE_DIR   = get_dir(os.path.join(SAVE_DIR, 'Models/', SAVE_NAME))
SUMMARY_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Summaries/', SAVE_NAME))
IMG_SAVE_DIR     = get_dir(os.path.join(SAVE_DIR, 'Images/', SAVE_NAME))

FULL_HEIGHT     = 96
FULL_WIDTH      = 96
TRAIN_HEIGHT    = 96
TRAIN_WIDTH     = 96
STATS_FREQ      = 10     
SUMMARY_FREQ    = 1000    
IMG_SAVE_FREQ   = 1000   
TEST_FREQ       = 1000   
MODEL_SAVE_FREQ = 10000  

ADVERSARIAL = True
BATCH_SIZE = 32
HIST_LEN = 5
L_NUM = 2
ALPHA_NUM = 1
LAM_ADV = 0
LAM_LP = 0.1
LAM_GDL = 0.1

# Generator model
LRATE_G = 0.0001  # Value in paper is 0.04
LEPSILON_G = 1e-1
PADDING_G = 'SAME'

# feature maps for each convolution of each scale network in the generator model
SCALE_FMS_G = [[3 * HIST_LEN, 128, 256, 128, 3],
               [3 * (HIST_LEN + 1), 128, 256, 128, 3],
               [3 * (HIST_LEN + 1), 128, 256, 512, 256, 128, 3],
               [3 * (HIST_LEN + 1), 128, 256, 512, 256, 128, 3]]
# kernel sizes for each convolution of each scale network in the generator model
SCALE_KERNEL_SIZES_G = [[3, 3, 3, 3],
                        [5, 3, 3, 5],
                        [5, 3, 3, 3, 3, 5],
                        [7, 5, 5, 5, 5, 7]]

# Discriminator model
LRATE_D = 0.02
PADDING_D = 'VALID'

# feature maps for each convolution of each scale network in the discriminator model
SCALE_CONV_FMS_D = [[3, 64],
                    [3, 64, 128, 128],
                    [3, 128, 256, 256],
                    [3, 128, 256, 512, 128]]
# kernel sizes for each convolution of each scale network in the discriminator model
SCALE_KERNEL_SIZES_D = [[3],
                        [3, 3, 3],
                        [5, 5, 5],
                        [7, 7, 5, 5]]
# layer sizes for each fully-connected layer of each scale network in the discriminator model
# layer connecting conv to fully-connected is dynamically generated when creating the model
SCALE_FC_LAYER_SIZES_D = [[512, 256, 1],
                          [1024, 512, 1],
                          [1024, 512, 1],
                          [1024, 512, 1]]
