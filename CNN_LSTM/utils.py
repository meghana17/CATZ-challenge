from __future__ import print_function
import os
import subprocess
import random
import glob
import numpy as np
import tensorflow as tf
np.random.seed(1)
tf.set_random_seed(2)
from keras.models import Sequential,Model
from keras.callbacks import Callback
import random
import glob
import subprocess
import os
from PIL import Image
from keras import backend as K
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Conv2D 
from keras.layers import ConvLSTM2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Input,merge,Add
from keras import optimizers
from keras.callbacks import LearningRateScheduler
import math
import cv2
import imgaug.augmenters as iaa
from keras.layers import Dropout
from keras.layers.merge import concatenate
import collections  

if not os.path.exists("catz"):
    print("Downloading catz dataset...")
    subprocess.check_output(
        "curl https://storage.googleapis.com/wandb/catz.tar.gz | tar xz", shell=True)


width =96
height=96 


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def normalized(rgb):
        norm_rgb=rgb.astype('float32')/255.0
        return norm_rgb

def denormalize(img):
   return (img*255.0)
  
batch_size =128

def cal_dist(cat_dirs,counter):
        la_input_images = np.zeros((width, height,3 ),dtype='float32')
        output_images   = np.zeros((width, height, 3),dtype='float32') 
        la_input_images = normalized(np.array(Image.open(cat_dirs[counter] + "/cat_4.jpg")))
        output_images =  normalized(np.array(Image.open(cat_dirs[counter] + "/cat_result.jpg")))
        percep = perceptual_distance_initial(output_images, la_input_images)      
        return (percep,cat_dirs[counter])
        
       
def perceptual_distance_initial(y_true, y_pred): 
    #print(y_true.shape,y_pred.shape)
    y_true = denormalize(y_true)
    y_pred = denormalize(y_pred)
    rmean = (y_true[:, :, 0] + y_pred[:, :, 0]) / 2
    r = y_true[:, :, 0] - y_pred[:, :, 0]
    g = y_true[:, :, 1] - y_pred[:, :, 1]
    b = y_true[:, :, 2] - y_pred[:, :, 2]

    return np.mean(np.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))
def calculate(img_dir):
  percep_dic ={}
  co =0
  for i in range(len(glob.glob(img_dir + "/*"))): 
    cat_dirs = glob.glob(img_dir + "/*")
    value,key =cal_dist(cat_dirs,co)
    percep_dic[key] = value
    co +=1 
    #print(key,percep_dic[key])
  percep_sort = collections.OrderedDict(sorted(percep_dic.items(), key=lambda x: x[1]))  
  return percep_sort
  
def list_dir(img_dir):
  list_direc =[]
  for i,j in calculate(img_dir).items():
   list_direc.append(i)
  return list_direc
  
  
val_dir ='catz/test'
val_list = list_dir(val_dir) 

"""
train_dir ='catz/train'
train_list1 =list_dir(train_dir)
"""

def gdl_loss(gen_frames, gt_frames, alpha=1):
    """
    Calculates the sum of GDL losses between the predicted and ground truth frames.
    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param alpha: The power to which each gradient term is raised.
    @return: The GDL loss.
    """
    print("alpha",alpha)
    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    pos = tf.constant(np.identity(3), dtype=tf.float32)
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(gen_frames, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(gen_frames, filter_y, strides, padding=padding))
    gt_dx = tf.abs(tf.nn.conv2d(gt_frames, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(gt_frames, filter_y, strides, padding=padding))

    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y = tf.abs(gt_dy - gen_dy)
    print(grad_diff_x.get_shape())
    out =((grad_diff_x ** alpha + grad_diff_y ** alpha))
    print("SHAPE_OF_GDL_LOSS",tf.reduce_sum(out,axis=-1).get_shape())

    # condense into one tensor and avg
    return tf.reduce_sum(out,axis=-1)




def mean_absolute_error(y_true, y_pred):
     print("y_true",y_true.shape)
     print("y_pred",y_pred.shape)
     print("SHAPE_OF_LOSS",K.int_shape(K.mean(K.abs(y_pred - y_true), axis=-1)))
     return K.mean(K.abs(y_pred - y_true), axis=-1)
 


 
def loss(y_true,y_pred):
      tot_loss = gdl_loss(y_true, y_pred,alpha=1) +mean_absolute_error(y_true, y_pred)
      return   tot_loss
   

