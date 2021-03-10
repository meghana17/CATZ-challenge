import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.ndimage import imread
from glob import glob
import os
import random
import cv2

import global_constants as c
from utils import log10

def normalize_frames(frames):
    new_frames = frames.astype(np.float32)
    new_frames /= (255 / 2)
    new_frames -= 1

    return new_frames

def denormalize_frames(frames):
    new_frames = frames + 1
    new_frames *= (255 / 2)
    # noinspection PyUnresolvedReferences
    new_frames = new_frames.astype(np.uint8)

    return new_frames

def clip_l2_diff(clip):
    diff = 0
    for i in range(c.HIST_LEN):
        frame = clip[:, :, 3 * i:3 * (i + 1)]
        next_frame = clip[:, :, 3 * (i + 1):3 * (i + 2)]
        # noinspection PyTypeChecker
        diff += np.sum(np.square(next_frame - frame))

    return diff


class data():
    def __init__(self, path):
        # Set up image dirs
        cat_dirs = glob(path + "*")
        random.shuffle(cat_dirs)
        
        # load all images
        self.images = np.zeros(
            (len(cat_dirs), c.FULL_HEIGHT, c.FULL_WIDTH, 3 * (c.HIST_LEN + 1)))
        for i in range(0, len(cat_dirs)):
            input_imgs = glob(cat_dirs[i] + "/cat_*")
            imgs = [imread(img, mode='RGB') for img in sorted(input_imgs)]
            self.images[i] = normalize_frames(np.concatenate(imgs, axis=2))
        
        self.instances = len(self.images)
        
        self.mode = 'test'
        if 'train' in path:
            self.mode = 'train'
        
        self.i = 0
        
    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.instances

        if self.i >= self.instances:
            self.i = 0
            np.random.shuffle(self.images)
    
        i = self.i
        self.i += batch_size
        
        batch = np.take(self.images, range(i, i+batch_size), axis=0, mode='wrap')
        
        if self.mode == 'train':
            indices = np.random.randint(0, batch_size, (batch_size//3))
            batch[indices] = np.fliplr(batch[indices])
            indices = np.random.randint(0, batch_size, (batch_size//3))
            lcrops = np.random.randint(
                        0, 
                        int(c.FULL_HEIGHT*0.2), 
                        (batch_size//2))
            hcrops = np.random.randint(
                        int(c.FULL_HEIGHT*0.8), 
                        c.FULL_HEIGHT, 
                        (batch_size//2))
            for x, i in enumerate(indices):
                new = batch[i,
                           lcrops[x]:hcrops[x],
                           lcrops[x]:hcrops[x],
                           :]
                new = cv2.resize(new, (c.FULL_HEIGHT, c.FULL_WIDTH)).copy()
                if np.amax(new) > 1 or np.amin(new) < -1: # If we get a bad image, discard
                    continue
                batch[i] = new
            
        return batch

def perceptual_distance(gen_frames, gt_frames):
    y_pred = gen_frames + 1
    y_true = gt_frames + 1
    y_pred *= (255 / 2)
    y_true *= (255 / 2)

    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
    
    return tf.reduce_mean(tf.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

def psnr_error(gen_frames, gt_frames):
    shape = tf.shape(gen_frames)
    num_pixels = tf.to_float(shape[1] * shape[2] * shape[3])
    square_diff = tf.square(gt_frames - gen_frames)

    batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.reduce_sum(square_diff, [1, 2, 3])))
    return tf.reduce_mean(batch_errors)

def sharp_diff_error(gen_frames, gt_frames):
    shape = tf.shape(gen_frames)
    num_pixels = tf.to_float(shape[1] * shape[2] * shape[3])
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

    gen_grad_sum = gen_dx + gen_dy
    gt_grad_sum = gt_dx + gt_dy

    grad_diff = tf.abs(gt_grad_sum - gen_grad_sum)

    batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.reduce_sum(grad_diff, [1, 2, 3])))
    return tf.reduce_mean(batch_errors)
    
def w(shape, stddev=0.01):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


def b(shape, const=0.1):
    return tf.Variable(tf.constant(const, shape=shape))


def conv_out_size(i, p, k, s):
    if p == 'SAME':
        p = k // 2
    elif p == 'VALID':
        p = 0
    else:
        raise ValueError('p must be "SAME" or "VALID".')

    return int(((i + (2 * p) - k) / s) + 1)


def log10(t):
    numerator = tf.log(t)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def batch_pad_to_bounding_box(images, offset_height, offset_width, target_height, target_width):
    batch_size, height, width, channels = tf.Session().run(tf.shape(images))

    if not offset_height >= 0:
        raise ValueError('offset_height must be >= 0')
    if not offset_width >= 0:
        raise ValueError('offset_width must be >= 0')
    if not target_height >= height + offset_height:
        raise ValueError('target_height must be >= height + offset_height')
    if not target_width >= width + offset_width:
        raise ValueError('target_width must be >= width + offset_width')

    num_tpad = offset_height
    num_lpad = offset_width
    num_bpad = target_height - (height + offset_height)
    num_rpad = target_width - (width + offset_width)

    tpad = np.zeros([batch_size, num_tpad, width, channels])
    bpad = np.zeros([batch_size, num_bpad, width, channels])
    lpad = np.zeros([batch_size, target_height, num_lpad, channels])
    rpad = np.zeros([batch_size, target_height, num_rpad, channels])

    padded = images
    if num_tpad > 0 and num_bpad > 0: padded = tf.concat(1, [tpad, padded, bpad])
    elif num_tpad > 0: padded = tf.concat(1, [tpad, padded])
    elif num_bpad > 0: padded = tf.concat(1, [padded, bpad])
    if num_lpad > 0 and num_rpad > 0: padded = tf.concat(2, [lpad, padded, rpad])
    elif num_lpad > 0: padded = tf.concat(2, [lpad, padded])
    elif num_rpad > 0: padded = tf.concat(2, [padded, rpad])

    return padded


def batch_crop_to_bounding_box(images, offset_height, offset_width, target_height, target_width):
    batch_size, height, width, channels = tf.Session().run(tf.shape(images))

    if not offset_height >= 0:
        raise ValueError('offset_height must be >= 0')
    if not offset_width >= 0:
        raise ValueError('offset_width must be >= 0')
    if not target_height + offset_height <= height:
        raise ValueError('target_height + offset_height must be <= height')
    if not target_width <= width - offset_width:
        raise ValueError('target_width + offset_width must be <= width')

    top = offset_height
    bottom = target_height + offset_height
    left = offset_width
    right = target_width + offset_width

    return images[:, top:bottom, left:right, :]
