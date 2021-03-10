import tensorflow as tf
import numpy as np

from utils import perceptual_distance, log10
import global_constants as c
 

def combined_loss(gen_frames, gt_frames, d_preds, lam_adv=c.LAM_ADV, lam_lp=c.LAM_LP, lam_gdl=c.LAM_GDL, l_num=c.L_NUM, alpha=c.ALPHA_NUM):
    batch_size = tf.shape(gen_frames[0])[0]  # variable batch size as a tensor

    loss = lam_lp * lp_loss(gen_frames, gt_frames, l_num)
    loss += lam_gdl * gdl_loss(gen_frames, gt_frames, alpha)
    if c.ADVERSARIAL: loss += lam_adv * adv_loss(d_preds, tf.ones([batch_size, 1]))
    pd_loss = perceptual_distance(gen_frames[-1], gt_frames[-1])
    loss += 1000*pd_loss

    return loss


def bce_loss(preds, targets):
    return tf.squeeze(-1 * (tf.matmul(targets, log10(preds), transpose_a=True) +
                            tf.matmul(1 - targets, log10(1 - preds), transpose_a=True)))


def lp_loss(gen_frames, gt_frames, l_num):
    scale_losses = []
    for i in range(len(gen_frames)):
        scale_losses.append(tf.reduce_sum(tf.abs(gen_frames[i] - gt_frames[i])**l_num))

    return tf.reduce_mean(tf.stack(scale_losses))


def gdl_loss(gen_frames, gt_frames, alpha):
    scale_losses = []
    for i in range(len(gen_frames)):
        pos = tf.constant(np.identity(3), dtype=tf.float32)
        neg = -1 * pos
        filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
        filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
        strides = [1, 1, 1, 1]  # stride of (1, 1)
        padding = 'SAME'

        gen_dx = tf.abs(tf.nn.conv2d(gen_frames[i], filter_x, strides, padding=padding))
        gen_dy = tf.abs(tf.nn.conv2d(gen_frames[i], filter_y, strides, padding=padding))
        gt_dx = tf.abs(tf.nn.conv2d(gt_frames[i], filter_x, strides, padding=padding))
        gt_dy = tf.abs(tf.nn.conv2d(gt_frames[i], filter_y, strides, padding=padding))

        grad_diff_x = tf.abs(gt_dx - gen_dx)
        grad_diff_y = tf.abs(gt_dy - gen_dy)

        scale_losses.append(tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha)))
        
    return tf.reduce_mean(tf.stack(scale_losses))


def adv_loss(preds, labels):
    scale_losses = []
    for i in range(len(preds)):
        loss = bce_loss(preds[i], labels)
        scale_losses.append(loss)

    return tf.reduce_mean(tf.stack(scale_losses))
