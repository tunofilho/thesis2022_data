# custom loss and metrics
from tensorflow.keras import backend as K
import tensorflow as tf

# default rates
alpha = 0.3
# theta = 0.3

# credits: https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
def generalized_dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - generalized_dice_coefficient(y_true, y_pred)
    return loss

def confusion(y_true, y_pred):
    smooth = 1.
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    prec = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return prec, recall

def precision(y_true, y_pred):
    precision, recall = confusion(y_true, y_pred)
    return precision

def recall(y_true, y_pred):
    precision, recall = confusion(y_true, y_pred)
    return recall

def f1(y_true, y_pred):
    precision, recall = confusion(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    loss = bce(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

# credits: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Tversky-Loss

def tversky(y_true, y_pred, alpha=alpha, smooth=1e-6):
    # flatten label and prediction tensors
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)

    # True Positives, False Positives & False Negatives
    TP = K.sum((y_pred * y_true))
    FP = K.sum(((1 - y_true) * y_pred))
    FN = K.sum((y_true * (1 - y_pred)))

    TI = (TP + smooth) / (TP + alpha * FP + (1 - alpha) * FN + smooth)

    return TI

def tversky_loss(y_true, y_pred, alpha=alpha, smooth=1e-6):
    return 1 - tversky(y_true, y_pred, alpha, smooth)

def focal_tversky_loss(y_true, y_pred, alpha=alpha, smooth=1e-6):
    pt_1 = tversky(y_true, y_pred, alpha, smooth)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)

def SSIM_loss(y_true, y_pred):  # feature size min = 11x11
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def bce_focal_tversky_SSIM_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    loss = bce(y_true, y_pred) + focal_tversky_loss(y_true, y_pred) + SSIM_loss(y_true, y_pred)
    return loss

def bce_dice_SSIM_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    loss = bce(y_true, y_pred) + dice_loss(y_true, y_pred) + SSIM_loss(y_true, y_pred)
    return loss

# credits: https://ai-pool.com/d/keras_iou_implementation
def mean_iou(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32')  # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))


import numpy as np
# t = [0.,0.,1.,1.,1.,0.,1.,1.,0.]
# t3 = [0.,0.,1.,1.,0.,0.,1.,1.,1.]
# t = tf.convert_to_tensor(np.array(t).reshape(1,3,3,1),tf.float32)
# t2 = tf.convert_to_tensor(np.array(t).reshape(1,3,3,1),tf.float32)
# t3 = tf.convert_to_tensor(np.array(t3).reshape(1,3,3,1),tf.float32)
# t4 = np.arange(242).reshape(2, 11, 11, 1)
# t5 = t4*2
# t4 = tf.convert_to_tensor(t4/t4.max(), tf.float32)
# t5 = tf.convert_to_tensor(t5/t5.max(), tf.float32)
# print(dice_loss(t,t3))
# print(bce_dice_loss(t4, t5))
# print(bce_dice_SSIM_loss(t4,t5))
# print(bce_focal_tversky_SSIM_loss(t4,t5))
# print(tf.reduce_sum(tf.keras.losses.binary_crossentropy(t4,t5)))
# print(confusion(t,t3))
# print(f1(t,t3))
# print(tversky_loss(t,t3))
# print(focal_tversky_loss(t,t3))
# print(mean_iou(t,t3))
# m = tf.keras.metrics.MeanIoU(2)
# m.update_state(t,t3)
# print(m.result().numpy())
# print(SSIM_loss(t4, t5))
# print(dice_loss(t4, t5))
# bce = tf.keras.losses.BinaryCrossentropy()
# print(bce(t4, t5))


