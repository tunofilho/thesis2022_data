# custom loss
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
import tensorflow as tf

# default rates
alpha = 0.3
# theta = 0.3
beta = 0.3

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

# credits: https://github.com/jocpae/clDice/blob/master/cldice_loss/keras/soft_skeleton.py
def soft_erode(img):
    """[This function performs soft-erosion operation on a float32 image]
    Args:
        img ([float32]): [image to be soft eroded]
    Returns:
        [float32]: [the eroded image]
    """
    p1 = -KL.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', data_format=None)(-img)
    p2 = -KL.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', data_format=None)(-img)

    return tf.math.minimum(p1, p2)


def soft_dilate(img):
    """[This function performs soft-dilation operation on a float32 image]
    Args:
        img ([float32]): [image to be soft dialated]
    Returns:
        [float32]: [the dialated image]
    """
    return KL.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', data_format=None)(img)


def soft_open(img):
    """[This function performs soft-open operation on a float32 image]
    Args:
        img ([float32]): [image to be soft opened]
    Returns:
        [float32]: [image after soft-open]
    """
    img = soft_erode(img)
    img = soft_dilate(img)
    return img


def soft_skel(img, iters):
    """[summary]
    Args:
        img ([float32]): [description]
        iters ([int]): [description]
    Returns:
        [float32]: [description]
    """
    img1 = soft_open(img)
    skel = tf.nn.relu(img - img1)

    for j in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = tf.nn.relu(img - img1)
        intersect = tf.math.multiply(skel, delta)
        skel += tf.nn.relu(delta - intersect)
    return skel


def soft_clDice_loss(y_true, y_pred, iter_=50):
    """[function to compute dice loss]
    Args:
        iter_ (int, optional): [skeletonization iteration]. Defaults to 50.
    """

    # def loss(y_true, y_pred):
    #     """[function to compute dice loss]
    #     Args:
    #         y_true ([float32]): [ground truth image]
    #         y_pred ([float32]): [predicted image]
    #     Returns:
    #         [float32]: [loss value]
    #     """
    smooth = 1.
    skel_pred = soft_skel(y_pred, iter_)
    skel_true = soft_skel(y_true, iter_)
    pres = (K.sum(tf.math.multiply(skel_pred, y_true)[:, 1:, :, :]) + smooth) / (
                K.sum(skel_pred[:, 1:, :, :]) + smooth)
    rec = (K.sum(tf.math.multiply(skel_true, y_pred)[:, 1:, :, :]) + smooth) / (
                K.sum(skel_true[:, 1:, :, :]) + smooth)
    cl_dice = 1. - 2.0 * (pres * rec) / (pres + rec)
    return cl_dice

    # return loss


def soft_dice_cldice_loss(y_true, y_pred, iters=50, beta=beta):
    """[function to compute dice+cldice loss]
    Args:
        iters (int, optional): [skeletonization iteration]. Defaults to 15.
        beta (float, optional): [weight for the cldice component]. Defaults to 0.3.
    """

    # def loss(y_true, y_pred):
    #     """[summary]
    #     Args:
    #         y_true ([float32]): [ground truth image]
    #         y_pred ([float32]): [predicted image]
    #     Returns:
    #         [float32]: [loss value]
    #     """
    smooth = 1.
    skel_pred = soft_skel(y_pred, iters)
    skel_true = soft_skel(y_true, iters)
    pres = (K.sum(K.flatten(tf.math.multiply(skel_pred, y_true))) + smooth) / (K.sum(K.flatten(skel_pred)) + smooth)
    rec = (K.sum(K.flatten(tf.math.multiply(skel_true, y_pred))) + smooth) / (K.sum(K.flatten(skel_true)) + smooth)
    cl_dice = 1. - 2.0 * (pres * rec) / (pres + rec)

    dice = dice_loss(y_true, y_pred)
    loss_cl = (1.0 - beta) * dice + beta * cl_dice
    return loss_cl

    # return loss
