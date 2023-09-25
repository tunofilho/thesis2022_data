import tensorflow as tf
from tensorflow.keras import Input, layers, Model
from no_fusion.utils_ import conv_block

filt = 64

# number of convolutional blocks
nblock = {
    'scratch16': (2, 3, 3, 3),
    'scratch19': (2, 4, 4, 4),
}

# build vgg encoder without pretrained weights
def build_encoder(hiper):
    y = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3))
    x = conv_block(y, hiper, filt, 3, 'blk1')
    x = conv_block(x, hiper, filt, 3, 'blk2')
    x = layers.MaxPooling2D((2, 2), name='blk2_pool')(x)

    for _ in range(nblock[hiper.backbone][0]):
        x = conv_block(x, hiper, filt * 2, 3, 'blk3' + str(_))
    x = layers.MaxPooling2D((2, 2), name='blk3_pool')(x)

    for _ in range(nblock[hiper.backbone][1]):
        x = conv_block(x, hiper, filt * 4, 3, 'blk4' + str(_))
    x = layers.MaxPooling2D((2, 2), name='blk4_pool')(x)

    for _ in range(nblock[hiper.backbone][2]):
        x = conv_block(x, hiper, filt * 8, 3, 'blk5' + str(_))
    x = layers.MaxPooling2D((2, 2), name='blk5_pool')(x)

    for _ in range(nblock[hiper.backbone][3]):
        x = conv_block(x, hiper, filt * 8, 3, 'blk6' + str(_))

    return Model(inputs=y, outputs=x)

def vgg(hiper):
    if hiper.finetuning:
        weights = 'imagenet'
        train = False
    else:
        weights = None
        train = True

    if hiper.backbone == 'vgg16':
        base = tf.keras.applications.VGG16(include_top=False, weights=weights,
                                           input_shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3))
        base.trainable=train

    if hiper.backbone == 'vgg19':
        base = tf.keras.applications.VGG19(include_top=False, weights=weights,
                                           input_shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3))
        base.trainable=train

    return base

def vgg_BN(hiper):
    encoder = build_encoder(hiper)
    return encoder
