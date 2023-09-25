import os, tempfile
import tensorflow as tf

def load_pretrained_weights(encoder_ : tf.keras.Model, hiper, weights_path=None):

    # if imagenet_weights:
    aux = []
    if hiper.backbone == 'vgg16':
        mod = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                                          input_shape=[hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3])
    elif hiper.backbone == 'resnet50':
        mod = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',
                                             input_shape=[hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3])
    else:
        raise ValueError('backbone error')

    for layer in mod.layers:
        if len(layer.weights) != 0:  # avoid layers without weights
            aux.append(layer.name)

    for i in encoder_.layers:
        if i.name in aux:
            i.set_weights(mod.get_layer(i.name).get_weights())  # rgb branch
            i.trainable = False
        if i.name[1::] in aux:
            i.set_weights(mod.get_layer(i.name[1::]).get_weights())  # depth branch  (depth layer name without 'd' is equal to rgb layer name
            i.trainable = False

    return encoder_

# class Hiper_:
#     def __init__(self):
#         self.IMG_HEIGHT=224
#         self.IMG_WIDTH=224
#         self.CHANNELS=3
#
# h = Hiper_()
# enc = tf.keras.applications.ResNet50(False,None,[h.IMG_WIDTH,h.IMG_HEIGHT,h.CHANNELS])
# enc = load_pretrained_weights(enc,h)
