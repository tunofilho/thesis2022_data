from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
# import os, tempfile


# def load_pretrained_weights(encoder_: tf.keras.Model, hiper, weights_path=None):
#     # if imagenet_weights:
#     # aux = []
#     if hiper.backbone == 'resnet34':
#         tmp_weights_path = os.path.join(tempfile.gettempdir(), 'resnet34_imagenet_1000_no_top.h5')
#         tmp_path = tf.keras.utils.get_file(fname=tmp_weights_path,
#                                            origin='https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5',
#                                            md5_hash='8caaa0ad39d927cb8ba5385bf945d582')
#         mod = tf.keras.models.load_model(tmp_path, compile=True)
#         encoder_.load_weights(tmp_path, by_name=True, skip_mismatch=True)
#         encoder_.trainable = False
#
#     else:
#         raise ValueError('backbone error')
#
#     # for layer in mod.layers:
#     #     if len(layer.weights) != 0:  # avoid layers without weights
#     #         aux.append(layer.name)
#     #
#     # for i in encoder_.layers:
#     #     if i.name in aux:
#     #         i.set_weights(mod.get_layer(i.name).get_weights())  # rgb branch
#     #         i.trainable = False
#
#     return encoder_


def dec_block_linknet(y, filt, n):
    """
    D-LinkNet decoder(fraction)
    :param y: keras model layer
    :param filt: number of input layer filters
    :param n: number of output layer filters
    :return: keras model layer
    """
    y1 = layers.Conv2D(filters=filt / 4, kernel_size=1, padding='same', kernel_initializer='glorot_normal')(y)
    y1 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(y1)
    y1 = layers.Activation('relu')(y1)
    y1 = layers.Conv2DTranspose(filters=filt / 4, kernel_size=2, strides=2, kernel_initializer='glorot_normal')(y1)
    y1 = layers.Conv2D(filters=n, kernel_size=1, padding='same', kernel_initializer='glorot_normal')(y1)
    y1 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(y1)
    y1 = layers.Activation('relu')(y1)
    return y1


# def convolutional_block(x, filter_, name=None):
#     # copy tensor to variable called x_skip
#     x_skip = x
#     # Layer 1
#     x = layers.Conv2D(filter_, (3, 3), padding='same', strides=(2, 2), name=name + '_1_conv')(x)
#     x = layers.BatchNormalization(axis=3, name=name + '_1_bn')(x)
#     x = layers.Activation('relu', name=name + '_1_relu')(x)
#     # Layer 2
#     x = layers.Conv2D(filter_, (3, 3), padding='same', name=name + '_2_conv')(x)
#     x = layers.BatchNormalization(axis=3, name=name + '_2_bn')(x)
#     # Processing Residue with conv(1,1)
#     x_skip = layers.Conv2D(filter_, (1, 1), strides=(2, 2), name=name + '_0_conv')(x_skip)
#     x_skip = layers.BatchNormalization(axis=3, name=name + '_0_bn')(x_skip)
#     # Add Residue
#     x = layers.Add(name=name + '_add')([x, x_skip])
#     x = layers.Activation('relu', name=name + '_out')(x)
#     return x
#
#
# def identity_block(x, filter_, name=None):
#     # copy tensor to variable called x_skip
#     x_skip = x
#     # Layer 1
#     x = layers.Conv2D(filter_, (3, 3), padding='same', name=name + '_1_conv')(x)
#     x = layers.BatchNormalization(axis=3, name=name + '_1_bn')(x)
#     x = layers.Activation('relu', name=name + '_1_relu')(x)
#     # Layer 2
#     x = layers.Conv2D(filter_, (3, 3), padding='same', name=name + '_2_conv')(x)
#     x = layers.BatchNormalization(axis=3, name=name + '_2_bn')(x)
#     # Add Residue
#     x = layers.Add(name=name + '_add')([x, x_skip])
#     x = layers.Activation('relu', name=name + '_out')(x)
#     return x


# def ResNet34(y):
#     # Step 1 (Setup Input Layer)
#     x_input = y
#     x = layers.ZeroPadding2D((3, 3), name='conv1_pad')(x_input)
#     # Step 2 (Initial Conv layer along with maxPool)
#     x = layers.Conv2D(64, kernel_size=7, strides=2, name='conv1_conv')(x)
#     x = layers.BatchNormalization(name='conv1_bn')(x)
#     x = layers.Activation('relu', name='conv1_relu')(x)
#     x = layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='pool1_pool')(x)
#     # Define size of sub-blocks and initial filter size
#     block_layers = [3, 4, 6, 3]
#     filter_size = 64
#     # Step 3 Add the Resnet Blocks
#     for i in range(4):
#         if i == 0:
#             # For sub-block 1 Residual/Convolutional block not needed
#             for j in range(block_layers[i]):
#                 x = identity_block(x, filter_size, name=f'conv{i + 2}_block{j + 1}')
#         else:
#             # One Residual/Convolutional Block followed by Identity blocks
#             # The filter size will go on increasing by a factor of 2
#             filter_size = filter_size * 2
#             x = convolutional_block(x, filter_size, name=f'conv{i + 2}_block1')
#             for j in range(block_layers[i] - 1):
#                 x = identity_block(x, filter_size, name=f'conv{i + 2}_block{j + 2}')
#     return keras.Model(inputs=[x_input], outputs=[x])


# encoder
def dlink(hiper):
    img_input = keras.Input(shape=(hiper.IMG_HEIGHT, hiper.IMG_WIDTH, 3))
    # img_input = keras.Input(shape=(None, None, hiper.IMG_CHANNELS))
    enc = tf.keras.applications.ResNet50(False, weights='imagenet', input_tensor=img_input)
    enc.trainable = False
    # enc = ResNet34(img_input)
    # enc = load_pretrained_weights(enc, hiper)  # load imagenet weights
    x = enc.get_layer(name='conv4_block6_out').output   # 14/1024
    x = layers.Conv2D(512, 3, dilation_rate=(1, 1), padding='same')(x)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)
    x = layers.Activation('relu')(x)

    # bottleneck
    x1 = layers.Conv2D(512, 3, dilation_rate=(1, 1), padding='same')(x)
    x1 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(512, 3, dilation_rate=(2, 2), padding='same')(x1)
    x1 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(512, 3, dilation_rate=(4, 4), padding='same')(x1)
    x1 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(512, 3, dilation_rate=(8, 8), padding='same')(x1)
    x1 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x1)
    x1 = layers.Activation('relu')(x1)

    x2 = layers.Conv2D(512, 3, dilation_rate=(1, 1), padding='same')(x)
    x2 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.Conv2D(512, 3, dilation_rate=(2, 2), padding='same')(x2)
    x2 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.Conv2D(512, 3, dilation_rate=(4, 4), padding='same')(x2)
    x2 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x2)
    x2 = layers.Activation('relu')(x2)

    x3 = layers.Conv2D(512, 3, dilation_rate=(1, 1), padding='same')(x)
    x3 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = layers.Conv2D(512, 3, dilation_rate=(2, 2), padding='same')(x3)
    x3 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x3)
    x3 = layers.Activation('relu')(x3)

    x4 = layers.Conv2D(512, 3, dilation_rate=(1, 1), padding='same')(x)
    x4 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x4)
    x4 = layers.Activation('relu')(x4)

    x = layers.add([x, x1, x2, x3, x4])

    model_enc = keras.Model(inputs=img_input, outputs=x)

    # decoder
    x = model_enc.output
    x = dec_block_linknet(x, 512, 512)
    # x = layers.add([x, model_enc.get_layer('conv4_block6_out').output])
    x = layers.add([x, model_enc.get_layer('conv3_block4_out').output])   # 28/512

    x = dec_block_linknet(x, 512, 256)
    # x = layers.add([x, model_enc.get_layer('conv3_block4_out').output])
    x = layers.add([x, model_enc.get_layer('conv2_block3_out').output])   # 56/256

    x = dec_block_linknet(x, 256, 64)
    x = layers.add([x, model_enc.get_layer('conv1_relu').output])  # 112/64

    x = dec_block_linknet(x, 64, 64)
    # x = layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, kernel_initializer='glorot_normal')(x)
    x = layers.Conv2D(1, 1, padding='same', kernel_initializer='glorot_normal', activation='sigmoid')(x)

    return keras.Model(inputs=img_input, outputs=[x])
