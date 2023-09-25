from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, Activation, UpSampling2D
import tensorflow as tf

IMAGE_ORDERING= "channels_last"

def load_pretrained_weights(encoder_ : tf.keras.Model, hiper, weights_path=None):

    # if imagenet_weights:
    aux = []
    if hiper.backbone == 'vgg16':
        mod = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
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

    return encoder_

def get_vgg_encoder(input_height=224,  input_width=224, channels=3):

    # assert input_height % 32 == 0
    # assert input_width % 32 == 0

    # if IMAGE_ORDERING == 'channels_first':
    #     img_input = Input(shape=(channels, input_height, input_width))
    # if IMAGE_ORDERING == 'channels_last':
    img_input = Input(shape=(input_height, input_width, channels))

    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',
                     data_format=IMAGE_ORDERING)(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',
                     data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',
                     data_format=IMAGE_ORDERING)(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool',
                     data_format=IMAGE_ORDERING)(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool',
                     data_format=IMAGE_ORDERING)(x)
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]


def segnet_decoder(f, n_classes, n_up=3):

    assert n_up >= 2

    o = f
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    for _ in range(n_up-2):
        o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = (Conv2D(128, (3, 3), padding='valid',
             data_format=IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    return o

def get_segmentation_model(input, output):

    img_input = input
    o = output

    o = (Activation('sigmoid'))(o)
    model = Model(img_input, o)


    return model


def _segnet(n_classes, encoder,  input_height=224, input_width=224,
            encoder_level=3, channels=3):

    img_input, levels = encoder(input_height=input_height,  input_width=input_width, channels=channels)

    feat = levels[encoder_level]
    o = segnet_decoder(feat, n_classes, n_up=4)
    model = get_segmentation_model(img_input, o)

    return model


def vgg_segnet(n_classes, hiper, input_height=224, input_width=224, encoder_level=3, channels=3):

    model = _segnet(n_classes, get_vgg_encoder,  input_height=hiper.IMG_HEIGHT,
                    input_width=hiper.IMG_WIDTH, encoder_level=encoder_level, channels=3)
    model.model_name = "vgg_segnet"

    model = load_pretrained_weights(model, hiper)

    return model

# class Hip():
#     def __init__(self):
#         self.backbone= 'vgg16'
#         self.IMG_HEIGHT=224
#         self.IMG_WIDTH=224
#         self.IMG_CHANNELS=3
#
# h = Hip()
# m = vgg_segnet(1, h)
# m.summary()
# print('chega')

