# credits https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/segnet.py
import tensorflow.keras.utils
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D, Input, Activation, concatenate
from tensorflow.keras import Model


IMAGE_ORDERING= "channels_last"

def get_vgg_encoder(input_height=224,  input_width=224, channels=3):

    # assert input_height % 32 == 0
    # assert input_width % 32 == 0

    # if IMAGE_ORDERING == 'channels_first':
    #     img_input = Input(shape=(channels, input_height, input_width))
    # if IMAGE_ORDERING == 'channels_last':
    img_input = Input(shape=(input_height, input_width, channels), name='RGB_input')

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

def get_vgg_encoder_depth(input_height=224,  input_width=224, channels=3):

    # assert input_height % 32 == 0
    # assert input_width % 32 == 0

    # if IMAGE_ORDERING == 'channels_first':
    #     img_input = Input(shape=(channels, input_height, input_width))
    # if IMAGE_ORDERING == 'channels_last':
    img_input = Input(shape=(input_height, input_width, channels), name='DEPTH_input')

    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='dblock1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='dblock1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='dblock1_pool',
                     data_format=IMAGE_ORDERING)(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='dblock2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='dblock2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='dblock2_pool',
                     data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='dblock3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='dblock3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='dblock3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='dblock3_pool',
                     data_format=IMAGE_ORDERING)(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='dblock4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='dblock4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='dblock4_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='dblock4_pool',
                     data_format=IMAGE_ORDERING)(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='dblock5_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='dblock5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='dblock5_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='dblock5_pool',
                     data_format=IMAGE_ORDERING)(x)
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]

def get_vgg_encoder_f(input_height=224,  input_width=224, channels=3):

    # assert input_height % 32 == 0
    # assert input_width % 32 == 0

    # if IMAGE_ORDERING == 'channels_first':
    #     img_input = Input(shape=(channels, input_height, input_width))
    # if IMAGE_ORDERING == 'channels_last':
    img_input = Input(shape=(input_height, input_width, channels))

    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='fblock1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='fblock1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='fblock1_pool',
                     data_format=IMAGE_ORDERING)(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='fblock2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='fblock2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='fblock2_pool',
                     data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='fblock3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='fblock3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='fblock3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='fblock3_pool',
                     data_format=IMAGE_ORDERING)(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='fblock4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='fblock4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='fblock4_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='fblock4_pool',
                     data_format=IMAGE_ORDERING)(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='fblock5_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='fblock5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='fblock5_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='fblock5_pool',
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

    # o_shape = Model(img_input, o).output_shape
    # i_shape = Model(img_input, o).input_shape

    # if IMAGE_ORDERING == 'channels_first':
    #     output_height = o_shape[2]
    #     output_width = o_shape[3]
    #     input_height = i_shape[2]
    #     input_width = i_shape[3]
    #     n_classes = o_shape[1]
    #     o = (Reshape((-1, output_height*output_width)))(o)
    #     o = (Permute((2, 1)))(o)
    # if IMAGE_ORDERING == 'channels_last':
    #     output_height = o_shape[1]
    #     output_width = o_shape[2]
    #     input_height = i_shape[1]
    #     input_width = i_shape[2]
    #     n_classes = o_shape[3]
        # o = (Reshape((output_height*output_width, -1)))(o)

    o = (Activation('sigmoid'))(o)
    model = Model(img_input, o)
    # model.output_width = output_width
    # model.output_height = output_height
    # model.n_classes = n_classes
    # model.input_height = input_height
    # model.input_width = input_width
    # model.model_name = ""

    return model

def _segnet(n_classes, encoder,  input_height=224, input_width=224,
            encoder_level=3, channels=3):

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width, channels=channels)

    feat = levels[encoder_level]
    o = segnet_decoder(feat, n_classes, n_up=4)
    model = get_segmentation_model(img_input, o)

    return model


def vgg_segnet(n_classes, hiper, input_height=224, input_width=224, encoder_level=3, channels=3):

    model = _segnet(n_classes, get_vgg_encoder,  input_height=hiper.IMG_HEIGHT,
                    input_width=hiper.IMG_WIDTH, encoder_level=encoder_level, channels=channels)
    model.model_name = "vgg_segnet"
    model_d = _segnet(n_classes, get_vgg_encoder_depth, input_height=hiper.IMG_HEIGHT,
                    input_width=hiper.IMG_WIDTH, encoder_level=encoder_level, channels=channels)
    model_d.model_name = "vgg_segnet_d"

    model_f = _segnet(n_classes, get_vgg_encoder_f, input_height=hiper.IMG_HEIGHT,
                      input_width=hiper.IMG_WIDTH, encoder_level=encoder_level, channels=2)
    model_f.model_name = "vgg_segnet_f"

    x1 = model.output
    x2 = model_d.output
    cat = concatenate(inputs=[x1, x2], name='concat_final')
    out = model_f(cat)

    return Model(inputs=[model.inputs, model_d.inputs], outputs=out)

def decoder_vgg_segnet(encoder, hiper):
    h = hiper
    return encoder  # encoder is already all model

# t = vgg_segnet(1)
# tensorflow.keras.utils.plot_model(t, show_shapes=True)
# t.summary()
# print('chega')
