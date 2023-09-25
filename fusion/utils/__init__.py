from tensorflow.keras import layers, Model
from tensorflow import repeat
import tensorflow as tf
import os, tempfile

def conv_block(x, hiper, filters, kernel_size, name):
    """
    Convolution + Batch Normalization + Activation layers in conv_block
    :param x: keras layer
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :param filters: number of conv features (channel dimension)
    :param kernel_size: 3x3 or 5x5 or 7x7
    :return: keras layer (conv + bn + act)
    """
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name=name+'_conv_')(x)
    if hiper.BN:
        x = layers.BatchNormalization(name=name+'_bn_')(x)
    x = layers.Activation(hiper.activation, name=name+'_activ_')(x)
    return x


def separable_conv_block(x, hiper, filters, name):
    """
    Separable Depthwise convolution block. 3x3 Depthwise conv + Batch Normalization + activation + conv_block (pointwise convolution).
    :param x: keras layer
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :param filters: number of conv features (channel dimension)
    :return: keras layer (separable depthwise conv)
    """
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', depthwise_initializer=hiper.initializer,
                               depthwise_regularizer=hiper.regularizer, name=name+'_depthconv_')(x)
    if hiper.BN:
        x = layers.BatchNormalization(name=name+'_bn0_')(x)
    x = layers.Activation(hiper.activation, name=name+'_activ0_')(x)
    x = conv_block(x, hiper, filters, 1, name)
    return x

# types of upsampling
def upsampling(choose, *kwargs):
    MAPUP = {
        'up_conv': up_conv,
        'transp_conv': transp_conv,
    }
    if choose not in MAPUP:
        raise ValueError('type of upsampling error')

    return MAPUP[choose](*kwargs)

def up_conv(x, hiper, filters, name):
    # up = layers.UpSampling2D(2, interpolation=hiper.interpolation)(x)
    # up = conv_block(x, hiper, filters, 3, name)
    up = layers.UpSampling2D(2, interpolation=hiper.interpolation, name=name+'_upsample_')(x)
    return up

def transp_conv(x, hiper, filters, name):
    x = layers.Conv2DTranspose(kernel_initializer=hiper.initializer, filters=filters, kernel_size=2, strides=2,
                               padding='same', kernel_regularizer=hiper.regularizer, name=name+'_transp_conv_')(x)
    return x

# types of skipconnect
def skipconnect(choose, *kwargs):
    MAPCONECT = {
        'add': add,
        'concat': concat,
        'conv1x1': conv1x1,
    }
    if choose not in MAPCONECT:
        raise ValueError('type of connection error')

    return MAPCONECT[choose](*kwargs)

def add(x, encoder, name):
    x = layers.add([encoder.get_layer(name=name).output, x], name=name+'_add_decod')
    return x

def concat(x, encoder, name, hiper):
    x = layers.concatenate([encoder.get_layer(name=name).output, x], axis=3, name=name+'_concat_decod')
    if hiper.drop:
        x = layers.Dropout(hiper.droprate)(x)
    return x

def conv1x1(x, encoder, name, hiper, filters):
    y = conv_block(encoder.get_layer(name=name).output, hiper, filters, 1, name=name+'_conv1x1_decod')
    x = layers.add([y, x], name=name+'_add_decod')
    return x

def CFB(fh: layers, fl, name, hiper):
    """fl is the skipconnection features (low-level) and fh is decoder feature (high-level) """
    channels = fh.shape.as_list()[-1]
    fh_ = fh  # (b,h,w,c)
    fl_ = fl  # (b,h,w,c)

    g_fh = layers.GlobalAveragePooling2D()(fh_)   # (b,c)
    g_fh = layers.Reshape((1, 1, channels))(g_fh)  # (b,1,1,c)
    g_fl = layers.GlobalAveragePooling2D()(fl_)   # (b,c)
    g_fl = layers.Reshape((1, 1, channels))(g_fl)   # (b,1,1,c)

    fcl_shared = tf.keras.Sequential(
      [
       layers.Dense(units=channels, activation='relu', name=name+'_fcl_1'),
       layers.Dense(units=channels, activation='sigmoid', name=name+'_fcl_2'),
      ]
    )

    fcl_fh = fcl_shared(g_fh)    # (b,1,1,c)
    fcl_fl = fcl_shared(g_fl)    # (b,1,1,c)

    fh_m = layers.multiply([fh_, fcl_fl], name=name+'_fhxfl')  # (b,h,w,c)
    fl_m = layers.multiply([fl_, fcl_fh], name=name+'_flxfh')   # (b,h,w,c)
    cat = layers.concatenate([fh_m, fl_m], name=name+'_concat')  # (b,h,w,2c)

    cat = layers.Conv2D(channels, 3, padding='same', name=name+'_conv1',
                        kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(cat)
    cat = layers.BatchNormalization(name=name+'_bn1')(cat)
    cat = layers.Activation(hiper.activation, name=name+'_actv1')(cat)
    cat = layers.Conv2D(channels, 3, padding='same', name=name+'_conv2',
                        kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(cat)
    cat = layers.BatchNormalization(name=name+'_bn2')(cat)
    cat = layers.Activation(hiper.activation, name=name+'_actv2')(cat)

    return layers.add([cat, fh_], name=name+'_add')  # (b,h,w,c)


def block1(x, filters, hiper, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.
    Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.
    Returns:
    Output tensor for the residual block.
    """

    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv',
                                 kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(x)
        shortcut = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv',
                      kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(x)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv',
                      kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(x)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv',
                      kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(x)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)

    return x


def stack1(x, filters, blocks, hiper, stride1=2, name=None):
    """A set of stacked residual blocks.
    Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    name: string, stack label.
    Returns:
    Output tensor for the stacked blocks.
    """
    x = block1(x, filters, hiper, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, hiper, conv_shortcut=False, name=name + '_block' + str(i))
    return x

def add_l2_regularizer(model_temp: Model, hiper):
    # Add L2 regularization to the whole model.
    # credits: https://github.com/keras-team/keras/issues/2717

    if hiper.regularizer is None:
        return model_temp

    reg_attributes = ['kernel_regularizer']
    # reg_attributes = ['kernel_regularizer', 'bias_regularizer',   bag-of-tricks-for-image-classification!!!
    #                   'beta_regularizer', 'gamma_regularizer']
    # if isinstance(reg_attributes, str):
    #     reg_attributes = [reg_attributes]

    regularize = hiper.regularizer

    for layer in model_temp.layers:
        for attr in reg_attributes:
            if hasattr(layer, attr):
                setattr(layer, attr, regularize)

    # So far, the regularizers only exist in the model config. We need to
    # reload the model so that Keras adds them to each layer's losses.
    model_json = model_temp.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model_temp.save_weights(tmp_weights_path)

    # Reload the model
    model_temp = tf.keras.models.model_from_json(model_json)
    model_temp.load_weights(tmp_weights_path, by_name=True)

    return model_temp


# credits: https://github.com/nabsabraham/focal-tversky-unet/blob/master/newmodels.py
def expend_as(tensor, rep, name):
    my_repeat = layers.Lambda(lambda x, rep_num: repeat(x, rep_num, axis=3), arguments={'rep_num': rep},  name='psi_up'+name)(tensor)
    return my_repeat


def AttnGatingBlock(x, g, inter_shape, name):
    """ take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same feature channels (theta_x)
    then, upsampling g to be same size as x
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients"""

    shape_x = x.shape.as_list()  # 32
    shape_g = g.shape.as_list()  # 16

    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', name='xl' + name)(x)  # 16
    shape_theta_x = theta_x.shape.as_list()

    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(g)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                        strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                        padding='same', name='g_up' + name)(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same', name='psi' + name)(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = sigmoid_xg.shape.as_list()
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3], name)
    y = layers.multiply([upsample_psi, x], name='q_attn' + name)

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same', name='q_attn_conv' + name)(y)
    result_bn = layers.BatchNormalization(name='q_attn_bn' + name)(result)
    return result_bn
