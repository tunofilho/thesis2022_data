from tensorflow.keras import Input, Model, layers, applications
from no_fusion.utils_ import conv_block, upsampling
from .focus_gated import FocusAttnGatingBlock

skip_layer = {'resnet50': ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu')}
filt = 64


def load_pretrained_weights(encoder_: Model, hiper, weights_path=None):
    # if imagenet_weights:
    aux = []
    if hiper.backbone == 'resnet50':
        mod = applications.ResNet50(include_top=False, weights='imagenet',
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


def fg_resnet50(hiper):
    """
    resnet50 rgb branch architecture
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :return: resnet50 rgb encoder keras model
    """

    input1 = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3), name='RGB_input')  # RGB input

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(input1)
    x = layers.Conv2D(64, 7, strides=2, name='conv1_conv',
                      kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(x)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn',
                                  beta_regularizer=hiper.regularizer, gamma_regularizer=hiper.regularizer)(x)
    x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack1(x, 64, 3, hiper, stride1=1, name='conv2')
    x = stack1(x, 128, 4, hiper, name='conv3')
    x = stack1(x, 256, 6, hiper, name='conv4')
    x = stack1(x, 512, 3, hiper, name='conv5')

    return Model(inputs=[input1], outputs=x)


def decoder_fg_resnet50_build_decoder(encoder: Model, hiper):

    enc = load_pretrained_weights(encoder, hiper)
    z = enc.output

    fg1 = FocusAttnGatingBlock(encoder.get_layer(name=skip_layer['resnet50'][0]).output,
                               z, filt * 16, fus='par', name='fg1')  #1024
    x = upsampling('transp_conv', z, hiper, filt * 16, 'bottleneck')

    x = layers.concatenate([fg1, x], axis=3, name='fg1_concat_decod')
    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk2')

    fg2 = FocusAttnGatingBlock(encoder.get_layer(name=skip_layer['resnet50'][1]).output,
                               z, filt * 8, fus='par', name='fg2')
    x = upsampling('transp_conv', x, hiper, filt * 8, 'decod2')
    x = layers.concatenate([fg2, x], axis=3, name='fg2_concat_decod')
    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk1')
    x_out1 = conv_block(x, hiper, filt * 4, 3, 'decod2_blk2')

    fg3 = FocusAttnGatingBlock(encoder.get_layer(name=skip_layer['resnet50'][2]).output,
                               z, filt * 4, fus='par', name='fg3')
    x = upsampling('transp_conv', x_out1, hiper, filt * 4, 'decod3')
    x = layers.concatenate([fg3, x], axis=3, name='fg3_concat_decod')
    x = conv_block(x, hiper, filt * 2, 3, 'decod3_blk1')
    x_out2 = conv_block(x, hiper, filt * 2, 3, 'decod3_blk2')

    fg4 = FocusAttnGatingBlock(encoder.get_layer(name=skip_layer['resnet50'][3]).output,
                               z, filt, fus='par', name='fg4')
    x = upsampling('transp_conv', x_out2, hiper, filt, 'decod4')
    x = layers.concatenate([fg4, x], axis=3, name='fg4_concat_decod')
    x = conv_block(x, hiper, filt // 2, 3, 'decod4_blk1')
    x_out3 = conv_block(x, hiper, filt // 2, 3, 'decod4_blk2')

    x = upsampling('transp_conv', x_out3, hiper, filt // 2, 'decod5')

    x = conv_block(x, hiper, filt // 4, 3, 'decod5_blk1')
    x_out4 = conv_block(x, hiper, filt // 4, 3, 'decod5_blk2')

    if hiper.ds!=0:

        out1 = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                             padding='same',
                             activation=hiper.activationfinal,
                             name='output_inter1')(x_out1)
        out2 = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                             padding='same',
                             activation=hiper.activationfinal,
                             name='output_inter2')(x_out2)

        out3 = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                             padding='same',
                             activation=hiper.activationfinal,
                             name='output_inter3')(x_out3)

        out4 = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                             padding='same',
                             activation=hiper.activationfinal,
                             name='output_final')(x_out4)
        return Model(inputs=[encoder.inputs], outputs=[out1, out2, out3, out4], name='fg_unet_ds')

    else:
        out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                            padding='same',
                            activation=hiper.activationfinal,
                            name='output')(x_out4)
        return Model(inputs=[encoder.inputs], outputs=out, name='fg_unet')
