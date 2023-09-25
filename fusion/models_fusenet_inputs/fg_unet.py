from tensorflow.keras import Model, Input, layers
from fusion.utils import stack1, conv_block, upsampling
from fusion.utils.focus_gate import FocusAttnGatingBlock
from fusion.utils import fusion_strategy

def fus_opt(choose, *args):
    opt = {'add': fusion_strategy.only_add,
           'dffm': fusion_strategy.DFFM,
           'esa_fus': fusion_strategy.esanet_seblock,
           'cbam': fusion_strategy.get_cbam}
    return opt[choose](*args)


skip_layer = {'resnet50': ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu')}
filt = 64

def depth_fg_resnet50(hiper):
    """
    resnet50 depth branch architecture
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :return: depth branch encoder model
    """

    input2 = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, hiper.depth_CHANNELS), name='DEPTH_input')  # depth input

    d = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='dconv1_pad')(input2)
    d = layers.Conv2D(64, 7, strides=2, name='dconv1_conv',
                      kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(d)
    d = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='dconv1_bn',
                                  beta_regularizer=hiper.regularizer, gamma_regularizer=hiper.regularizer)(d)
    d1 = layers.Activation('relu', name='dconv1_relu')(d)

    d = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='dpool1_pad')(d1)
    d = layers.MaxPooling2D(3, strides=2, name='dpool1_pool')(d)

    d2 = stack1(d, 64, 3, hiper, stride1=1, name='dconv2')
    d3 = stack1(d2, 128, 4, hiper, name='dconv3')
    d4 = stack1(d3, 256, 6, hiper, name='dconv4')
    d5 = stack1(d4, 512, 3, hiper, name='dconv5')

    return Model(inputs=input2, outputs=[d1, d2, d3, d4, d5])

def fg_resnet50(depth: Model, hiper):
    """
    resnet50 rgb branch architecture
    :param depth: depth branch keras model
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :return: resnet50 rgb/ddd encoder keras model
    """

    input1 = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, hiper.IMG_CHANNELS), name='RGB_input')  # RGB input

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
    # x = layers.add([depth.output[4], x], name='d_add_4')  # fifth fusion
    x = fus_opt(hiper.fus_str, depth.output[4], x, 'd_add_4')

    return Model(inputs=[input1, depth.inputs], outputs=x)

def decoder_fg_resnet50_build_decoder(encoder: Model, hiper):

    z = encoder.output

    fg1 = FocusAttnGatingBlock(encoder.get_layer(name=skip_layer['resnet50'][0]).output, z,
                               filt*16, fus='par', name='fg1')
    x = upsampling('transp_conv', z, hiper, filt * 16, 'bottleneck')

    x = layers.concatenate([fg1, x], axis=3, name='fg1_concat_decod')
    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk2')

    fg2 = FocusAttnGatingBlock(encoder.get_layer(name=skip_layer['resnet50'][1]).output, z,
                               filt * 8, fus='par', name='fg2')
    x = upsampling('transp_conv', x, hiper, filt * 8, 'decod2')
    x = layers.concatenate([fg2, x], axis=3, name='fg2_concat_decod')
    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk1')
    x_out = conv_block(x, hiper, filt * 4, 3, 'decod2_blk2')

    fg3 = FocusAttnGatingBlock(encoder.get_layer(name=skip_layer['resnet50'][2]).output, z,
                               filt * 4, fus='par', name='fg3')
    x = upsampling('transp_conv', x_out, hiper, filt * 4, 'decod3')
    x = layers.concatenate([fg3, x], axis=3, name='fg3_concat_decod')
    x = conv_block(x, hiper, filt * 2, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt * 2, 3, 'decod3_blk2')

    fg4 = FocusAttnGatingBlock(encoder.get_layer(name=skip_layer['resnet50'][3]).output, z,
                               filt, fus='par', name='fg4')
    x = upsampling('transp_conv', x, hiper, filt, 'decod4')
    x = layers.concatenate([fg4, x], axis=3, name='fg4_concat_decod')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = upsampling('transp_conv', x, hiper, filt//2, 'decod5')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x_out2 = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')

    if hiper.ds:

        out1 = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                               padding='same',
                               activation=hiper.activationfinal,
                               name='output_inter')(x_out)
        out2 = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                             padding='same',
                             activation=hiper.activationfinal,
                             name='output_final')(x_out2)
        return Model(inputs=[encoder.inputs], outputs=[out1, out2], name='fg_unet_ds')

    else:
        out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                            padding='same',
                            activation=hiper.activationfinal,
                            name='output')(x_out2)
        return Model(inputs=[encoder.inputs], outputs=out, name='fg_unet')
