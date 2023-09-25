import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from fusion.utils import conv_block, upsampling, fusion_strategy

def fus_opt(choose, *args):
    opt = {'add': fusion_strategy.only_add,
           'dffm': fusion_strategy.DFFM,
           'esa_fus': fusion_strategy.esanet_seblock,
           'cbam': fusion_strategy.get_cbam}
    return opt[choose](*args)

def block1_afnet(x, filters, hiper, ratio=1, kernel_size=3, stride=1, conv_shortcut=True, name=None):
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
                      kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer, dilation_rate=ratio)(x)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv',
                      kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(x)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)

    return x

def stack1_afnet(x, filters, blocks, hiper, ratio=1, stride1=2, name=None):
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
    x = block1_afnet(x, filters, hiper, ratio, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1_afnet(x, filters, hiper, ratio, conv_shortcut=False, name=name + '_block' + str(i))
    return x

# encoder

def depth_af_resnet50(hiper):
    """
    resnet50 depth branch architecture
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :return: depth branch encoder model
    """

    input2 = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3), name='DEPTH_input')  # depth input
    d = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='dconv1_pad')(input2)
    d = layers.Conv2D(64, 7, strides=2, name='dconv1_conv',
                      kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(d)
    d = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='dconv1_bn',
                                  beta_regularizer=hiper.regularizer, gamma_regularizer=hiper.regularizer)(d)
    d1 = layers.Activation('relu', name='dconv1_relu')(d)  # 112/64

    d = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='dpool1_pad')(d1)
    d = layers.MaxPooling2D(3, strides=2, name='dpool1_pool')(d)

    d2 = stack1_afnet(d, 64, 3, hiper, stride1=1, name='dconv2')  # 56/256
    d3 = stack1_afnet(d2, 128, 4, hiper, name='dconv3')  # 28/512
    d4 = stack1_afnet(d3, 256, 6, hiper, ratio=2, stride1=1, name='dconv4')  # 14/1024   # dilatation rate 2  # 28/1024
    d5 = stack1_afnet(d4, 512, 3, hiper, ratio=4, stride1=1, name='dconv5')  # 7/2048    # dilatation rate 4  # 28/2048

    return Model(inputs=input2, outputs=[d5])

def af_resnet50(hiper):
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
    x = layers.Activation('relu', name='conv1_relu')(x)  # 112/64

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack1_afnet(x, 64, 3, hiper, stride1=1, name='conv2')  # 56/256
    x = stack1_afnet(x, 128, 4, hiper, name='conv3')  # 28/512
    x = stack1_afnet(x, 256, 6, hiper, ratio=2, stride1=1, name='conv4')  # 14/1024   # dilatation rate 2  # 28/1024
    x = stack1_afnet(x, 512, 3, hiper, ratio=4, stride1=1, name='conv5')  # 7/2048    # dilatation rate 4  # 28/2048

    # return Model(inputs=[input1, depth.inputs], outputs=x)
    return Model(inputs=[input1], outputs=x)

def att_cos_mod(k: tf.keras.layers.Layer, i, act='softmax', name='cos_att'):
    fcm_shape = k.get_shape().as_list()  # 28/2048
    a = layers.Reshape((fcm_shape[1] * fcm_shape[2], fcm_shape[3]))(k)  # (b, w*h,c)
    b = layers.Reshape((fcm_shape[1] * fcm_shape[2], fcm_shape[3]))(i)  # (b, w*h,c)
    c = layers.Dot(axes=2, normalize=True, name=name+'cosine')([a, b])  # (b, w*h, w*h)
    d = layers.Activation(act)(c)   # (b, w*h, w*h)
    # p = layers.Dot(axes=(1, 2))([b, d])  # (b,c,w*h)
    p = layers.Dot(axes=(1, 2))([a, d])  # (b,c,w*h)
    p = layers.Permute((2, 1))(p)  # (b,w*h,c)
    p = layers.Reshape((fcm_shape[1], fcm_shape[2], fcm_shape[3]))(p)  # (b,w,h,c)
    # dff = layers.Reshape((fcm_shape[1], fcm_shape[2], fcm_shape[3]))(dff)  # (b,w,h,c)
    # sum_ = layers.Add(name=name+'_add')([k, p])
    sum_ = layers.Add(name=name + '_add')([i, p])

    return sum_

#Residual refinement module
def refine(k: tf.keras.layers.Layer, hiper, name='refine'):
    # encode part
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='afnet_out'+'_conv_')(k) #224
    x1 = conv_block(x, hiper, 64, 3, 'res_blk1_') # 224
    x2 = layers.MaxPooling2D()(x1)  #112

    x3 = conv_block(x2, hiper, 64, 3, 'res_blk2_') #112
    x4 = layers.MaxPooling2D()(x3)  # 56

    x5 = conv_block(x4, hiper, 64, 3, 'res_blk3_') # 56
    x6 = layers.MaxPooling2D()(x5)  # 28

    x7 = conv_block(x6, hiper, 64, 3, 'res_blk4_') # 28
    x8 = layers.MaxPooling2D()(x7)  # 14

    x9 = conv_block(x8, hiper, 64, 3, 'res_bottleneck_')

    d1 = upsampling('up_conv', x9, hiper, 0, 'resup1_up_') #28
    conc1 = layers.concatenate([d1, x7])
    d2 = conv_block(conc1, hiper, 64, 3, 'res_dec1_')

    d3 = upsampling('up_conv', d2, hiper, 0, 'resup2_up_')  # 56
    conc2 = layers.concatenate([d3, x5])
    d4 = conv_block(conc2, hiper, 64, 3, 'res_dec2_')

    d5 = upsampling('up_conv', d4, hiper, 0, 'resup3_up_')  # 112
    conc3 = layers.concatenate([d5, x3])
    d6 = conv_block(conc3, hiper, 64, 3, 'res_dec3_')

    d7 = upsampling('up_conv', d6, hiper, 0, 'resup4_up_')  # 224
    conc4 = layers.concatenate([d7, x1])
    d8 = conv_block(conc4, hiper, 64, 3, 'res_dec4_')

    d9 = layers.Conv2D(1, 3, padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='res_out' + '_conv_')(d8)
    out = layers.add([d9, k])
    out = layers.Activation('sigmoid')(out)
    return out

def decoder_afnet_resnet50_build_decoder(encoder: Model, hiper):

    r = encoder.outputs[0]
    d = encoder.outputs[1]
    att_coef1 = att_cos_mod(r, d, 'softmax', 'cos_r_d')
    att_coef2 = att_cos_mod(d, r, 'softmax', 'cos_d_r')
    att = layers.Add(name='att_sum')([att_coef1, att_coef2])   # 28/2048
    up1 = conv_block(att, hiper, 512, 3, 'up1')
    up1 = upsampling('up_conv', up1, hiper, 0, 'up1_up')   # 56/512
    up2 = conv_block(up1, hiper, 128, 3, 'up2')
    up2 = upsampling('up_conv', up2, hiper, 0, 'up2_up')   # 112/128
    x_out = conv_block(up2, hiper, 2, 3, 'up3')
    x_out = upsampling('up_conv', x_out, hiper, 0, 'up3_up')  # 224/2

    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                        padding='same',
                        activation=hiper.activationfinal,
                        name='output')(x_out)
    return Model(inputs=[encoder.inputs], outputs=out, name='af_net')


def decoder_afnet_resnet50_build_decoder_v2(encoder: Model, hiper):
    r = encoder.outputs[0]
    d = encoder.outputs[1]
    # att = layers.Add(name='add_sum')([r, d])  # 28/2048
    att = fus_opt(hiper.fus_str, d, r, 'add_sum')  # 28/2048
    up1 = conv_block(att, hiper, 512, 3, 'up1')
    up1 = upsampling('up_conv', up1, hiper, 0, 'up1_up')  # 56/512
    up2 = conv_block(up1, hiper, 128, 3, 'up2')
    up2 = upsampling('up_conv', up2, hiper, 0, 'up2_up')  # 112/128
    x_out = conv_block(up2, hiper, 2, 3, 'up3')
    x_out = upsampling('up_conv', x_out, hiper, 0, 'up3_up')  # 224/2

    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                        padding='same',
                        activation=hiper.activationfinal,
                        name='output')(x_out)
    return Model(inputs=[encoder.inputs], outputs=out, name='af_net_v2')

def decoder_afnet_resnet50_build_decoder_v3(encoder: Model, hiper):

    r = encoder.outputs[0]
    d = encoder.outputs[1]
    att_coef1 = att_cos_mod(r, d, 'softmax', 'cos_r_d')
    att_coef2 = att_cos_mod(d, r, 'softmax', 'cos_d_r')
    att = layers.Add(name='att_sum')([att_coef1, att_coef2])   # 28/2048
    up1 = conv_block(att, hiper, 512, 3, 'up1')
    up1 = upsampling('up_conv', up1, hiper, 0, 'up1_up')   # 56/512
    up2 = conv_block(up1, hiper, 128, 3, 'up2')
    up2 = upsampling('up_conv', up2, hiper, 0, 'up2_up')   # 112/128
    x_out = conv_block(up2, hiper, 2, 3, 'up3')
    out1 = upsampling('up_conv', x_out, hiper, 0, 'up3_up')  # 224/2
    out2 = refine(out1, hiper)

    if hiper.ds==2:

        out1 = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                             padding='same',
                             activation=hiper.activationfinal,
                             name='output_inter')(out1)
        out2 = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                             padding='same',
                             activation=hiper.activationfinal,
                             name='output_final')(out2)
        return Model(inputs=[encoder.inputs], outputs=[out1, out2], name='af_net_ds2')

    else:
        out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                            padding='same',
                            activation=hiper.activationfinal,
                            name='output')(out2)
        return Model(inputs=[encoder.inputs], outputs=out, name='af_net_v3')


def decoder_afnet_resnet50_build_decoder_v4(encoder: Model, hiper):
    r = encoder.outputs[0]
    d = encoder.outputs[1]
    att_coef1 = att_cos_mod(r, d, 'softmax', 'cos_r_d')
    att_coef2 = att_cos_mod(d, r, 'softmax', 'cos_d_r')
    att = layers.Add(name='att_sum')([att_coef1, att_coef2])  # 28/2048
    up1 = conv_block(att, hiper, 512, 3, 'up1')
    up1 = upsampling('up_conv', up1, hiper, 0, 'up1_up')  # 56/512
    up2 = conv_block(up1, hiper, 128, 3, 'up2')
    up2 = upsampling('up_conv', up2, hiper, 0, 'up2_up')  # 112/128
    x_out = conv_block(up2, hiper, 32, 3, 'up3')
    out1 = upsampling('up_conv', x_out, hiper, 0, 'up3_up')  # 224/2

    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                        padding='same',
                        activation=hiper.activationfinal,
                        name='output')(out1)
    return Model(inputs=[encoder.inputs], outputs=out, name='af_net_v4')

#
# class hiper:
#     def __init__(self, IMG_WIDTH=224, IMG_HEIGHT=224, IMG_CHANNELS=3, depth_CHANNELS=3, regularizer = None, initializer= tf.keras.initializers.GlorotNormal):
#         self.IMG_WIDTH=IMG_WIDTH
#         self.IMG_HEIGHT=IMG_HEIGHT
#         self.IMG_CHANNELS=IMG_CHANNELS
#         self.depth_CHANNELS=depth_CHANNELS
#         self.regularizer=regularizer
#         self.initializer=initializer
#         self.activationfinal='sigmoid'
#         self.activation='relu'
#         self.BN = True
#         self.ds=True
#         self.interpolation='bilinear'
#
# hip = hiper()
# test = depth_af_resnet50(hip)
# test2 = af_resnet50(hip)
# enc = Model(inputs=[test2.inputs, test.inputs], outputs=[test2.outputs, test.outputs])
# fim = decoder_afnet_resnet50_build_decoder(enc, hip)
# tf.keras.utils.plot_model(fim, show_shapes=True)
# i = tf.keras.Input((224, 224, 1))
# ex = refine(i, hip)
# ti = Model(inputs=i, outputs=ex)
# ti.summary()
#
# print('chega')

