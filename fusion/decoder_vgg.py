# import tensorflow as tf
from tensorflow.keras import Input, layers, Model, applications, utils
from fusion.utils import conv_block, separable_conv_block, upsampling, skipconnect

# def conv_blockv2(x, hiper, filters, kernel_size, name=None):
#     """
#     Convolution + Batch Normalization + Activation layers in conv_block
#     :param x: keras layer
#     :param hiper: choice of initializer, regularizer, activation and batch normalization option
#     :param filters: number of conv features (channel dimension)
#     :param kernel_size: 3x3 or 5x5 or 7x7
#     :return: keras layer (conv + bn + act)
#     :name: convolutional block name
#     """
#     x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer=hiper.initializer,
#                       kernel_regularizer=hiper.regularizer, name=name+'_conv_')(x)
#     if hiper.BN:
#         x = layers.BatchNormalization(name=name+'_bn_')(x)
#     x = layers.Activation(hiper.activation, name=name+'_activ_')(x)
#     return x


# build vgg encoder without pretrained weights
def build_encoder(hiper, input_name: str):
    y = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, hiper.IMG_CHANNELS), name=input_name)
    x = conv_block(y, hiper, filt, 3, 'blk1')
    x = conv_block(x, hiper, filt, 3, 'blk2')
    x = layers.MaxPooling2D((2, 2), name='blk2_pool')(x)

    for _ in range(nblock[hiper.backbone][0]):
        x = conv_block(x, hiper, filt * 2, 3, 'blk3'+str(_))
    x = layers.MaxPooling2D((2, 2), name='blk3_pool')(x)

    for _ in range(nblock[hiper.backbone][1]):
        x = conv_block(x, hiper, filt * 4, 3, 'blk4'+str(_))
    x = layers.MaxPooling2D((2, 2), name='blk4_pool')(x)

    for _ in range(nblock[hiper.backbone][2]):
        x = conv_block(x, hiper, filt * 8, 3, 'blk5'+str(_))
    x = layers.MaxPooling2D((2, 2), name='blk5_pool')(x)

    for _ in range(nblock[hiper.backbone][3]):
        x = conv_block(x, hiper, filt * 8, 3, 'blk6'+str(_))

    return Model(inputs=y, outputs=x)


filt = 64

# number of convolutional blocks
nblock = {
    'scratch16': (2, 3, 3, 3),
    'scratch19': (2, 4, 4, 4),
}

# encoder layers index to skipconnections

connectlayers = {
    'scratch16': ('blk52_out', 'blk42_out', 'blk31_out', 'blk2_out'),
    'scratch19': ('blk53_out', 'blk43_out', 'blk31_out', 'blk2_out'),
    'vgg16': ('block4_conv3', 'block3_conv3', 'block2_conv2', 'd_add'),  # early fusion
    'vgg19': ('block4_conv4', 'block3_conv4', 'block2_conv2', 'd_add'),  # early fusion
    'fuse': ('d_add_3', 'd_add_2', 'd_add_1', 'd_add'),
    # 'vgg16': ('block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
    # 'vgg19': ('block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'),
}


# build decoder up_convolution
def build_decoder_vgg1(encoder, hiper):
    if hiper.typefusion in ('fusenet', 'resnet'):
        keys_layers = 'fuse'
    else:
        keys_layers = hiper.backbone

    z = encoder.output

    if hiper.backbone=='vgg16' :   # bottleneck is different - exception
        x = upsampling('up_conv', z, hiper, filt * 8, 'bottleneck')
        for _ in range(2):
            x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk'+str(_))
    else:
        x = conv_block(z, hiper, filt * 8, 3, 'bottleneck_blk1')
        x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk2')

    x = upsampling('up_conv', x, hiper, filt * 8, 'decod1')
    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][0], hiper, filt * 8)

    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk2')

    x = upsampling('up_conv', x, hiper, filt * 4, 'decod2')
    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][1], hiper, filt * 4)

    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk2')

    x = upsampling('up_conv', x, hiper, filt * 2, 'decod3')
    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][2], hiper, filt * 2)

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')

    x = upsampling('up_conv', x, hiper, filt, 'decod4')
    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][3], hiper, filt)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)

# build decoder transp_convolution
def build_decoder_vgg2(encoder, hiper):
    if hiper.typefusion in ('fusenet', 'resnet'):
        keys_layers = 'fuse'
    else:
        keys_layers = hiper.backbone

    z = encoder.output

    if hiper.backbone=='vgg16' :  # bottleneck is different - exception
        x = upsampling('transp_conv', z, hiper, filt * 8, 'bottleneck')
        for _ in range(2):
            x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk'+str(_))
    else:
        x = conv_block(z, hiper, filt * 8, 3, 'bottleneck_blk1')
        x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 8, 'decod1')
    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][0], hiper, filt * 8)

    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 4, 'decod2')
    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][1], hiper, filt * 4)

    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 2, 'decod3')
    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][2], hiper, filt * 2)

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')

    x = upsampling('transp_conv', x, hiper, filt, 'decod4')
    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][3], hiper, filt)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)

# build decoder up_convolution (concat)
def build_decoder_vgg3(encoder: Model, hiper):
    if hiper.typefusion in ('fusenet', 'resnet'):
        keys_layers = 'fuse'
    else:
        keys_layers = hiper.backbone

    z = encoder.output

    if hiper.backbone=='vgg16':   # bottleneck is different - exception
        x = upsampling('up_conv', z, hiper, filt * 8, 'bottleneck')
        for _ in range(2):
            x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk'+str(_))
    else:
        x = conv_block(z, hiper, filt * 8, 3, 'bottleneck_blk1')
        x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk2')

    x = upsampling('up_conv', x, hiper, filt * 8, 'decod1')
    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][0], hiper)

    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk2')

    x = upsampling('up_conv', x, hiper, filt * 4, 'decod2')
    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][1], hiper)

    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk2')

    x = upsampling('up_conv', x, hiper, filt * 2, 'decod3')
    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][2], hiper)

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')

    x = upsampling('up_conv', x, hiper, filt, 'decod4')
    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][3], hiper)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)


# build decoder transp_convolution (concat)
def build_decoder_vgg4(encoder, hiper):
    if hiper.typefusion in ('fusenet', 'resnet'):
        keys_layers = 'fuse'
    else:
        keys_layers = hiper.backbone

    z = encoder.output

    if hiper.backbone=='vgg16':  # bottleneck is different - exception
        x = upsampling('transp_conv', z, hiper, filt * 8, 'bottleneck')
        for _ in range(2):
            x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk'+str(_))
    else:
        x = conv_block(z, hiper, filt * 8, 3, 'bottleneck_blk1')
        x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 8, 'decod1')
    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][0], hiper)

    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 4, 'decod2')
    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][1], hiper)

    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 2, 'decod3')
    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][2], hiper)

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')

    x = upsampling('transp_conv', x, hiper, filt, 'decod4')
    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][3], hiper)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)


# t = build_encoder(hiper)
# t = applications.VGG19(False, input_shape=[224,224,3])
# f = build_decoder_vgg2(t, hiper)
# f.summary()
# utils.plot_model(f, show_shapes=True)
# print('chega')