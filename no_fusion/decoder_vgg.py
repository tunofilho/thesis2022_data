from tensorflow.keras import layers, Model, applications, utils
from no_fusion.utils_ import conv_block, upsampling, skipconnect


filt = 64

# number of convolutional blocks
# nblock = {
#     'scratch16': (2, 3, 3, 3),
#     'scratch19': (2, 4, 4, 4),
# }

# encoder layers index to skipconnections

connectlayers = {
    'scratch16': ('blk52_activ_', 'blk42_activ_', 'blk31_activ_', 'blk2_activ_'),
    'scratch19': ('blk53_activ_', 'blk43_activ_', 'blk31_activ_', 'blk2_activ_'),
    'vgg16': ('block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
    'vgg19': ('block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'),
}



def build_decoder_vgg1(encoder, hiper):

    z = encoder.output

    if hiper.backbone=='vgg16' or hiper.backbone=='vgg19':   # bottleneck is different - exception
        x = upsampling('up_conv', z, hiper, filt * 8, 'bottleneck')
        for _ in range(2):
            x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk'+str(_))
    else:
        x = conv_block(z, hiper, filt * 8, 3, 'bottleneck_blk1')
        x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk2')

    x = upsampling('up_conv', x, hiper, filt * 8, 'decod1')
    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][0], hiper, filt * 8)

    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk2')

    x = upsampling('up_conv', x, hiper, filt * 4, 'decod2')
    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][1], hiper, filt * 4)

    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk2')

    x = upsampling('up_conv', x, hiper, filt * 2, 'decod3')
    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][2], hiper, filt * 2)

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')

    x = upsampling('up_conv', x, hiper, filt, 'decod4')
    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][3], hiper, filt)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)


def build_decoder_vgg2(encoder, hiper):

    z = encoder.output

    if hiper.backbone=='vgg16' or hiper.backbone=='vgg19':  # bottleneck is different - exception
        x = upsampling('transp_conv', z, hiper, filt * 8, 'bottleneck')
        for _ in range(2):
            x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk'+str(_))
    else:
        x = conv_block(z, hiper, filt * 8, 3, 'bottleneck_blk1')
        x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 8, 'decod1')
    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][0], hiper, filt * 8)

    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 4, 'decod2')
    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][1], hiper, filt * 4)

    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 2, 'decod3')
    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][2], hiper, filt * 2)

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')

    x = upsampling('transp_conv', x, hiper, filt, 'decod4')
    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][3], hiper, filt)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)


def build_decoder_vgg3(encoder: Model, hiper):

    z = encoder.output

    if hiper.backbone=='vgg16' or hiper.backbone=='vgg19':   # bottleneck is different - exception
        x = upsampling('up_conv', z, hiper, filt * 8, 'bottleneck')
        for _ in range(2):
            x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk'+str(_))
    else:
        x = conv_block(z, hiper, filt * 8, 3, 'bottleneck_blk1')
        x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk2')

    x = upsampling('up_conv', x, hiper, filt * 8, 'decod1')
    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][0], hiper)

    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk2')

    x = upsampling('up_conv', x, hiper, filt * 4, 'decod2')
    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][1], hiper)

    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk2')

    x = upsampling('up_conv', x, hiper, filt * 2, 'decod3')
    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][2], hiper)

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')

    x = upsampling('up_conv', x, hiper, filt, 'decod4')
    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][3], hiper)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)


def build_decoder_vgg4(encoder, hiper):

    z = encoder.output

    if hiper.backbone=='vgg16' or hiper.backbone=='vgg19':  # bottleneck is different - exception
        x = upsampling('transp_conv', z, hiper, filt * 8, 'bottleneck')
        for _ in range(2):
            x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk'+str(_))
    else:
        x = conv_block(z, hiper, filt * 8, 3, 'bottleneck_blk1')
        x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 8, 'decod1')
    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][0], hiper)

    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 4, 'decod2')
    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][1], hiper)

    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 2, 'decod3')
    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][2], hiper)

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')

    x = upsampling('transp_conv', x, hiper, filt, 'decod4')
    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][3], hiper)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)

# t = applications.VGG19(False, input_shape=[224,224,3])
# f = build_decoder_vgg4(t, hiper)
# f.summary()
# utils.plot_model(f, show_shapes=True)
# print('chega')