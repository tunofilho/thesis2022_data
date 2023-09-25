from tensorflow.keras import Input, layers, applications, Model, utils
from fusion.utils import conv_block, separable_conv_block, upsampling, skipconnect, CFB

filt = 64


connectlayers = {
    'resnet50': ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'd_add'),  # early fusion
    'resnet101': ('conv4_block23_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu'),
    'resnet152': ('conv4_block36_out', 'conv3_block8_out', 'conv2_block3_out', 'conv1_relu'),
    'fuse': ('d_add_3', 'd_add_2', 'd_add_1', 'd_add'),
    # 'resnet50': ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu'),
}


# decoder
def build_decoder_resnet1(encoder, hiper):
    if hiper.typefusion in ('fusenet', 'fusenetwise', 'resnet'):
        keys_layers = 'fuse'
    else:
        keys_layers = hiper.backbone

    z = encoder.output

    # x = conv_block(z, hiper, filt * 16, 3, 'bottleneck_blk1')
    # x = conv_block(x, hiper, filt * 16, 3, 'bottleneck_blk2')
    x = upsampling('up_conv', z, hiper, filt * 16, 'bottleneck')
    x = conv_block(x, hiper, filt * 16, 3, 'bottleneck_blk1')
#    for _ in range(2):
#        x = conv_block(x, hiper, filt * 16, 3)

    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][0], hiper, filt * 16)

    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk2')

    x = upsampling('up_conv', x, hiper, filt * 8, 'decod2')
    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][1], hiper, filt * 8)

    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk2')

    x = upsampling('up_conv', x, hiper, filt * 4, 'decod3')
    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][2], hiper, filt * 4)

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')  # only 1x1conv
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')   # only 1x1conv

    x = upsampling('up_conv', x, hiper, filt, 'decod4')
    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][3], hiper, filt)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = upsampling('up_conv', x, hiper, filt//2, 'decod5')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)

def build_decoder_resnet2(encoder, hiper):
    if hiper.typefusion in ('fusenet', 'fusenetwise', 'resnet'):
        keys_layers = 'fuse'
    else:
        keys_layers = hiper.backbone

    z = encoder.output

    x = upsampling('transp_conv', z, hiper, filt * 16, 'bottleneck')
#    for _ in range(2):
#        x = conv_block(x, hiper, filt * 16, 3)

    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][0], hiper, filt * 16)

    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 8, 'decod2')
    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][1], hiper, filt * 8)

    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 4, 'decod3')
    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][2], hiper, filt * 4)

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')

    x = upsampling('transp_conv', x, hiper, filt, 'decod4')
    x = skipconnect('conv1x1', x, encoder, connectlayers[keys_layers][3], hiper, filt)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = upsampling('transp_conv', x, hiper, filt//2, 'decod5')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)

# decoder
def build_decoder_resnet3(encoder, hiper):
    if hiper.typefusion in ('fusenet', 'fusenetwise', 'resnet'):
        keys_layers = 'fuse'
    else:
        keys_layers = hiper.backbone

    z = encoder.output

    x = upsampling('up_conv', z, hiper, filt * 16, 'bottleneck')
    x = conv_block(x, hiper, filt * 16, 3, 'bottleneck_blk1')
#    for _ in range(2):
#        x = conv_block(x, hiper, filt * 16, 3)

    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][0], hiper)

    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk2')

    x = upsampling('up_conv', x, hiper, filt * 8, 'decod2')
    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][1], hiper)

    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk2')

    x = upsampling('up_conv', x, hiper, filt * 4, 'decod3')
    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][2], hiper)

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')

    x = upsampling('up_conv', x, hiper, filt, 'decod4')
    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][3], hiper)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = upsampling('up_conv', x, hiper, filt//2, 'decod5')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)

def build_decoder_resnet4(encoder, hiper):
    if hiper.typefusion in ('fusenet', 'fusenetwise', 'resnet'):
        keys_layers = 'fuse'
    else:
        keys_layers = hiper.backbone

    z = encoder.output

    x = upsampling('transp_conv', z, hiper, filt * 16, 'bottleneck')
#    for _ in range(2):
#        x = conv_block(x, hiper, filt * 16, 3)

    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][0], hiper)

    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 8, 'decod2')
    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][1], hiper)

    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 4, 'decod3')
    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][2], hiper)

    x = conv_block(x, hiper, filt * 2, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt * 2, 3, 'decod3_blk2')

    x = upsampling('transp_conv', x, hiper, filt, 'decod4')
    x = skipconnect('concat', x, encoder, connectlayers[keys_layers][3], hiper)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = upsampling('transp_conv', x, hiper, filt//2, 'decod5')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)

def build_decoder_resnet5(encoder, hiper):
    if hiper.typefusion in ('fusenet', 'fusenetwise', 'resnet'):
        keys_layers = 'fuse'
    else:
        keys_layers = hiper.backbone

    z = encoder.output

    x = upsampling('up_conv', z, hiper, filt * 16, 'bottleneck')
    x = conv_block(x, hiper, filt * 16, 3, 'bottleneck_blk1')
#    for _ in range(2):
#        x = conv_block(x, hiper, filt * 16, 3)
    x = CFB(x, encoder.get_layer(name=connectlayers[keys_layers][0]).output, 'cfb1', hiper)

    # x = skipconnect('concat', x, encoder, connectlayers[keys_layers][0], hiper)

    # x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk1')
    # x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk2')

    x = upsampling('up_conv', x, hiper, filt * 8, 'decod2')
    x = conv_block(x, hiper, filt * 8, 3, 'decod2_blk1')
    x = CFB(x, encoder.get_layer(name=connectlayers[keys_layers][1]).output, 'cfb2', hiper)
    # x = skipconnect('concat', x, encoder, connectlayers[keys_layers][1], hiper)

    # x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk1')
    # x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk2')

    x = upsampling('up_conv', x, hiper, filt * 4, 'decod3')
    x = conv_block(x, hiper, filt * 4, 3, 'decod3_blk1')
    x = CFB(x, encoder.get_layer(name=connectlayers[keys_layers][2]).output, 'cfb3', hiper)
    # x = skipconnect('concat', x, encoder, connectlayers[keys_layers][2], hiper)

    # x = conv_block(x, hiper, filt * 2, 3, 'decod3_blk1')
    # x = conv_block(x, hiper, filt * 2, 3, 'decod3_blk2')

    x = upsampling('up_conv', x, hiper, filt, 'decod4')
    x = conv_block(x, hiper, filt, 3, 'decod4_blk1')
    x = CFB(x, encoder.get_layer(name=connectlayers[keys_layers][3]).output, 'cfb4', hiper)
    # x = skipconnect('concat', x, encoder, connectlayers[keys_layers][3], hiper)

    # x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    # x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = upsampling('up_conv', x, hiper, filt//2, 'decod5')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    # x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)

# print('chega')
