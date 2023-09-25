# import tensorflow as tf
from tensorflow.keras import Input, layers, applications, Model
from no_fusion.utils_ import conv_block, upsampling, skipconnect, CFB


filt = 64


connectlayers={
    'resnet34': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet50': ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu'),
    'resnet101': ('conv4_block23_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu'),
    'resnet152': ('conv4_block36_out', 'conv3_block8_out', 'conv2_block3_out', 'conv1_relu'),
}


def build_decoder_resnet1(encoder, hiper):

    z = encoder.output

    x = upsampling('up_conv', z, hiper, filt * 16, 'bottleneck')
    x = conv_block(x, hiper, filt * 16, 3, 'bottleneck_blk1')

    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][0], hiper, filt * 16)

    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk2')

    x = upsampling('up_conv', x, hiper, filt * 8, 'decod2')
    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][1], hiper, filt * 8)

    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk2')

    x = upsampling('up_conv', x, hiper, filt * 4, 'decod3')
    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][2], hiper, filt * 4)

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')  # only 1x1conv
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')   # only 1x1conv

    x = upsampling('up_conv', x, hiper, filt, 'decod4')
    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][3], hiper, filt)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = upsampling('up_conv', x, hiper, filt//2, 'decod5')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)



def build_decoder_resnet2(encoder, hiper):

    z = encoder.output

    x = upsampling('transp_conv', z, hiper, filt * 16, 'bottleneck')

    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][0], hiper, filt * 16)

    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 8, 'decod2')
    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][1], hiper, filt * 8)

    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 4, 'decod3')
    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][2], hiper, filt * 4)

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')

    x = upsampling('transp_conv', x, hiper, filt, 'decod4')
    x = skipconnect('conv1x1', x, encoder, connectlayers[hiper.backbone][3], hiper, filt)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = upsampling('transp_conv', x, hiper, filt//2, 'decod5')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)


def build_decoder_resnet3(encoder, hiper):

    z = encoder.output

    x = upsampling('up_conv', z, hiper, filt * 16, 'bottleneck')
    x = conv_block(x, hiper, filt * 16, 3, 'bottleneck_blk1')

    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][0], hiper)

    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk2')

    x = upsampling('up_conv', x, hiper, filt * 8, 'decod2')
    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][1], hiper)

    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk2')

    x = upsampling('up_conv', x, hiper, filt * 4, 'decod3')
    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][2], hiper)

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')

    x = upsampling('up_conv', x, hiper, filt, 'decod4')
    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][3], hiper)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = upsampling('up_conv', x, hiper, filt//2, 'decod5')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)


def build_decoder_resnet4(encoder, hiper):

    z = encoder.output

    x = upsampling('transp_conv', z, hiper, filt * 16, 'bottleneck')

    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][0], hiper)

    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 8, 'decod2')
    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][1], hiper)

    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk2')

    x = upsampling('transp_conv', x, hiper, filt * 4, 'decod3')
    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][2], hiper)

    x = conv_block(x, hiper, filt * 2, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt * 2, 3, 'decod3_blk2')

    x = upsampling('transp_conv', x, hiper, filt, 'decod4')
    x = skipconnect('concat', x, encoder, connectlayers[hiper.backbone][3], hiper)

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = upsampling('transp_conv', x, hiper, filt//2, 'decod5')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)

def build_decoder_resnet5(encoder, hiper):

    z = encoder.output

    x = upsampling('up_conv', z, hiper, filt * 16, 'bottleneck')
    x = conv_block(x, hiper, filt * 16, 3, 'bottleneck_blk1')

    x = CFB(x, encoder.get_layer(name=connectlayers[hiper.backbone][0]).output, 'cfb1', hiper)

    x = upsampling('up_conv', x, hiper, filt * 8, 'decod2')
    x = conv_block(x, hiper, filt * 8, 3, 'decod2_blk1')
    x = CFB(x, encoder.get_layer(name=connectlayers[hiper.backbone][1]).output, 'cfb2', hiper)

    x = upsampling('up_conv', x, hiper, filt * 4, 'decod3')
    x = conv_block(x, hiper, filt * 4, 3, 'decod3_blk1')
    x = CFB(x, encoder.get_layer(name=connectlayers[hiper.backbone][2]).output, 'cfb3', hiper)

    x = upsampling('up_conv', x, hiper, filt, 'decod4')
    x = conv_block(x, hiper, filt, 3, 'decod4_blk1')
    x = CFB(x, encoder.get_layer(name=connectlayers[hiper.backbone][3]).output, 'cfb4', hiper)

    x = upsampling('up_conv', x, hiper, filt//2, 'decod5')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')

    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)



# x = transp_conv(z, hiper, 512)
# x = up_conv(z,hiper, 512)
#
# x = concat(x, encoder, 33, hiper)
# x = add(x, encoder, 33)
# x = conv1x1(x, encoder, 33, hiper, 512)
#
#
# final = Model(inputs=encoder.input, outputs=x)
# final.summary()
#
# def create_decoder(encoder,backbone,skipconnectype,decoder_type,final_activation):
#    x = encoder.output
