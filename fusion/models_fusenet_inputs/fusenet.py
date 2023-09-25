import tensorflow.keras.utils
from tensorflow.keras import Model, layers
from fusion.utils import conv_block, upsampling


filt = 64

def decoder_vgg_fusenet(encoder, hiper):

    z = encoder.output

    if hiper.backbone=='vgg16':   # bottleneck is different - exception
        x = upsampling('up_conv', z, hiper, filt * 8, 'bottleneck')
        for _ in range(2):
            x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk'+str(_))
    else:
        x = conv_block(z, hiper, filt * 8, 3, 'bottleneck_blk1')
        x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk2')

    x = upsampling('up_conv', x, hiper, filt * 8, 'decod1')

    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod1_blk2')

    x = upsampling('up_conv', x, hiper, filt * 4, 'decod2')

    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 2, 3, 'decod2_blk2')

    x = upsampling('up_conv', x, hiper, filt * 2, 'decod3')

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')

    x = upsampling('up_conv', x, hiper, filt, 'decod4')

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)

# t = vgg_segnet(1)
# tensorflow.keras.utils.plot_model(t, show_shapes=True)
# t.summary()
# print('chega')
