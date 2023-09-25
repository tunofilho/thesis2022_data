import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, hiper, filters, kernel_size, name):

    x = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name=name+'_conv_')(x)
    if hiper.BN:
        x = layers.BatchNormalization(name=name+'_bn_')(x)
    x = layers.Activation(hiper.activation, name=name+'_activ_')(x)
    return x


# def concat(x, encoder, name, hiper): ##without mpdm
#     x = layers.concatenate([encoder.get_layer(name=name).output, x], axis=3, name=name+'_concat_decod')
#
#     return x


def up_conv(x, hiper, filters, name):
    up = layers.UpSampling2D(2, interpolation=hiper.interpolation, name=name+'_upsample_')(x)
    return up

connectlayers = {
    'resnet50': ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'd_add')  # early fusion
    # 'fuse': ('d_add_3', 'd_add_2', 'd_add_1', 'd_add'),
    # 'resnet50': ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu'),
}

filt=64

def dilated_conv(f, hiper, filters, kernel_size, BN, ratio, name):
    f = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      padding='same', kernel_initializer=hiper.initializer,
                      dilation_rate=ratio, kernel_regularizer=hiper.regularizer, name=name+'_conv_')(f)
    if BN:
        f = layers.BatchNormalization(name=name+'_bn_')(f)
    return layers.Activation(hiper.activation, name=name+'_activ_')(f)

def mpdm(f: layers.Layer, hiper, filters, name='mpdm'):
    f = dilated_conv(f, hiper, filters, 3, True, 1, name+'_bconv3_1')
    f = dilated_conv(f, hiper, filters, 3, True, 2, name+'_bconv3_2')
    f = dilated_conv(f, hiper, filters, 3, True, 4, name+'_bconv3_4')
    return f

def build_decoder_final(encoder, hiper):

    keys_layers = hiper.backbone

    z = encoder.output

    x = up_conv(z, hiper, filt * 16, 'bottleneck')
    x = conv_block(x, hiper, filt * 16, 3, 'bottleneck_blk1')

    s1 = mpdm(encoder.get_layer(name= connectlayers[keys_layers][0]).output, hiper, filt * 16, 'mpdm_f4g')
    x = layers.concatenate([s1, x], axis=3, name=connectlayers[keys_layers][0] + '_concat_decod')

    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk1')
    x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk2')

    x = up_conv(x, hiper, filt * 8, 'decod2')
    s2 = mpdm(encoder.get_layer(name=connectlayers[keys_layers][1]).output, hiper, filt * 8, 'mpdm_f3g')
    x = layers.concatenate([s2, x], axis=3, name=connectlayers[keys_layers][1] + '_concat_decod')

    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk1')
    x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk2')

    x = up_conv(x, hiper, filt * 4, 'decod3')
    s3 = mpdm(encoder.get_layer(name=connectlayers[keys_layers][2]).output, hiper, filt * 4, 'mpdm_f2g')
    x = layers.concatenate([s3, x], axis=3, name=connectlayers[keys_layers][2] + '_concat_decod')

    x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
    x = conv_block(x, hiper, filt, 3, 'decod3_blk2')

    x = up_conv(x, hiper, filt, 'decod4')
    s4 = mpdm(encoder.get_layer(name=connectlayers[keys_layers][3]).output, hiper, filt, 'mpdm_f1g')
    x = layers.concatenate([s4, x], axis=3, name=connectlayers[keys_layers][3] + '_concat_decod')

    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
    x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')

    x = up_conv(x, hiper, filt//2, 'decod5')

    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
    x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
    out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)

    return Model(inputs=[encoder.inputs], outputs=out)


# def dlink_rates(x, choose_=False):
#     # x_ with 1024 channels... encoder output
#
#     f_init=512
#     # x = layers.Conv2D(f_init, 3, dilation_rate=1, padding='same')(x_)
#     # x = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)
#     # x = layers.Activation('relu')(x)
#
#     # bottleneck
#     # ratio 1,2,4,8,16
#     x1 = layers.Conv2D(f_init, 3, dilation_rate=1, padding='same')(x)
#     x1 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x1)
#     x1 = layers.Activation('relu')(x1)
#     for r in [2, 4, 8, 16]:
#         x1 = layers.Conv2D(f_init, 3, dilation_rate=r, padding='same')(x1)
#         x1 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x1)
#         x1 = layers.Activation('relu')(x1)
#
#     # ratio 1,2,4,8
#     x2 = layers.Conv2D(f_init, 3, dilation_rate=1, padding='same')(x)
#     x2 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x2)
#     x2 = layers.Activation('relu')(x2)
#     for r in [2, 4, 8]:
#         x2 = layers.Conv2D(f_init, 3, dilation_rate=r, padding='same')(x2)
#         x2 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x2)
#         x2 = layers.Activation('relu')(x2)
#
#     # ratio 1,2,4
#     x3 = layers.Conv2D(f_init, 3, dilation_rate=1, padding='same')(x)
#     x3 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x3)
#     x3 = layers.Activation('relu')(x3)
#     for r in [2, 4]:
#         x3 = layers.Conv2D(f_init, 3, dilation_rate=r, padding='same')(x3)
#         x3 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x3)
#         x3 = layers.Activation('relu')(x3)
#
#     # ratio 1,2
#     x4 = layers.Conv2D(f_init, 3, dilation_rate=1, padding='same')(x)
#     x4 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x4)
#     x4 = layers.Activation('relu')(x4)
#     x4 = layers.Conv2D(f_init, 3, dilation_rate=2, padding='same')(x4)
#     x4 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x4)
#     x4 = layers.Activation('relu')(x4)
#
#     # ratio 1
#     x5 = layers.Conv2D(f_init, 3, dilation_rate=1, padding='same')(x)
#     x5 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x5)
#     x5 = layers.Activation('relu')(x5)
#
#     out1 = layers.add([x1, x2, x3, x4, x5])
#
#     x6 = layers.Conv2D(f_init, 1, dilation_rate=1, padding='same')(x)
#     x6 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x6)
#     x6 = layers.Activation('relu')(x6)
#     x7 = layers.GlobalAveragePooling2D()(x)  # (b, c)
#     # recover the initial shape
#     b_, h_, w_, c_ = x.shape.as_list()
#     x7 = layers.Reshape((1, 1, f_init))(x7)
#     # 1x1 conv + upsample
#     x7 = layers.Conv2D(f_init, 1, dilation_rate=1, padding='same')(x7)
#     x7 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x7)
#     x7 = layers.Activation('relu')(x7)  # (b, c)
#     x7 = layers.UpSampling2D(size=(h_, w_), interpolation='bilinear')(x7)  # (b, w, h , c)
#
#     if choose_:
#         out2 = layers.multiply([x6, out1])
#         out3 = layers.add([out2, x7])
#         return layers.add([out3, out1])
#
#     else:
#         return out1

# def build_decoder_final(encoder, hiper):
#
#     keys_layers = hiper.backbone
#
#     z = encoder.output
#
#     x = up_conv(z, hiper, filt * 8, 'bottleneck')
#     x = conv_block(x, hiper, filt * 8, 3, 'bottleneck_blk1')
# #    for _ in range(2):
# #        x = conv_block(x, hiper, filt * 16, 3)
#
#     x = dlink_rates(x, True)  # True --> global average pooling , False --> without average , only dilated
#
#     x = concat(x, encoder, connectlayers[keys_layers][0], hiper)  # 1024 channels
#
#     x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk1')
#     x = conv_block(x, hiper, filt * 8, 3, 'decod1_blk2')
#
#     x = up_conv(x, hiper, filt * 8, 'decod2')
#     x = concat(x, encoder, connectlayers[keys_layers][1], hiper)   # 512 channels
#
#     x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk1')
#     x = conv_block(x, hiper, filt * 4, 3, 'decod2_blk2')
#
#     x = up_conv(x, hiper, filt * 4, 'decod3')
#     x = concat(x, encoder, connectlayers[keys_layers][2], hiper)  # 256 channels
#
#     x = conv_block(x, hiper, filt, 3, 'decod3_blk1')
#     x = conv_block(x, hiper, filt, 3, 'decod3_blk2')
#
#     x = up_conv(x, hiper, filt, 'decod4')
#     x = concat(x, encoder, connectlayers[keys_layers][3], hiper)  # 64 channels
#
#     x = conv_block(x, hiper, filt//2, 3, 'decod4_blk1')
#     x = conv_block(x, hiper, filt//2, 3, 'decod4_blk2')
#
#     x = up_conv(x, hiper, filt//2, 'decod5')
#
#     x = conv_block(x, hiper, filt//4, 3, 'decod5_blk1')
#     x = conv_block(x, hiper, filt//4, 3, 'decod5_blk2')
#     out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer, padding='same', activation=hiper.activationfinal, name='output')(x)
#
#     return Model(inputs=[encoder.inputs], outputs=out)

# class Hiper:
#     # def __init__(self, initializer, backbone, BN, activationfinal, activation, regularizer, interpolation):
#     def __init__(self):
#         self.initializer=tf.keras.initializers.GlorotNormal(seed=4)
#         self.activationfinal=self.activation='relu'
#         self.regularizer=None
#         self.interpolation='bilinear'
#         self.backbone='resnet50'
#         self.BN = True
#
#
# hip = Hiper()
# # test = layers.Input((224, 224, 3))
# test = tf.keras.applications.ResNet50(False, input_shape=[224, 224, 3])
# final = build_decoder_final(test, hip, True)
# # mod = Model(inputs=test.inputs, outputs=final)
# final.summary()
# tf.keras.utils.plot_model(final, show_shapes=True)
# print('fim')