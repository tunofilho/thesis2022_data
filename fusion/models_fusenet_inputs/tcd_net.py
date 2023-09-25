import tensorflow as tf
from tensorflow.keras import layers, Model
from fusion.utils import upsampling

skip_layer = {'resnet50': ('d_add_3', 'd_add_2', 'd_add_1', 'd_add')}

def mult_add(f, up, name):
    m = layers.Multiply(name=name+'_mult')([f, up])
    m = layers.Add(name=name+'_add')([m, up])
    return m

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

def dpm(dff: tf.keras.layers.Layer, up, hiper, filters, name='dpm'):
    dff = dilated_conv(dff, hiper, filters, 3, True, 1, name+'_conv1')  # 56/256
    fcm_shape = dff.get_shape().as_list()
    dff = layers.Reshape((fcm_shape[1] * fcm_shape[2], fcm_shape[3]))(dff)  # (b, w*h,c)
    up = layers.Reshape((fcm_shape[1] * fcm_shape[2], fcm_shape[3]))(up)  # (b, w*h,c)
    smw = layers.Dot(axes=2, normalize=True, name=name+'cosine')([dff, up])  # (b, w*h, w*h)
    ef = layers.Dot(axes=(1, 2))([dff, smw])  # (b,c,w*h)
    sf = layers.Dot(axes=(1, 2))([dff, 1-smw])  # (b,c,w*h)
    ef = layers.Permute((2, 1))(ef)  # (b,w*h,c)
    sf = layers.Permute((2, 1))(sf)   # (b,w*h,c)
    ef = layers.Reshape((fcm_shape[1], fcm_shape[2], fcm_shape[3]))(ef)  # (b,w,h,c)
    sf = layers.Reshape((fcm_shape[1], fcm_shape[2], fcm_shape[3]))(sf)  # (b,w,h,c)
    cat = layers.concatenate([ef, sf], name=name+'cat')
    cat = dilated_conv(cat, hiper, filters, 3, True, 1, name+'_conv2')
    return cat

def dpm2(dff: tf.keras.layers.Layer, up, hiper, filters, name='dpm'):
    dff = dilated_conv(dff, hiper, filters, 3, True, 1, name + '_conv1')  # 56/256
    dff_flatt = layers.Reshape((-1,))(dff)  # (b, w*h*c)
    up_flatt = layers.Reshape((-1,))(up)  # (b, w*h*c)
    smw = layers.Dot(axes=1, normalize=True, name=name + 'cosine')([dff_flatt, up_flatt])  # (b, 1)
    ef = layers.Multiply()([dff, smw])
    sf = layers.Multiply()([dff, 1-smw])
    cat = layers.concatenate([ef, sf], name=name + 'cat')
    cat = dilated_conv(cat, hiper, filters, 3, True, 1, name + '_conv2')
    return cat


def decoder_tcd_resnet50_build_decoder(encoder: Model, hiper):

    z = encoder.output[4]
    x = mpdm(z, hiper, 2048, 'mpdm_f5g')
    x = dilated_conv(x, hiper, 1024, 3, True, 1, 'conv1_U')
    up = upsampling('up_conv', x, hiper, 1024, 'conv1_U')   # 14/1024

    x1 = mpdm(encoder.get_layer(name=skip_layer[hiper.backbone][0]).output, hiper, 1024, 'mpdm_f4g')
    x1 = mult_add(x1, up, name='merge1')
    x1 = dilated_conv(x1, hiper, 512, 3, True, 1, 'conv2_U')
    up1 = upsampling('up_conv', x1, hiper, 512, 'conv2_U')  # 28/512

    hp = mpdm(encoder.get_layer(name=skip_layer[hiper.backbone][1]).output, hiper, 512, 'mpdm_f3g')
    hp = mult_add(hp, up1, name='merge2')  # 28/512

    x2 = dilated_conv(hp, hiper, 256, 3, True, 1, 'conv3_U')
    up2 = upsampling('up_conv', x2, hiper, 256, 'conv3_U')  # 56/256

    dp2 = dpm2(encoder.get_layer(name=skip_layer[hiper.backbone][2]).output, up2, hiper, 256, 'dpm_2')  # 56/256 dpm or dpm2??
    dp2 = mult_add(dp2, up2, name='merge3')  # 56/256
    dp2 = dilated_conv(dp2, hiper, 64, 3, True, 1, 'conv4_U')  # 56/64
    dp2 = upsampling('up_conv', dp2, hiper, 64, 'conv4_U')  # 112/64

    x3 = dilated_conv(up2, hiper, 64, 3, True, 1, 'conv5_U')  # 56/64
    up3 = upsampling('up_conv', x3, hiper, 64, 'conv5_U')  # 112/64
    dp1 = dpm2(encoder.get_layer(name=skip_layer[hiper.backbone][3]).output, up3, hiper, 64, 'dpm_1')  # 112/64 dpm or dpm2??
    dp1 = mult_add(dp1, up3, name='merge4')  # 112/64

    x_out = layers.concatenate([dp1, dp2], name='low_level_concat')
    x_out = dilated_conv(x_out, hiper, 64, 3, True, 1, 'conv6_U')  # 112/64
    x_out = upsampling('up_conv', x_out, hiper, 64, 'conv6_U')  # 224/64

    if hiper.ds==1:

        out1 = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                             padding='same',
                             activation=hiper.activationfinal,
                             name='output_inter')(hp)
        out2 = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                             padding='same',
                             activation=hiper.activationfinal,
                             name='output_final')(x_out)
        return Model(inputs=[encoder.inputs], outputs=[out1, out2], name='tcd_net_ds')

    else:
        out = layers.Conv2D(1, 1, kernel_initializer=hiper.initializer,
                            padding='same',
                            activation=hiper.activationfinal,
                            name='output')(x_out)
        return Model(inputs=[encoder.inputs], outputs=out, name='tcd_net')