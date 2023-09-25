from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, \
    Concatenate, Conv2D, Add, Activation, Lambda, add, Conv2DTranspose
from tensorflow.keras import backend as K


def cbam_block(cbam_feature, ratio=16, fusion='seq', name=None):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    if fusion == 'seq':

        c = channel_attention(cbam_feature, ratio, name)
        cbam_feature = multiply([cbam_feature, c], name='cbam_ch_'+name)
        p = spatial_attention(cbam_feature, name)
        cbam_feature = multiply([cbam_feature, p], name='cbam_sp_'+name)
        return cbam_feature
    else:
        c = channel_attention(cbam_feature, ratio, name)
        p = spatial_attention(cbam_feature, name)
        att_coef = multiply([c, p], name='cbam_att_'+name)
        att_coef = Activation('sigmoid', name='cbam_atv_'+name)(att_coef)
        return att_coef

def cbam_block_v2(cbam_feature, ratio=16, fusion='seq', name=None):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    if fusion == 'seq':

        c = channel_attention(cbam_feature, ratio, name)
        cbam_feature = multiply([cbam_feature, c], name='cbam_ch_'+name)
        p = spatial_attention(cbam_feature, name)
        cbam_feature = multiply([cbam_feature, p], name='cbam_sp_'+name)
        return cbam_feature
    else:
        c = channel_attention(cbam_feature, ratio, name)
        p = spatial_attention(cbam_feature, name)
        att_coef = multiply([c, p], name='cbam_att_'+name)
        att_coef = Activation('softmax', name='cbam_atv_'+name)(att_coef)
        return att_coef

def channel_attention(input_feature, ratio=16, name=None):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape.as_list()[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape.as_list()[1:] == [1, 1, channel]
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape.as_list()[1:] == [1, 1, channel // ratio]
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape.as_list()[1:] == [1, 1, channel]

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape.as_list()[1:] == [1, 1, channel]
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape.as_list()[1:] == [1, 1, channel // ratio]
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape.as_list()[1:] == [1, 1, channel]

    cbam_feature = Add(name='ch_add_'+name)([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    return cbam_feature
    # return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature, name=None):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape.as_list()[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape.as_list()[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape.as_list()[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape.as_list()[-1] == 1
    concat = Concatenate(axis=3, name='sp_conca_'+name)([avg_pool, max_pool])
    assert concat.shape.as_list()[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name='sp_conv_'+name)(concat)
    assert cbam_feature.shape.as_list()[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    return cbam_feature


def FocusAttnGatingBlock(x, g, inter_shape, fus, name):
    """ take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same feature channels (theta_x)
    then, upsampling g to be same size as x
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients"""

    shape_x = x.shape.as_list()  # 32
    shape_g = g.shape.as_list()  # 16 gate

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', name='xl_' + name)(x)  # 16
    shape_theta_x = theta_x.shape.as_list()

    phi_g = Conv2D(inter_shape, (1, 1), padding='same', name='conv_g_'+name)(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same',
                                 name='g_up_' + name)(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x], name='concat_'+name)
    act_xg = Activation('relu')(concat_xg)

    att_feat = cbam_block(act_xg, 16, fus, name)
    shape_att_feat = att_feat.shape.as_list()
    upsample_psi = Conv2DTranspose(inter_shape, (3, 3),
                                   strides=(shape_x[1] // shape_att_feat[1], shape_x[2] // shape_att_feat[2]),
                                   padding='same', name='psi_up'+name)(att_feat)  # 500

    y = multiply([upsample_psi, x], name='q_attn' + name)

    return y

def FocusAttnGatingBlock_v2(x, g, inter_shape, fus, name):
    """ take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same feature channels (theta_x)
    then, upsampling g to be same size as x
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients"""

    shape_x = x.shape.as_list()  # 32
    # shape_g = g.shape.as_list()  # 32 gate

    # theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', name='xl_' + name)(x)  # 16
    # shape_theta_x = theta_x.shape.as_list()

    phi_g = Conv2D(inter_shape, (1, 1), padding='same', name='conv_g_'+name)(g)
    # upsample_g = Conv2DTranspose(inter_shape, (3, 3),
    #                              strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
    #                              padding='same',
    #                              name='g_up_' + name)(phi_g)  # 16

    concat_xg = add([phi_g, x], name='concat_'+name)
    act_xg = Activation('relu')(concat_xg)

    att_feat = cbam_block_v2(act_xg, 16, fus, name)
    # shape_att_feat = att_feat.shape.as_list()
    # upsample_psi = Conv2DTranspose(inter_shape, (3, 3),
    #                                strides=(shape_x[1] // shape_att_feat[1], shape_x[2] // shape_att_feat[2]),
    #                                padding='same', name='psi_up'+name)(att_feat)  # 500

    y = multiply([att_feat, x], name='q_attn' + name)

    return y