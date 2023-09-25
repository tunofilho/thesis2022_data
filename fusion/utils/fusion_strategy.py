from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from fusion.utils.focus_gate import cbam_block
from fusion.models_fusenet_inputs.af_net import att_cos_mod

def only_add(depth_model, x, name):
    out = layers.add([depth_model, x], name=name)
    return out

def DFFM(depth_model, x, name):
    fr_sig = layers.Activation('sigmoid')(x)
    mult1 = layers.Multiply()([x, fr_sig])
    mult2 = layers.Multiply()([depth_model, fr_sig])
    out = layers.add([x, mult1, mult2], name=name)
    return out

# credits: https://github.com/titu1994/keras-squeeze-excite-network
def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    # channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # filters = init._keras_shape[channel_axis]
    filters = init.shape.as_list()[-1]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='glorot_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='glorot_normal', use_bias=False)(se)

    # if K.image_data_format() == 'channels_first':
    #     se = Permute((3, 1, 2))(se)

    x = layers.multiply([init, se])
    return x

def esanet_seblock(depth_model, x, name):
    d = squeeze_excite_block(depth_model)
    r = squeeze_excite_block(x)
    out = layers.add([d, r], name=name)
    return out

def get_cbam(depth_model, x, name):
    d = cbam_block(depth_model, fusion='par', name='depth_')
    r = cbam_block(x, fusion='par', name='rgb_')
    out = layers.add([d, r], name=name)
    return out

def att_cosine(depth_model, x, name):
    r = x  #rgb
    d = depth_model  #depth
    att_coef1 = att_cos_mod(r, d, 'softmax', name+'cos_r_d')
    att_coef2 = att_cos_mod(d, r, 'softmax', name+'cos_d_r')
    out = layers.Add(name=name)([att_coef1, att_coef2])
    return out