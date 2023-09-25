import tensorflow as tf
from tensorflow.keras import Model, layers
import os, tempfile
from tensorflow.keras.optimizers import Adam, SGD

def conv_block(x, hiper, filters, kernel_size, name):
    """
    Convolution + Batch Normalization + Activation layers in conv_block
    :param x: keras layer
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :param filters: number of conv features (channel dimension)
    :param kernel_size: 3x3 or 5x5 or 7x7
    :return: keras layer (conv + bn + act)
    """
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name=name+'_conv_')(x)
    if hiper.BN:
        x = layers.BatchNormalization(name=name+'_bn_')(x)
    x = layers.Activation(hiper.activation, name=name+'_activ_')(x)
    return x


def separable_conv_block(x, hiper, filters, name):
    """
    Separable Depthwise convolution block. 3x3 Depthwise conv + Batch Normalization + activation + conv_block (pointwise convolution).
    :param x: keras layer
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :param filters: number of conv features (channel dimension)
    :return: keras layer (separable depthwise conv)
    """
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', depthwise_initializer=hiper.initializer,
                               depthwise_regularizer=hiper.regularizer, name=name+'_depthconv_')(x)
    if hiper.BN:
        x = layers.BatchNormalization(name=name+'_bn0_')(x)
    x = layers.Activation(hiper.activation, name=name+'_activ0_')(x)
    x = conv_block(x, hiper, filters, 1, name)
    return x

# types of upsampling
def upsampling(choose, *kwargs):
    MAPUP = {
        'up_conv': up_conv,
        'transp_conv': transp_conv,
    }
    if choose not in MAPUP:
        raise ValueError('type of upsampling error')

    return MAPUP[choose](*kwargs)

def up_conv(x, hiper, filters, name):
    # up = layers.UpSampling2D(2, interpolation=hiper.interpolation)(x)
    # up = conv_block(x, hiper, filters, 3, name)
    up = layers.UpSampling2D(2, interpolation=hiper.interpolation, name=name+'_upsample_')(x)
    return up

def transp_conv (x, hiper, filters, name):
    x = layers.Conv2DTranspose(kernel_initializer=hiper.initializer, filters=filters, kernel_size=2, strides=2,
                               padding='same', kernel_regularizer=hiper.regularizer, name=name+'_transp_conv_')(x)
    return x

# types of skipconnect
def skipconnect(choose, *kwargs):
    MAPCONECT = {
        'add': add,
        'concat': concat,
        'conv1x1': conv1x1,
    }
    if choose not in MAPCONECT:
        raise ValueError('type of connection error')

    return MAPCONECT[choose](*kwargs)

def add(x, encoder, name):
    x = layers.add([encoder.get_layer(name=name).output, x], name=name+'_add_decod')
    return x

def concat(x, encoder, name, hiper):
    x = layers.concatenate([encoder.get_layer(name=name).output, x], axis=3, name=name+'_concat_decod')
    if hiper.drop:
        x = layers.Dropout(hiper.droprate)(x)
    return x

def conv1x1(x, encoder, name, hiper, filters):
    y = conv_block(encoder.get_layer(name=name).output, hiper, filters, 1, name=name+'_conv1x1_decod')
    x = layers.add([y, x], name=name+'_add_decod')
    return x

def CFB(fh: layers, fl, name, hiper):
    """fl is the skipconnection features (low-level) and fh is decoder feature (high-level) """
    channels = fh.shape.as_list()[-1]
    fh_ = fh  # (b,h,w,c)
    fl_ = fl  # (b,h,w,c)

    g_fh = layers.GlobalAveragePooling2D()(fh_)   # (b,c)
    g_fh = layers.Reshape((1, 1, channels))(g_fh)  # (b,1,1,c)
    g_fl = layers.GlobalAveragePooling2D()(fl_)   # (b,c)
    g_fl = layers.Reshape((1, 1, channels))(g_fl)   # (b,1,1,c)

    fcl_shared = tf.keras.Sequential(
      [
       layers.Dense(units=channels, activation='relu', name=name+'_fcl_1'),
       layers.Dense(units=channels, activation='sigmoid', name=name+'_fcl_2'),
      ]
    )

    fcl_fh = fcl_shared(g_fh)    # (b,1,1,c)
    fcl_fl = fcl_shared(g_fl)    # (b,1,1,c)

    fh_m = layers.multiply([fh_, fcl_fl], name=name+'_fhxfl')  # (b,h,w,c)
    fl_m = layers.multiply([fl_, fcl_fh], name=name+'_flxfh')   # (b,h,w,c)
    cat = layers.concatenate([fh_m, fl_m], name=name+'_concat')  # (b,h,w,2c)

    cat = layers.Conv2D(channels, 3, padding='same', name=name+'_conv1',
                        kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(cat)
    cat = layers.BatchNormalization(name=name+'_bn1')(cat)
    cat = layers.Activation(hiper.activation, name=name+'_actv1')(cat)
    cat = layers.Conv2D(channels, 3, padding='same', name=name+'_conv2',
                        kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(cat)
    cat = layers.BatchNormalization(name=name+'_bn2')(cat)
    cat = layers.Activation(hiper.activation, name=name+'_actv2')(cat)

    return layers.add([cat, fh_], name=name+'_add')  # (b,h,w,c)

def add_l2_regularizer(model_temp: Model, hiper):
    # Add L2 regularization to the whole model.
    # credits: https://github.com/keras-team/keras/issues/2717

    if hiper.regularizer is None:
        return model_temp

    reg_attributes = ['kernel_regularizer']
    # reg_attributes = ['kernel_regularizer', 'bias_regularizer', bag-of-tricks-for-image-classification!!!
    #                   'beta_regularizer', 'gamma_regularizer']
    # if isinstance(reg_attributes, str):
    #     reg_attributes = [reg_attributes]

    regularize = hiper.regularizer

    for layer in model_temp.layers:
        for attr in reg_attributes:
            if hasattr(layer, attr):
                setattr(layer, attr, regularize)

    # So far, the regularizers only exist in the model config. We need to
    # reload the model so that Keras adds them to each layer's losses.
    model_json = model_temp.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model_temp.save_weights(tmp_weights_path)

    # Reload the model
    model_temp = tf.keras.models.model_from_json(model_json)
    model_temp.load_weights(tmp_weights_path, by_name=True)

    return model_temp

# credits : keras.io site

class GC_SGD(SGD):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads

class GC_ADAM(Adam):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads

def opt_model(fn, hiper):
    if hiper.optimizer == 'sgd':
        if hiper.gc:
            # opt = gc_opt.sgd(learning_rate=fn, momentum=0.9)
            opt = GC_SGD(learning_rate=fn, momentum=0.9)

        else:
            opt = tf.keras.optimizers.SGD(learning_rate=fn, momentum=0.9)

        return opt
    else:
        if hiper.gc:
            # opt = gc_opt.adam(learning_rate=fn)
            opt = GC_ADAM(learning_rate=fn)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=fn)

        return opt

