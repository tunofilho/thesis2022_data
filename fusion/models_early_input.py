from tensorflow.keras import layers, Input, Model, applications, utils
from fusion.utils import conv_block, separable_conv_block, stack1
from fusion.utils import fusion_strategy


def fus_opt(choose, *args):
    opt = {'add': fusion_strategy.only_add,
           'dffm': fusion_strategy.DFFM,
           'esa_fus': fusion_strategy.esanet_seblock,
           'cbam': fusion_strategy.get_cbam,
           'att_cos': fusion_strategy.att_cosine}
    return opt[choose](*args)


# early inputs

def early_depth_branch_vgg(hiper):
    """
    early fusion convolution block
    :param op: block connection - vgg or resnet
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :return: keras model early fusion block
    """
    # depth branch
    input2 = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3), name='DEPTH_input')  # depth input
    # d = conv_block(input2, hiper, 64, 3, 'early_blk1')
    # d = conv_block(d, hiper, 64, 3, 'early_blk2')
    d = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock1_conv1')(input2)
    d = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock1_conv2')(d)
    # if op=='resnet':
    #     d = layers.MaxPooling2D((2, 2), strides=(2, 2), name='dblock1_pool')(d)

    return Model(input2, d)

def early_depth_branch_resnet(hiper):
    input2 = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3), name='DEPTH_input')  # RGB input

    d = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='dconv1_pad')(input2)
    d = layers.Conv2D(64, 7, strides=2, name='dconv1_conv',
                      kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(d)
    d = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='dconv1_bn',
                                  beta_regularizer=hiper.regularizer, gamma_regularizer=hiper.regularizer)(d)
    d = layers.Activation('relu', name='dconv1_relu')(d)

    return Model(input2, d)

def vgg_16_earlyfusion(depth, hiper):
    """
    vgg16 rgb branch architecture
    :param depth: depth branch keras model
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :return: vgg16 early fusion encoder keras model
    """

    # RGB branch
    input1 = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3), name='RGB_input')  # RGB input
    # block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block1_conv1')(input1)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block1_conv2')(x)

    # x = layers.add([depth.output, x], name='add')  # early fusion!!!
    x = fus_opt(hiper.fus_str, depth.output, x, 'd_add')

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block5_conv3')(x)
    out = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return Model(inputs=[input1, depth.inputs], outputs=out)


def resnet50_earlyfusion(depth, hiper):
    """
    resnet rgb branch architecture
    :param depth: depth branch keras model
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :return: resnet early fusion encoder keras model
    """

    input1 = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3), name='RGB_input')  # RGB input

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(input1)
    x = layers.Conv2D(64, 7, strides=2, name='conv1_conv',
                      kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(x)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn',
                                  beta_regularizer=hiper.regularizer, gamma_regularizer=hiper.regularizer)(x)
    x = layers.Activation('relu', name='conv1_relu')(x)

    # x = layers.add([depth.output, x], name='add')  # early fusion!!!
    x = fus_opt(hiper.fus_str, depth.output, x, 'd_add')

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack1(x, 64, 3, hiper, stride1=1, name='conv2')
    x = stack1(x, 128, 4, hiper, name='conv3')
    x = stack1(x, 256, 6, hiper, name='conv4')
    x = stack1(x, 512, 3, hiper, name='conv5')

    return Model(inputs=[input1, depth.inputs], outputs=x)


# branch = early_depthwise_branch('resnet', hiper)
# test = resnet50_earlyfusion(branch, hiper)
# utils.plot_model(test, show_shapes=True)
# test.summary()
# test2 = applications.ResNet50(False)
# test2.save_weights('weights/res50.hdf5')
# test.load_weights('weights/res50.hdf5', by_name=True)

# para finetuning -- top head training
# branch.trainable = False
# for i in test.layers:
#    if i.trainable:
#        i.trainable=False
# branch.trainable = True
# test.summary()

# print('chega')
