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


def depth_branch_fusenet(hiper):
    """
    fusenet vgg16 depth branch architecture
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :return: depth branch encoder model
    """
    # depth branch
    input2 = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3), name='DEPTH_input')  # depth input
    # block 1
    d = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock1_conv1')(input2)
    d1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock1_conv2')(d)  # first out

    d = layers.MaxPooling2D((2, 2), strides=(2, 2), name='dblock1_pool')(d1)

    # Block 2
    d = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock2_conv1')(d)
    d2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock2_conv2')(d)  # second out
    d = layers.MaxPooling2D((2, 2), strides=(2, 2), name='dblock2_pool')(d2)

    # Block 3
    d = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock3_conv1')(d)
    d = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock3_conv2')(d)
    d3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock3_conv3')(d)  # third out
    d = layers.MaxPooling2D((2, 2), strides=(2, 2), name='dblock3_pool')(d3)

    # Block 4
    d = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock4_conv1')(d)
    d = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock4_conv2')(d)
    d4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock4_conv3')(d)  # fourth out
    d = layers.MaxPooling2D((2, 2), strides=(2, 2), name='dblock4_pool')(d4)

    # Block 5
    d = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock5_conv1')(d)
    d = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock5_conv2')(d)
    d5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='dblock5_conv3')(d)  # fifth out

    return Model(inputs=input2, outputs=[d1, d2, d3, d4, d5])

def depth_branch_resnet50(hiper):
    """
    resnet50 depth branch architecture
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :return: depth branch encoder model
    """

    input2 = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3), name='DEPTH_input')  # depth input

    d = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='dconv1_pad')(input2)
    d = layers.Conv2D(64, 7, strides=2, name='dconv1_conv',
                      kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(d)
    d = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='dconv1_bn',
                                  beta_regularizer=hiper.regularizer, gamma_regularizer=hiper.regularizer)(d)
    d1 = layers.Activation('relu', name='dconv1_relu')(d)

    d = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='dpool1_pad')(d1)
    d = layers.MaxPooling2D(3, strides=2, name='dpool1_pool')(d)

    d2 = stack1(d, 64, 3, hiper, stride1=1, name='dconv2')
    d3 = stack1(d2, 128, 4, hiper, name='dconv3')
    d4 = stack1(d3, 256, 6, hiper, name='dconv4')
    d5 = stack1(d4, 512, 3, hiper, name='dconv5')

    return Model(inputs=input2, outputs=[d1, d2, d3, d4, d5])

def depth_branch_tcd_resnet50(rgb: Model, hiper):
    """
    resnet50 depth branch architecture
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :return: depth branch encoder model
    """

    input2 = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3), name='DEPTH_input')  # depth input

    d = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='dconv1_pad')(input2)
    d = layers.Conv2D(64, 7, strides=2, name='dconv1_conv',
                      kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(d)
    d = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='dconv1_bn',
                                  beta_regularizer=hiper.regularizer, gamma_regularizer=hiper.regularizer)(d)
    d_blk = layers.Activation('relu', name='dconv1_relu')(d)
    d0 = fus_opt(hiper.fus_str, d_blk, rgb.output[0], 'd_add')

    d_ = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='dpool1_pad')(d0)
    d_ = layers.MaxPooling2D(3, strides=2, name='dpool1_pool')(d_)

    d_blk1 = stack1(d_, 64, 3, hiper, stride1=1, name='dconv2')
    d1 = fus_opt(hiper.fus_str, d_blk1, rgb.output[1], 'd_add_1')
    d_blk2 = stack1(d1, 128, 4, hiper, name='dconv3')
    d2 = fus_opt(hiper.fus_str, d_blk2, rgb.output[2], 'd_add_2')
    d_blk3 = stack1(d2, 256, 6, hiper, name='dconv4')
    d3 = fus_opt(hiper.fus_str, d_blk3, rgb.output[3], 'd_add_3')
    d_blk4 = stack1(d3, 512, 3, hiper, name='dconv5')
    d4 = fus_opt(hiper.fus_str, d_blk4, rgb.output[4], 'd_add_4')

    return Model(inputs=[rgb.inputs, input2], outputs=[d0, d1, d2, d3, d4])

def rgb_branch_fusenet_vgg16(depth: Model, hiper):
    """
    fusenet vgg16 rgb branch architecture
    :param depth: depth branch keras model
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :return: fusenet vgg16 rgb/ddd encoder keras model
    """

    # RGB branch
    input1 = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3), name='RGB_input')  # RGB input
    # block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block1_conv1')(input1)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block1_conv2')(x)

    # x = layers.add([depth.output[0], x], name='d_add')  # first fusion
    x = fus_opt(hiper.fus_str, depth.output[0], x, 'd_add')
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block2_conv2')(x)
    # x = layers.add([depth.output[1], x], name='d_add_1')  # second fusion
    x = fus_opt(hiper.fus_str, depth.output[1], x, 'd_add_1')
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block3_conv3')(x)
    # x = layers.add([depth.output[2], x], name='d_add_2')  # third fusion
    x = fus_opt(hiper.fus_str, depth.output[2], x, 'd_add_2')
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block4_conv3')(x)
    # x = layers.add([depth.output[3], x], name='d_add_3')  # fourth fusion
    x = fus_opt(hiper.fus_str, depth.output[3], x, 'd_add_3')
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=hiper.initializer,
                      kernel_regularizer=hiper.regularizer, name='block5_conv3')(x)
    # x = layers.add([depth.output[4], x], name='d_add_4')  # fifth fusion
    x = fus_opt(hiper.fus_str, depth.output[4], x, 'd_add_4')
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return Model(inputs=[input1, depth.inputs], outputs=x)


def rgb_branch_resnet50(depth: Model, hiper):
    """
    resnet50 rgb branch architecture
    :param depth: depth branch keras model
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :return: resnet50 rgb/ddd encoder keras model
    """

    input1 = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3), name='RGB_input')  # RGB input

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(input1)
    x = layers.Conv2D(64, 7, strides=2, name='conv1_conv',
                      kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(x)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn',
                                  beta_regularizer=hiper.regularizer, gamma_regularizer=hiper.regularizer)(x)
    x = layers.Activation('relu', name='conv1_relu')(x)

    # x = layers.add([depth.output[0], x], name='d_add')  # first fusion
    x = fus_opt(hiper.fus_str, depth.output[0], x, 'd_add')

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack1(x, 64, 3, hiper, stride1=1, name='conv2')
    # x = layers.add([depth.output[1], x], name='d_add_1')  # second fusion
    x = fus_opt(hiper.fus_str, depth.output[1], x, 'd_add_1')
    x = stack1(x, 128, 4, hiper, name='conv3')
    # x = layers.add([depth.output[2], x], name='d_add_2')  # third fusion
    x = fus_opt(hiper.fus_str, depth.output[2], x, 'd_add_2')
    x = stack1(x, 256, 6, hiper, name='conv4')
    # x = layers.add([depth.output[3], x], name='d_add_3')  # fourth fusion
    x = fus_opt(hiper.fus_str, depth.output[3], x, 'd_add_3')
    x = stack1(x, 512, 3, hiper, name='conv5')
    # x = layers.add([depth.output[4], x], name='d_add_4')  # fifth fusion
    x = fus_opt(hiper.fus_str, depth.output[4], x, 'd_add_4')

    return Model(inputs=[input1, depth.inputs], outputs=x)

def rgb_branch_tcd_resnet50(hiper):
    """
    resnet50 rgb branch architecture
    :param hiper: choice of initializer, regularizer, activation and batch normalization option
    :return: resnet50 rgb encoder keras model
    """

    input1 = Input(shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3), name='RGB_input')  # RGB input

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(input1)
    x = layers.Conv2D(64, 7, strides=2, name='conv1_conv',
                      kernel_initializer=hiper.initializer, kernel_regularizer=hiper.regularizer)(x)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn',
                                  beta_regularizer=hiper.regularizer, gamma_regularizer=hiper.regularizer)(x)
    x1 = layers.Activation('relu', name='conv1_relu')(x)

    # x = layers.add([depth.output[0], x], name='d_add')  # first fusion
    # x = fus_opt(hiper.fus_str, depth.output[0], x, 'd_add')

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x1)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x2 = stack1(x, 64, 3, hiper, stride1=1, name='conv2')
    # x = layers.add([depth.output[1], x], name='d_add_1')  # second fusion
    # x = fus_opt(hiper.fus_str, depth.output[1], x, 'd_add_1')
    x3 = stack1(x2, 128, 4, hiper, name='conv3')
    # x = layers.add([depth.output[2], x], name='d_add_2')  # third fusion
    # x = fus_opt(hiper.fus_str, depth.output[2], x, 'd_add_2')
    x4 = stack1(x3, 256, 6, hiper, name='conv4')
    # x = layers.add([depth.output[3], x], name='d_add_3')  # fourth fusion
    # x = fus_opt(hiper.fus_str, depth.output[3], x, 'd_add_3')
    x5 = stack1(x4, 512, 3, hiper, name='conv5')
    # x = layers.add([depth.output[4], x], name='d_add_4')  # fifth fusion
    # x = fus_opt(hiper.fus_str, depth.output[4], x, 'd_add_4')

    return Model(inputs=[input1], outputs=[x1, x2, x3, x4, x5])

# dep = depth_branch_resnet50(hiper)
# fuse = rgb_branch_resnet50(dep, hiper)
# fuse.summary()
# utils.plot_model(fuse, show_shapes=True)

# print('chega')