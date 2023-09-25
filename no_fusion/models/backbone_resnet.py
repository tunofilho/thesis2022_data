import tensorflow as tf

def resnet(hiper):
    if hiper.finetuning:
        weights='imagenet'
        train = False
    else:
        weights=None
        train = True

    if hiper.backbone == 'resnet34':
        base = tf.keras.models.load_model('/Users/antonioguimaraesfilho/PycharmProjects/fusion_tf24/resnet34_quvel.h5', compile=False)
        # x = tf.keras.layers.Input((hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3))
        # y = temp(x)
        # base = tf.keras.Model(inputs=x, outputs=y)
        base.trainable=train

    if hiper.backbone == 'resnet50':
        base = tf.keras.applications.ResNet50(include_top=False, weights=weights,
                                           input_shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3))
        base.trainable=train

    if hiper.backbone == 'resnet101':
        base = tf.keras.applications.ResNet101(include_top=False, weights=weights,
                                           input_shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3))
        base.trainable=train

    if hiper.backbone == 'resnet152':
        base = tf.keras.applications.ResNet152(include_top=False, weights=weights,
                                           input_shape=(hiper.IMG_WIDTH, hiper.IMG_HEIGHT, 3))
        base.trainable=train

    return base
