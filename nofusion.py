import tensorflow as tf
import os, datetime, glob, yaml
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger, LearningRateScheduler, EarlyStopping
import numpy as np
np.random.seed(1)
tf.random.set_seed(123)
import no_fusion.loss_metric_custom as loss_var
from attrdict import AttrDict
from no_fusion.hiper_ import Hyperparameters
from no_fusion.load_data import test_data, valid_data, train_data, convert_text_img_dataset_train, \
    convert_text_img_dataset_val_test
from no_fusion.decoder_vgg import build_decoder_vgg1, build_decoder_vgg2, build_decoder_vgg3, build_decoder_vgg4
from no_fusion.decoder_resnet import build_decoder_resnet1, build_decoder_resnet2, build_decoder_resnet3, \
    build_decoder_resnet4, build_decoder_resnet5
from no_fusion.utils_ import add_l2_regularizer, opt_model
from no_fusion.models.build_unet_encoder import encoder_model, sota

AUTOTUNE = tf.data.experimental.AUTOTUNE


def environment():
    """memory management"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


environment()

""" load Hyperparameters -------------------------------------------"""

with open('nofusion.yaml', 'r') as file:
    param_ = yaml.load(file, Loader=yaml.FullLoader)

for k in param_.keys():
    param = AttrDict(param_[k])

    hiper = Hyperparameters(lr=param.lr, IMG_WIDTH=param.IMG_WIDTH, IMG_HEIGHT=param.IMG_HEIGHT,
                            batch_size=param.batch_size, epochs=param.epochs,
                            activation=param.activation, step=param.step, inicialization=param.inicialization,
                            loss=param.loss, drop=param.drop, dropout_rate=param.dropout_rate, backbone=param.backbone,
                            fine_tuning=param.fine_tuning, optimizer=param.optimizer, gc=param.gradientcentral,
                            path=param.path, aug=param.aug, activationfinal=param.activationfinal,
                            alpha=param.alpha, beta=param.beta, interpolation=param.interpolation,
                            decoderchoose=param.decoderchoose, regularizer=param.regularizer,
                            val_visual=param.img_visual, loss1=param.loss1, loss2=param.loss2,
                            schedule=param.schedule, architecture=param.architecture, ds=param.ds, loss3=param.loss3,
                            w=param.w, w1=param.w1, w2=param.w2, w3=param.w3, sched_ch=param.schedule_choose,
                            lr_limit=param.lr_limit)

    os.chdir(hiper.path)
    path = os.getcwd()
    loss_var.alpha = hiper.alpha
    # loss_var.theta = hiper.theta
    loss_var.beta = hiper.beta
    tf.keras.backend.clear_session()

    """ model builder -------------------------------------------"""


    def decoder_choose(choose, *kwargs):
        DEC = {
            'vgg_up_conv': build_decoder_vgg1,
            'vgg_transp': build_decoder_vgg2,
            'vgg_up_conv_concat': build_decoder_vgg3,
            'vgg_transp_concat': build_decoder_vgg4,
            'resnet_up_conv': build_decoder_resnet1,
            'resnet_transp': build_decoder_resnet2,
            'resnet_up_conv_concat': build_decoder_resnet3,
            'resnet_transp_concat': build_decoder_resnet4,
            'resnet_cfb': build_decoder_resnet5,
        }
        if choose not in DEC:
            raise ValueError('decoder choose error')

        return DEC[choose](*kwargs)


    if hiper.architecture == 'unet':
        # base encoder model
        encoder_ = encoder_model(hiper)
        model = decoder_choose(hiper.decoderchoose, encoder_, hiper)
    else:
        model = sota(hiper)

    model = add_l2_regularizer(model, hiper)  # insert l2norm if != None
    # model.compile(optimizer=optimizer, loss=hiper.loss, metrics=hiper.metrics)
    tf.keras.utils.plot_model(model, show_shapes=True)
    # model.summary()

    """ data loader -------------------------------------------"""

    train = train_data(path)
    val = valid_data(path)
    test = test_data(path)

    """ --------------------------"""
    # checkpoint
    n_train = len(list(train.as_numpy_iterator()))
    n_val = len(list(val.as_numpy_iterator()))
    n_test = len(list(test.as_numpy_iterator()))
    print(n_train, n_val, n_test)


    # def check_data(dataset):
    #     dataset = list(dataset.as_numpy_iterator())
    #     print(dataset)
    #
    #
    # check_data(val)
    steps = int(n_train / hiper.batch_size) * hiper.step

    """ --------------------------"""

    # load images from list_files
    train = convert_text_img_dataset_train(train, hiper.batch_size, AUTOTUNE, hiper.IMG_WIDTH, hiper.IMG_HEIGHT,
                                           hiper.aug, hiper.ds)

    val = convert_text_img_dataset_val_test(val, hiper.batch_size, AUTOTUNE, hiper.IMG_WIDTH, hiper.IMG_HEIGHT,
                                            hiper.ds)

    test = convert_text_img_dataset_val_test(test, hiper.batch_size, AUTOTUNE, hiper.IMG_WIDTH, hiper.IMG_HEIGHT,
                                             hiper.ds)

    """ logging -------------------------------------------"""

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # schedule

    if hiper.epochs < 5:
        limit = 5
    else:
        limit = (hiper.epochs // 5)


    # split iterations in 5 parts, 3 parts with same lr and 2 last parts with lower lr
    def step_decay_train(epoch, lr):
        if epoch < limit * 3:
            return lr
        else:
            # return lr * tf.math.exp(-0.1)
            if epoch in [limit * 3, limit * 4]:
                return tf.math.multiply(lr, 0.1)  # 2 steps  decay
            else:
                return lr


    def poly_decay_train(epoch, lr):

        epoch = min(epoch, hiper.epochs)
        lr = ((hiper.lr - hiper.lr_limit) * tf.math.pow((1 - epoch / hiper.epochs), 0.9)) + hiper.lr_limit
        return lr


    checkpointer = ModelCheckpoint(path + '/' + logdir + '/weight_best.hdf5',
                                   verbose=1, save_best_only=True, save_freq='epoch')
    # reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, min_delta=1e-4)
    # early = EarlyStopping(monitor='val_loss', patience=15)
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=5, write_graph=False,
                              write_images=False, update_freq='epoch', profile_batch=0, embeddings_freq=0)
    csv_logger = CSVLogger(path + '/' + logdir + '/train.csv', separator=',', append=False)

    callbacks = [tensorboard, checkpointer, csv_logger]

    file_writer_img = tf.summary.create_file_writer(logdir)


    def display_img(epoch, logs):
        # model_vis = tf.keras.models.clone_model(model)
        # model_vis.load_weights(path + '/' + logdir + '/weight_best.hdf5')
        img = model.predict(val)
        orig = val.unbatch().map(lambda x, y: x)
        orig = list(orig.as_numpy_iterator())
        orig = np.stack(orig)
        # orig_m = val.unbatch().map(lambda x, y: x)
        # orig_m = list(orig_m.as_numpy_iterator())
        # orig_m = np.stack(orig_m)
        with file_writer_img.as_default():
            tf.summary.image("Training data1", img, step=epoch, max_outputs=10)
            tf.summary.image("original data", orig, step=0, max_outputs=10)


    img_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=display_img)

    def get_img(x, y):
        return x['RGB_input']

    def display_img_2ds(epoch, logs):
        img = model.predict(val)
        orig = val.unbatch().map(lambda x, y: get_img(x, y))
        # orig = aux.map(lambda x, y: x)
        orig = list(orig.as_numpy_iterator())
        orig = np.stack(orig)
        with file_writer_img.as_default():
            tf.summary.image("Training data1", img[0], step=epoch, max_outputs=10)
            tf.summary.image("Training data2", img[1], step=epoch, max_outputs=10)
            tf.summary.image("Training data3", img[2], step=epoch, max_outputs=10)
            tf.summary.image("Training data4", img[3], step=epoch, max_outputs=10)
            tf.summary.image("original data", orig, step=0, max_outputs=10)


    img_callback_2ds = tf.keras.callbacks.LambdaCallback(on_epoch_end=display_img_2ds)

    if hiper.schedule:
        if hiper.sched_ch == 'step':
            schedule = LearningRateScheduler(schedule=step_decay_train, verbose=2)
        else:
            schedule = LearningRateScheduler(schedule=poly_decay_train, verbose=2)
        callbacks.append(schedule)

    if hiper.ds!=0:
        loss = {'output_inter1': hiper.loss1, 'output_inter2': hiper.loss2, 'output_inter3': hiper.loss3, 'output_final': hiper.loss}
        w = {'output_inter1': hiper.w1, 'output_inter2': hiper.w2, 'output_inter3': hiper.w3, 'output_final': hiper.w}
        if hiper.visual:
            callbacks.append(img_callback_2ds)
    else:
        loss = hiper.loss
        w = None
        if hiper.visual:
            callbacks.append(img_callback)

    """ model fit ------------------------------------------- """

    if hiper.finetuning:
        schedule_fn = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10,
                                                                     decay_rate=0.96)
        optimizer = opt_model(schedule_fn, hiper)
        model.compile(optimizer=optimizer, loss=loss, metrics=hiper.metrics, loss_weights=w)
        history_ = model.fit(train, epochs=5, steps_per_epoch=(n_train // hiper.batch_size), validation_data=val,
                             verbose=2)
        model.trainable = True
        model.compile(optimizer=opt_model(hiper.lr, hiper), loss=loss, metrics=hiper.metrics, loss_weights=w)

        history_ = model.fit(train, epochs=hiper.epochs, steps_per_epoch=steps,
                             validation_data=val, callbacks=callbacks, verbose=2)

    else:
        model.compile(optimizer=opt_model(hiper.lr, hiper), loss=loss, metrics=hiper.metrics, loss_weights=w)
        history_ = model.fit(train, epochs=hiper.epochs, steps_per_epoch=steps, validation_data=val,
                             callbacks=callbacks, verbose=2)

    """ evaluation -------------------------------------------"""
    evaluate_ = model.evaluate(test, verbose=1, return_dict=True)
    model.save(filepath=path + '/' + logdir + '/weight_final.h5')

    # load weights
    weights = glob.glob(path + '/' + logdir + '/*.hdf5')
    weights.sort()
    count = len(weights)

    model.load_weights(weights[count - 1])
    evaluate__ = model.evaluate(test, verbose=1, return_dict=True)

    os.chdir(path + '/' + logdir)

    with open('test_final.csv', 'w') as f:
        for key in evaluate_.keys():
            f.write("%s,%s\n" % (key, evaluate_[key]))
    with open('test.csv', 'w') as f:
        for key in evaluate__.keys():
            f.write("%s,%s\n" % (key, evaluate__[key]))

    # model.save('model')

if __name__ == '__main__':
    environment()
