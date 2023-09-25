import tensorflow as tf
from no_fusion.loss_metric_custom import *
from tensorflow.keras import metrics, losses

act = {'relu': 'relu', 'elu': 'elu'}
k_init = {'glorot': tf.keras.initializers.GlorotNormal(seed=4), 'he': tf.keras.initializers.HeNormal(seed=4)}
losses = {'dice': dice_loss, 'bce_dice': bce_dice_loss, 'tversky': tversky_loss,
          'focal_tversky': focal_tversky_loss, 'bce': losses.BinaryCrossentropy(),
          'combo_dice': bce_dice_SSIM_loss, 'combo_focal_tversky': bce_focal_tversky_SSIM_loss,
          'cl_dice': soft_dice_cldice_loss, 'cl': soft_clDice_loss}
encoder = {'resnet34': 'resnet34', 'resnet50': 'resnet50', 'resnet101': 'resnet101', 'resnet152': 'resnet152',
           'vgg16': 'vgg16', 'vgg19': 'vgg19', 'scratch16': 'scratch16', 'scratch19': 'scratch19'}
actfinal = {'sigmoid': 'sigmoid', 'softmax': 'softmax'}
interpol = {'nearest': 'nearest', 'bilinear': 'bilinear'}
dchoose = {'resnet_up_conv': 'resnet_up_conv', 'resnet_transp': 'resnet_transp',
           'resnet_up_conv_concat': 'resnet_up_conv_concat', 'resnet_transp_concat': 'resnet_transp_concat',
           'resnet_cfb': 'resnet_cfb', 'resnet_up_conv_concat2': 'resnet_up_conv_concat2',
           'vgg_up_conv': 'vgg_up_conv', 'vgg_transp': 'vgg_transp',
           'vgg_up_conv_concat': 'vgg_up_conv_concat', 'vgg_transp_concat': 'vgg_transp_concat'}
regul = {'none': None, 'l2': tf.keras.regularizers.L2(0.0001)}

lr_sched = {'step': 'step', 'poly': 'poly'}



class Hyperparameters:

    def __init__(self, lr, IMG_WIDTH, IMG_HEIGHT, batch_size, epochs, activation, step, inicialization, loss,
                 drop, dropout_rate, backbone, fine_tuning, optimizer, gc, path, aug, activationfinal, alpha,
                 beta, interpolation, decoderchoose, regularizer, val_visual, schedule, architecture, ds, loss2, loss3, loss1, w,
                 w1, w2, w3, sched_ch, lr_limit):

        self.lr=lr
        self.schedule = schedule
        self.IMG_WIDTH=IMG_WIDTH
        self.IMG_HEIGHT=IMG_HEIGHT
        self.batch_size=batch_size
        self.epochs = epochs
        self.activation = act[activation]
        self.activationfinal = actfinal[activationfinal]
        self.step = step
        self.initializer = k_init[inicialization]
        self.drop = drop
        self.droprate = dropout_rate
        self.BN = True
        self.architecture = architecture
        self.backbone = encoder[backbone]
        self.alpha = alpha
        self.beta=beta
        self.finetuning = fine_tuning
        self.optimizer = optimizer
        self.gc = gc
        self.loss = losses[loss]
        self.loss1 = losses[loss1]
        self.loss2 = losses[loss2]
        self.loss3 = losses[loss3]
        self.interpolation = interpol[interpolation]
        self.decoderchoose = dchoose[decoderchoose]
        self.regularizer = regul[regularizer]
        self.path = path
        self.aug = aug
        self.metrics = [recall, precision, f1, 'accuracy', mean_iou, metrics.MeanIoU(num_classes=2), metrics.FalsePositives(), metrics.FalseNegatives()]
        self.visual = val_visual
        self.ds = ds
        self.w = w
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.sched_ch = lr_sched[sched_ch]
        self.lr_limit = lr_limit








