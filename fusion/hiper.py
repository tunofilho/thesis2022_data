from tensorflow.keras import initializers, regularizers, optimizers, metrics, losses
from fusion.custom_metric_loss import *
# from gctf import optimizers as gc_opt

typefusion = {'early': 'early',
              'unet': 'unet', 'fusenet_vgg16': 'fusenet_vgg16',
              'resnet': 'resnet', 'fg_resnet50': 'fg_resnet50',
              'tcd_resnet50': 'tcd_resnet50', 'af_resnet50': 'af_resnet50',
              'trisegnet_vgg16': 'trisegnet_vgg16', 'tcd_resnet50_v2': 'tcd_resnet50_v2',
              'earlyres':'earlyres'}

fusion_strg = {'add': 'add', 'dffm': 'dffm', 'esa_fus': 'esa_fus', 'cbam': 'cbam', 'att_cos': 'att_cos'}

act = {'relu': 'relu', 'elu': 'elu'}

k_init = {'glorot': initializers.GlorotNormal(seed=4), 'he': initializers.HeNormal(seed=4)}

losses_ = {'dice': dice_loss, 'bce_dice': bce_dice_loss, 'tversky': tversky_loss,
           'focal_tversky': focal_tversky_loss, 'bce': losses.BinaryCrossentropy(), 'combo_dice': bce_dice_SSIM_loss,
           'combo_focal_tversky': bce_focal_tversky_SSIM_loss}

encoder = {'resnet50': 'resnet50', 'vgg16': 'vgg16'}

actfinal = {'sigmoid': 'sigmoid', 'softmax': 'softmax'}

interpol = {'nearest': 'nearest', 'bilinear': 'bilinear'}

dchoose = {'resnet_up_conv': 'resnet_up_conv', 'resnet_transp': 'resnet_transp',
           'resnet_up_conv_concat': 'resnet_up_conv_concat', 'resnet_transp_concat': 'resnet_transp_concat',
           'resnet_cfb': 'resnet_cfb',
           'vgg_up_conv': 'vgg_up_conv', 'vgg_transp': 'vgg_transp',
           'vgg_up_conv_concat': 'vgg_up_conv_concat', 'vgg_transp_concat': 'vgg_transp_concat',
           'fg_unet': 'fg_unet', 'tcd_net': 'tcd_net', 'af_net': 'af_net', 'af_net_v2': 'af_net_v2',
           'af_net_v3': 'af_net_v3', 'af_net_v4': 'af_net_v4', 'fusenet': 'fusenet',
           'trisegnet': 'trisegnet', 'earlyres_dec': 'earlyres_dec'}

regul = {'none': None, 'l2': regularizers.L2(0.0001)}

lr_sched = {'step': 'step', 'poly': 'poly'}


class Hyperparameters:

    def __init__(self, lr, IMG_WIDTH, IMG_HEIGHT,  batch_size, epochs, activation, step, inicialization, loss,
                 drop, dropout_rate, backbone, fine_tuning, optimizer, gc, path, aug, activationfinal, alpha,
                 interpolation, decoderchoose, regularizer, val_visual, type_fusion, schedule, fus_str,
                 ds, loss2, w, w2, sched_ch, lr_limit):

        self.lr = lr
        self.schedule = schedule
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.batch_size = batch_size
        self.epochs = epochs
        self.activation = act[activation]
        self.activationfinal = actfinal[activationfinal]
        self.step = step
        self.initializer = k_init[inicialization]
        self.drop = drop
        self.droprate = dropout_rate
        self.BN = True
        self.backbone = encoder[backbone]
        self.alpha = alpha
        self.finetuning = fine_tuning
        self.optimizer = optimizer
        self.gc = gc
        self.loss = losses_[loss]
        self.loss2 = losses_[loss2]
        self.interpolation = interpol[interpolation]
        self.decoderchoose = dchoose[decoderchoose]
        self.regularizer = regul[regularizer]
        self.path = path
        self.aug = aug
        self.metrics = [recall, precision, f1, 'accuracy', mean_iou, metrics.MeanIoU(num_classes=2), metrics.FalsePositives(), metrics.FalseNegatives()]
        self.visual = val_visual
        self.typefusion = typefusion[type_fusion]
        self.fus_str = fusion_strg[fus_str]
        self.ds = ds
        self.w = w
        self.w2 = w2
        self.sched_ch = lr_sched[sched_ch]
        self.lr_limit = lr_limit

    # def opt(self):
    #     if self.optimizer == 'sgd':
    #         if self.gc:
    #             opt = gc_opt.sgd(self.lr, momentum=0.9)
    #         else:
    #             opt = optimizers.SGD(self.lr, momentum=0.9)
    #
    #         return opt
    #     else:
    #         if self.gc:
    #             opt = gc_opt.adam(learning_rate=self.lr)
    #         else:
    #             opt = optimizers.Adam(learning_rate=self.lr)
    #
    #         return opt
