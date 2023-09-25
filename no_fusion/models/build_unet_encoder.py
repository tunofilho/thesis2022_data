from .backbone_vgg import vgg, vgg_BN
from .backbone_resnet import resnet
from ._deep_labv3plus import DeeplabV3Plus
from ._segnet import vgg_segnet
from ._dlinknet import dlink
from ._fg_net import fg_resnet50, decoder_fg_resnet50_build_decoder
from ._fg_net_v2 import decoder_fg_resnet50_build_decoder_v2


def encoder_model(hiper):
    if hiper.backbone == ('resnet34'):
        encoder = resnet(hiper)

    elif hiper.backbone == ('resnet50'):
        encoder = resnet(hiper)

    elif hiper.backbone == ('resnet101'):
        encoder = resnet(hiper)

    elif hiper.backbone == ('resnet152'):
        encoder = resnet(hiper)

    elif hiper.backbone == ('vgg16'):
        encoder = vgg(hiper)

    elif hiper.backbone == ('vgg19'):
        encoder = vgg(hiper)

    elif hiper.backbone == ('scratch16'):
        encoder = vgg_BN(hiper)

    elif hiper.backbone == ('scratch19'):
        encoder = vgg_BN(hiper)

    else:
        raise ValueError('backbone name error')

    return encoder

def sota(hiper):
    if hiper.architecture == 'deeplabv3plus':
        return DeeplabV3Plus(hiper.IMG_WIDTH, 1, hiper)
    if hiper.architecture == 'segnet':
        return vgg_segnet(1, hiper)
    if hiper.architecture == 'dlink':
        return dlink(hiper)
    if hiper.architecture == 'focus_gated':
        enc = fg_resnet50(hiper)
        return decoder_fg_resnet50_build_decoder(enc, hiper)
    if hiper.architecture == 'focus_gated_v2':
        enc = fg_resnet50(hiper)
        return decoder_fg_resnet50_build_decoder_v2(enc, hiper)
    # if hiper.architecture == ''