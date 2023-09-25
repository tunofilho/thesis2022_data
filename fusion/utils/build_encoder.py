import fusion.models_early_input as models_early_input
import fusion.models_fusenet_inputs as models_fusenet_inputs
from fusion.models_fusenet_inputs.fg_unet import depth_fg_resnet50,fg_resnet50
from fusion.models_fusenet_inputs.af_net import af_resnet50, depth_af_resnet50
from fusion.models_fusenet_inputs.triseg_net import vgg_segnet
from tensorflow.keras import Model

def build(hiper):

    if hiper.typefusion == 'early':
        if hiper.backbone == 'vgg16':
            depth = models_early_input.early_depth_branch_vgg(hiper)
            encoder_ = models_early_input.vgg_16_earlyfusion(depth, hiper)

        elif hiper.backbone == 'resnet50':
            depth = models_early_input.early_depth_branch_resnet(hiper)
            encoder_ = models_early_input.resnet50_earlyfusion(depth, hiper)
        else:
            raise ValueError('typefusion and/or backbone error')

    elif hiper.typefusion == 'earlyres':
        if hiper.backbone == 'resnet50':
            depth = models_early_input.early_depth_branch_resnet(hiper)
            encoder_ = models_early_input.resnet50_earlyfusion(depth, hiper)
        else:
            raise ValueError('typefusion and/or backbone error')

    elif hiper.typefusion == 'unet':
        if hiper.backbone == 'vgg16':
            depth = models_fusenet_inputs.depth_branch_fusenet(hiper)
            encoder_ = models_fusenet_inputs.rgb_branch_fusenet_vgg16(depth, hiper)

        else:
            raise ValueError('typefusion and/or backbone error')

    elif hiper.typefusion == 'resnet':
        if hiper.backbone == 'resnet50':
            depth = models_fusenet_inputs.depth_branch_resnet50(hiper)
            encoder_ = models_fusenet_inputs.rgb_branch_resnet50(depth, hiper)

    elif hiper.typefusion == 'fg_resnet50':
        if hiper.backbone == 'resnet50':
            depth = depth_fg_resnet50(hiper)
            encoder_ = fg_resnet50(depth, hiper)

    elif hiper.typefusion == 'tcd_resnet50':
        if hiper.backbone == 'resnet50':
            depth = models_fusenet_inputs.depth_branch_resnet50(hiper)
            encoder_ = models_fusenet_inputs.rgb_branch_resnet50(depth, hiper)

    elif hiper.typefusion == 'tcd_resnet50_v2':
        if hiper.backbone == 'resnet50':
            rgb = models_fusenet_inputs.rgb_branch_tcd_resnet50(hiper)
            encoder_ = models_fusenet_inputs.depth_branch_tcd_resnet50(rgb, hiper)

    elif hiper.typefusion == 'af_resnet50':
        if hiper.backbone == 'resnet50':
            depth = depth_af_resnet50(hiper)
            color = af_resnet50(hiper)
            encoder_ = Model(inputs=[color.inputs, depth.inputs], outputs=[color.outputs, depth.outputs])

    elif hiper.typefusion == 'trisegnet_vgg16':
        if hiper.backbone == 'vgg16':
            encoder_ = vgg_segnet(1, hiper)   # all model build

    elif hiper.typefusion == 'fusenet_vgg16':
        if hiper.backbone == 'vgg16':
            depth = models_fusenet_inputs.depth_branch_fusenet(hiper)
            encoder_ = models_fusenet_inputs.rgb_branch_fusenet_vgg16(depth, hiper)

    else:
        raise ValueError('typefusion and/or backbone error')

    return encoder_