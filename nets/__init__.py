from nets.SSeg.EFPENet import EFPENet
from nets.SSeg.FPENet import FPENet
from nets.SSeg.unet import Unet
from nets.SSeg.deeplabv3_plus import DeepLab
from nets.SSeg.segnet import SegNet
from nets.SSeg.pspnet import PSPNet
from nets.SSeg.FCN8s import FCN8
from nets.SSeg.UNetplusplus import UnetPlusPlus
from nets.SSeg.swinunet import SwinTransformerSys
from nets.SSeg.improved_segnet import improved_SegNet

get_model_from_name = {
    "FPENet": FPENet,
    "EFPENet": EFPENet,
    "UNet": Unet,
    "DeepLabV3+": DeepLab,
    'SegNet': SegNet,
    'PSPNet': PSPNet,
    'FCN8': FCN8,
    'UnetPlusPlus': UnetPlusPlus,
    'swinUNet': SwinTransformerSys,
    'improved-segnet':improved_SegNet
}
