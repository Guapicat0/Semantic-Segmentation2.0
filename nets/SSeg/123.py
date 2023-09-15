import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from thop import clever_format, profile
from nets.conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution as DSConv
from nets.blocks.ASPP import ASPP

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        #定义空洞卷积
        self.DConv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=2,dilation=2)
        self.DConv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,padding=4,dilation=4)
        self.DConv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=8,dilation=8)
        self.DConv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=4,dilation=4)
        self.DConv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2,dilation=2)
        self.aspp = ASPP(in_channel=512)


        self.encode1 = nn.Sequential(
            DSConv(in_ch=in_channels, out_ch=64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DSConv(in_ch=64, out_ch=64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encode2 = nn.Sequential(
            DSConv(in_ch=64, out_ch=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            DSConv(in_ch=128, out_ch=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.encode3 = nn.Sequential(
            DSConv(in_ch=128, out_ch=256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            DSConv(in_ch=256, out_ch=256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            DSConv(in_ch=256, out_ch=256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.encode4 = nn.Sequential(
            DSConv(in_ch=256, out_ch=512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            DSConv(in_ch=512, out_ch=512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            DSConv(in_ch=512, out_ch=512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.encode5 = nn.Sequential(
            DSConv(in_ch=512, out_ch=512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            DSConv(in_ch=512, out_ch=512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            DSConv(in_ch=512, out_ch=512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):

        # (3, 512, 512) -> (64,512,512)
        feat1 = self.encode1(x)
        # (64, 512, 512) -> (64,512,512)
        x = self.DConv1(feat1)
        # (64,256,256) ->  (128,256,256)
        feat2 = self.encode2(x)
        # (128,256,256) ->  (128,128,128)
        x = self.DConv2(feat2)
        # (128,128,128) ->  (256,128,128)->(256,128,128)
        feat3 = self.encode3(x)
        # (256,128,128)-> (256,64,64)
        x = self.DConv3(feat3)
        # (256,64,64)->(512,64,64)
        feat4 = self.encode4(x)
        # (512,64,64)->(512,32,32)
        x = self.DConv4(feat4)
        # (512,32,32)->(512,32,32)
        x = self.encode5(x)
        # (512,32,32)->(512,16,16)
        x = self.DConv5(x)
        # (512, 16, 16)->(512,16,16)
        x = self.aspp(x)

        return x

class SegNet(nn.Module):
    def __init__(self, num_classes,pretrained = False):
        super(SegNet, self).__init__()

        self.encoder = Encoder(in_channels=3)

    def forward(self, x):
        x = self.encoder(x)
        return x

if __name__ == '__main__':
    inputs = torch.randn(1, 3, 512, 512)
    model = SegNet(num_classes=2)
    output = model(inputs)
    print("最终输出",output.shape)


    SSeg = "SegNet"
    classes = 2
    input_shape = [512, 512]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    summary(model, (3, input_shape[0], input_shape[1]))
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input,), verbose=False)
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))

