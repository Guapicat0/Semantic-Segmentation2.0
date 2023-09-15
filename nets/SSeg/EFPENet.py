import torch
import torch.nn as nn
import torch.nn.functional as F

import math


__all__ = ["EFPENet"]



def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, groups=groups,bias=bias)

#这里所以卷积核大小为1的卷积，步长皆为1
def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

##SE注意力模块(通道注意力机制CA）
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class FPEBlock(nn.Module):
    # 总的定义，由于在conv3x3之后（通道数由3变成16）并且conv3x3后面直接使用FPEBlock，
    # layer1，inplanes=16；layer2，inplanes=64
    # outplanes（在layer1为16，layer2为32，layer3为64)
    def __init__(self, inplanes, outplanes,  downsample=None, stride=1, t=1, scales=4, se=False, norm_layer=None):
        super(FPEBlock, self).__init__()
        # scales为残差块中使用分层的特征组数（多尺度），t表示其中3*3卷积层数量，SE模块和BN层
        if inplanes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d #给norm_layer这个定义使用批归一化
        bottleneck_planes = inplanes * t
        #这里Layer1bottleneck_planes=inplans=16 layaer2==64
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride) #这里layer1，layer2，layer3卷积为conv1x1（16，16，1）
        self.bn1 = norm_layer(bottleneck_planes) #输入批归一化的通道数为16
        #......................................................................................................#
        # 3*3的卷积层，一共有3个卷积层和3个BN层......
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales,
                        #layer1，layer2，layer3为[4，4，groups=4，dia=1，padding=1]，这里共有四次卷积，对应图里四次3x3卷积核
                                            groups=(bottleneck_planes // scales),
                                            padding=1) for i in range(scales)]) #填充和padding抵消掉了，导致map大小没变
        ##Python中的//是向下取整的意思 a//b，应该是对除以b的结果向负无穷方向取整后的数 5//2=2
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales)]) #从头到尾
        #......................................................................................................#

        self.conv3 = conv1x1(bottleneck_planes, outplanes) #逐点卷积
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = eca_block(outplanes)
        self.downsample = downsample
        self.stride = stride
        self.scales = scales
        #summary FPEBLOCK的出现，代表该layer1定义结束，只有stage变幻时，map大小菜变

    def forward(self, x):
        identity = x
        # 1x1的卷积层，三部曲
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # scales个(3x3)的残差分层架构
        xs = torch.chunk(out, self.scales, 1)
        ys = [] #没看懂
        for s in range(self.scales):
            if s == 0:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity #没看懂
        out = self.relu(out)

        return out



class MEUModule(nn.Module):
    def __init__(self, channels_high, channels_low, channel_out):
        super(MEUModule, self).__init__()

        self.conv1x1_low = nn.Conv2d(channels_low, channel_out, kernel_size=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channel_out)
        self.sa_conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)

        self.conv1x1_high = nn.Conv2d(channels_high, channel_out, kernel_size=1, bias=False)
        self.bn_high = nn.BatchNorm2d(channel_out)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_conv = nn.Conv2d(channel_out, channel_out, kernel_size=1, bias=False)

        self.sa_sigmoid = nn.Sigmoid()
        self.ca_sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low): #传入进去的低特征图的tensor和高特征图的tensor，高语义特征图在网络深处
        """
        :param fms_high:  High level Feature map. Tensor.
        :param fms_low: Low level Feature map. Tensor.
        """
        _, _, h, w = fms_low.shape

        #
        fms_low = self.conv1x1_low(fms_low)
        fms_low= self.bn_low(fms_low)
        sa_avg_out = self.sa_sigmoid(self.sa_conv(torch.mean(fms_low, dim=1, keepdim=True)))
        #低语义实现spatial attention得到sa_avg_out

        #
        fms_high = self.conv1x1_high(fms_high)
        fms_high = self.bn_high(fms_high)
        ca_avg_out = self.ca_sigmoid(self.relu(self.ca_conv(self.avg_pool(fms_high))))

        #
        fms_high_up = F.interpolate(fms_high, size=(h,w), mode='bilinear', align_corners=True)
        fms_sa_att = sa_avg_out * fms_high_up
        #
        fms_ca_att = ca_avg_out * fms_low

        out = fms_ca_att + fms_sa_att

        return out


class EFPENet(nn.Module):
    def __init__(self, classes=4, zero_init_residual=False, width=16, scales=4, se=False, norm_layer=None):
        super(EFPENet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        outplanes = [int(width * 2 ** i) for i in range(3)] # **i平方，outplanes=[16,32,64] i=[0,1,2]
        # outplanes=[16x2^0，16x2^1，16x2^2]
        self.block_num = [1,3,9]  #stage 1 ，stage2和stage3使用的块多少
        self.inplanes = outplanes[0] #inplanes=16
        self.conv1 = nn.Conv2d(3, outplanes[0], kernel_size=3, stride=2, padding=1,bias=False)
        #  最开始的卷积，输入通道3.输出通道16，卷积核大小为3
        self.bn1 = norm_layer(outplanes[0]) #批归一处理的输入是卷积后的16
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(FPEBlock, outplanes[0], self.block_num[0],
                                       stride=1, t=1, scales=scales, se=se, norm_layer=norm_layer)
        self.layer2 = self._make_layer(FPEBlock, outplanes[1], self.block_num[1],
                                       stride=2, t=4, scales=scales, se=se, norm_layer=norm_layer)
        self.layer3 = self._make_layer(FPEBlock, outplanes[2], self.block_num[2],
                                       stride=2, t=4, scales=scales, se=se, norm_layer=norm_layer)
        self.meu1 = MEUModule(64,32,64)
        self.meu2 = MEUModule(64,16,32)

        # Projection layer （利用1x1卷积，按feature map最终转化为classes个特征图）
        self.project_layer = nn.Conv2d(32, classes, kernel_size = 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, FPEBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, t=1, scales=4, se=False, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(   # 一些列的步骤的使用工具
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes,  downsample=downsample, stride=stride, t=t, scales=scales, se=se,
                            norm_layer=norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,  scales=scales, se=se, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        ## stage 1
        x = self.conv1(x)   ##3x3 CONV
        x = self.bn1(x)
        x = self.relu(x)
        x_1 = self.layer1(x)  ##FPEBlock

        ## stage 2
        x_2_0 = self.layer2[0](x_1)     ## stage：FPEBLOCK==3
        x_2_1 = self.layer2[1](x_2_0)
        x_2_2 = self.layer2[2](x_2_1)
        x_2 = x_2_0 + x_2_2

        ## stage 3
        x_3_0 = self.layer3[0](x_2)    ##stage：FPEBLOCK==9
        x_3_1 = self.layer3[1](x_3_0)
        x_3_2 = self.layer3[2](x_3_1)
        x_3_3 = self.layer3[3](x_3_2)
        x_3_4 = self.layer3[4](x_3_3)
        x_3_5 = self.layer3[5](x_3_4)
        x_3_6 = self.layer3[6](x_3_5)
        x_3_7 = self.layer3[7](x_3_6)
        x_3_8 = self.layer3[8](x_3_7)
        x_3 = x_3_0 + x_3_8



        x2 = self.meu1(x_3, x_2)

        x1 = self.meu2(x2, x_1)

        output = self.project_layer(x1)

        # Bilinear interpolation x2
        output = F.interpolate(output,scale_factor=2, mode = 'bilinear', align_corners=True)

        return output



