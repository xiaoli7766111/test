from __future__ import absolute_import
import torch
import torch.nn as nn
import math

__all__ = ['Resnet20']

defaultcfg = {
    20: [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64],
    44: [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
         32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
         64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]}


class ResNetBlock(nn.Module):
    def __init__(self, channels, stride):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels[0], channels[1], kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels[1], channels[2], kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[2])

    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        output += residual
        return torch.relu(output)


class RestBlock_Downsample(nn.Module):
    def __init__(self, channels, stride):
        super(RestBlock_Downsample, self).__init__()
        self.conv1 = nn.Conv2d(
            channels[0], channels[1], kernel_size=(3, 3), stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels[1], channels[2], kernel_size=(3, 3), stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[2])

        self.downsample = nn.Sequential(
            nn.Conv2d(
                channels[0], channels[2], kernel_size=(1, 1), stride=stride[0], padding=0, bias=False))

    def forward(self, x):
        residual = x
        residual = self.downsample(residual)  # 图像大小减半
        output = self.conv1(x)
        output = self.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        output += residual
        return torch.relu(output)


class Resnet20(nn.Module):
    def __init__(self, depth=None, dataset='cifar10', cfg=None):
        super(Resnet20, self).__init__()
        # 卷积 (W-F+2p)/stride[取下] + 05flex
        if cfg is None:
            cfg = defaultcfg[depth]

        # group 1
        self.conv = nn.Conv2d(in_channels=3, out_channels=cfg[0], kernel_size=(3, 3),
                               stride=(1, 1), padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)

        if depth == 20:
            self.layer1 = nn.Sequential(RestBlock_Downsample(cfg[0:3], [1, 1]),
                                        RestBlock_Downsample(cfg[2:5], [1, 1]),
                                        RestBlock_Downsample(cfg[4:7], [1, 1]))

            self.layer2 = nn.Sequential(RestBlock_Downsample(cfg[6:9], [2, 1]),
                                        RestBlock_Downsample(cfg[8:11], [1, 1]),
                                        RestBlock_Downsample(cfg[10:13], [1, 1]))

            self.layer3 = nn.Sequential(RestBlock_Downsample(cfg[12:15], [2, 1]),
                                        RestBlock_Downsample(cfg[14:17], [1, 1]),
                                        RestBlock_Downsample(cfg[16:19], [1, 1]))

        else:
            self.layer1 = nn.Sequential(RestBlock_Downsample(cfg[0:3], [1, 1]),
                                        RestBlock_Downsample(cfg[2:5], [1, 1]),
                                        RestBlock_Downsample(cfg[4:7], [1, 1]),
                                        RestBlock_Downsample(cfg[6:9], [1, 1]),
                                        RestBlock_Downsample(cfg[8:11], [1, 1]),
                                        RestBlock_Downsample(cfg[10:13], [1, 1]),
                                        RestBlock_Downsample(cfg[12:15], [1, 1]))

            self.layer2 = nn.Sequential(RestBlock_Downsample(cfg[14:17], [2, 1]),
                                        RestBlock_Downsample(cfg[16:19], [1, 1]),
                                        RestBlock_Downsample(cfg[18:21], [1, 1]),
                                        RestBlock_Downsample(cfg[20:23], [1, 1]),
                                        RestBlock_Downsample(cfg[22:25], [1, 1]),
                                        RestBlock_Downsample(cfg[24:27], [1, 1]),
                                        RestBlock_Downsample(cfg[26:29], [1, 1]))

            self.layer3 = nn.Sequential(RestBlock_Downsample(cfg[28:31], [2, 1]),
                                        RestBlock_Downsample(cfg[30:33], [1, 1]),
                                        RestBlock_Downsample(cfg[32:35], [1, 1]),
                                        RestBlock_Downsample(cfg[34:37], [1, 1]),
                                        RestBlock_Downsample(cfg[36:39], [1, 1]),
                                        RestBlock_Downsample(cfg[38:41], [1, 1]),
                                        RestBlock_Downsample(cfg[40:43], [1, 1]))

        # group 3
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if dataset == 'cifar10':
            self.num_classes = 10
        elif dataset == 'cifar100':
            self.num_classes = 100
        self.fc = nn.Linear(cfg[-1], self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def forward(self, x):
        output = self.conv(x)
        output = self.relu(self.bn(output))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.avgpool(output)
        batch_size = output.shape[0]
        output = output.reshape(batch_size, -1)
        output = self.fc(output)
        return output
