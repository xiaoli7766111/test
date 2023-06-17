from __future__ import absolute_import
import torch
import torch.nn as nn
import math

__all__ = ['Resnet101']

defaultcfg = {
    101: [64, 64, 64, 256, 64, 64, 256, 64, 64, 256,
          128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512,
          256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
          256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
          256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
          256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
          512, 512, 2048, 512, 512, 2048, 512, 512, 2048],
            }


class ResNetBasicBlock(nn.Module):
    def __init__(self, channels, stride):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels[0], channels[1], kernel_size=(1, 1), stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels[1], channels[2], kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            channels[2], channels[3], kernel_size=(1, 1), stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[3])

    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.relu(self.bn2(output))
        output = self.conv3(output)
        output = self.bn3(output)
        output += residual
        return torch.relu(output)


class BaseRestBlockDownsample(nn.Module):
    def __init__(self, channels, stride):
        super(BaseRestBlockDownsample, self).__init__()
        self.conv1 = nn.Conv2d(
            channels[0], channels[1], kernel_size=(1, 1), stride=stride[1], bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels[1], channels[2], kernel_size=(3, 3), stride=stride[0], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            channels[2], channels[3], kernel_size=(1, 1), stride=stride[1], bias=False)
        self.bn3 = nn.BatchNorm2d(channels[3])

        self.downsample = nn.Sequential(
            nn.Conv2d(
                channels[0], channels[3], kernel_size=(1, 1), stride=stride[0], padding=0, bias=False))

    def forward(self, x):
        residual = x
        residual = self.downsample(residual)
        output = self.conv1(x)
        output = self.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.relu(self.bn2(output))
        output = self.conv3(output)
        output = self.bn3(output)
        output += residual
        return torch.relu(output)


class Resnet101(nn.Module):
    def __init__(self, depth=None, dataset='data.cifar10', cfg=None):
        super(Resnet101, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=cfg[0], kernel_size=(3, 3),
                               stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)

        # group 2
        if depth == 101:
            self.layer1 = nn.Sequential(BaseRestBlockDownsample(cfg[0:4], [1, 1]),
                                        ResNetBasicBlock(cfg[3:7], 1),
                                        ResNetBasicBlock(cfg[6:10], 1))

            self.layer2 = nn.Sequential(BaseRestBlockDownsample(cfg[9:13], [2, 1]),
                                        ResNetBasicBlock(cfg[12:16], 1),
                                        ResNetBasicBlock(cfg[15:19], 1),
                                        ResNetBasicBlock(cfg[18:22], 1))

            self.layer3 = nn.Sequential(BaseRestBlockDownsample(cfg[21:25], [2, 1]),
                                        ResNetBasicBlock(cfg[24:28], 1),
                                        ResNetBasicBlock(cfg[27:31], 1),
                                        ResNetBasicBlock(cfg[30:34], 1),
                                        ResNetBasicBlock(cfg[33:37], 1),
                                        ResNetBasicBlock(cfg[36:40], 1),
                                        ResNetBasicBlock(cfg[39:43], 1),
                                        ResNetBasicBlock(cfg[42:46], 1),
                                        ResNetBasicBlock(cfg[45:49], 1),
                                        ResNetBasicBlock(cfg[48:52], 1),
                                        ResNetBasicBlock(cfg[51:55], 1),
                                        ResNetBasicBlock(cfg[54:58], 1),
                                        ResNetBasicBlock(cfg[57:61], 1),
                                        ResNetBasicBlock(cfg[60:64], 1),
                                        ResNetBasicBlock(cfg[63:67], 1),
                                        ResNetBasicBlock(cfg[66:70], 1),
                                        ResNetBasicBlock(cfg[69:73], 1),
                                        ResNetBasicBlock(cfg[72:76], 1),
                                        ResNetBasicBlock(cfg[75:79], 1),
                                        ResNetBasicBlock(cfg[78:82], 1),
                                        ResNetBasicBlock(cfg[81:85], 1),
                                        ResNetBasicBlock(cfg[84:88], 1),
                                        ResNetBasicBlock(cfg[87:91], 1))

            self.layer4 = nn.Sequential(BaseRestBlockDownsample(cfg[90:94], [2, 1]),
                                        ResNetBasicBlock(cfg[93:97], 1),
                                        ResNetBasicBlock(cfg[96:100], 1))

        else:
            print("no have the depth!")

        # group 3
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if dataset == 'data.cifar10':
            self.a = 10
        elif dataset == 'cifar100':
            self.a = 100
        self.fc = nn.Linear(cfg[-1], self.a)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(self.bn1(output))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool(output)
        batch_size = output.shape[0]
        output = output.reshape(batch_size, -1)
        output = self.fc(output)
        return output
