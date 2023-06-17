from __future__ import absolute_import
import torch
import torch.nn as nn
import math

__all__ = ['Resnet']

defaultcfg = {
    18: [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512],
    50: [64, 64, 64, 256, 64, 64, 256, 64, 64, 256,
         128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512,
         256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
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


class BaseRestBlock_Downsample(nn.Module):
    def __init__(self, channels, stride):
        super(BaseRestBlock_Downsample, self).__init__()
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
        residual = self.downsample(residual)
        output = self.conv1(x)
        output = self.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        output += residual
        return torch.relu(output)


class Resnet(nn.Module):
    def __init__(self, depth=None, dataset='cifar10', cfg=None):
        super(Resnet, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        # group 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=cfg[0], kernel_size=(3, 3),
                               stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)

        # group 2
        if depth == 50:
            self.layer1 = nn.Sequential(BaseRestBlock_Downsample(cfg[0:4], [1, 1]),
                                        BaseRestBlock_Downsample(cfg[3:7], [1, 1]),
                                        BaseRestBlock_Downsample(cfg[6:10], [1, 1]))

            self.layer2 = nn.Sequential(BaseRestBlock_Downsample(cfg[9:13], [2, 1]),
                                        BaseRestBlock_Downsample(cfg[12:16], [1, 1]),
                                        BaseRestBlock_Downsample(cfg[15:19], [1, 1]),
                                        BaseRestBlock_Downsample(cfg[18:22], [1, 1]))

            self.layer3 = nn.Sequential(BaseRestBlock_Downsample(cfg[21:25], [2, 1]),
                                        BaseRestBlock_Downsample(cfg[24:28], [1, 1]),
                                        BaseRestBlock_Downsample(cfg[27:31], [1, 1]),
                                        BaseRestBlock_Downsample(cfg[30:34], [1, 1]),
                                        BaseRestBlock_Downsample(cfg[33:37], [1, 1]),
                                        BaseRestBlock_Downsample(cfg[36:40], [1, 1]))

            self.layer4 = nn.Sequential(BaseRestBlock_Downsample(cfg[39:43], [2, 1]),
                                        BaseRestBlock_Downsample(cfg[42:46], [1, 1]),
                                        BaseRestBlock_Downsample(cfg[45:49], [1, 1]))

        elif depth == 18:
            self.layer1 = nn.Sequential(RestBlock_Downsample(cfg[0:3], [1, 1]),
                                        RestBlock_Downsample(cfg[2:5], [1, 1]))

            self.layer2 = nn.Sequential(RestBlock_Downsample(cfg[4:7], [2, 1]),
                                        RestBlock_Downsample(cfg[6:9], [1, 1]))

            self.layer3 = nn.Sequential(RestBlock_Downsample(cfg[8:11], [2, 1]),
                                        RestBlock_Downsample(cfg[10:13], [1, 1]))

            self.layer4 = nn.Sequential(RestBlock_Downsample(cfg[12:15], [2, 1]),
                                        RestBlock_Downsample(cfg[14:17], [1, 1]))

        else:
            print("no have the depth!")

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
