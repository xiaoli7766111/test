import math
import torch.nn as nn

# 调用时使用 限制只能用 类，函数，属性vgg
__all__ = ['Vgg']

default_cfg = {
    16: [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    # 16: [6, 60, 50, 66, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    19: [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
}


class Block(nn.Module):
    def __init__(self, channels, stride):
        super(Block, self).__init__()
        # 卷积， stride=2, 图像大小减半， 通道加倍
        self.conv1 = nn.Conv2d(
            channels[0], channels[1], kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        return output


class Vgg(nn.Module):
    def __init__(self, depth, dataset, init_weights=True, cfg=None):
        super(Vgg, self).__init__()
        if cfg is None:
            cfg = default_cfg[depth]
        # block lam001
        self.conv1 = nn.Conv2d(
            3, cfg[0], kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            cfg[0], cfg[1], kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.relu = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 2
        self.layer2 = nn.Sequential(Block(cfg[1:3], 1),
                                    Block(cfg[2:4], 1))
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        if depth == 16:
            # block 3
            self.layer3 = nn.Sequential(Block(cfg[3:5], 1),
                                        Block(cfg[4:6], 1),
                                        Block(cfg[5:7], 1))
            self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

            # block 4
            self.layer4 = nn.Sequential(Block(cfg[6:8], 1),
                                        Block(cfg[7:9], 1),
                                        Block(cfg[8:10], 1))
            self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

            # block 5
            self.layer5 = nn.Sequential(Block(cfg[9:11], 1),
                                        Block(cfg[10:12], 1),
                                        Block(cfg[11:13], 1))
        elif depth == 19:
            # block 3
            self.layer3 = nn.Sequential(Block(cfg[3:5], 1),
                                        Block(cfg[4:6], 1),
                                        Block(cfg[5:7], 1),
                                        Block(cfg[6:8], 1))
            self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

            # block 4
            self.layer4 = nn.Sequential(Block(cfg[7:9], 1),
                                        Block(cfg[8:10], 1),
                                        Block(cfg[9:11], 1),
                                        Block(cfg[10:12], 1))
            self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

            # block 5
            self.layer5 = nn.Sequential(Block(cfg[11:13], 1),
                                        Block(cfg[12:14], 1),
                                        Block(cfg[13:15], 1),
                                        Block(cfg[14:16], 1))
        if dataset == 'cifar10':
            self.num_classes = 10
        elif dataset == 'cifar100':
            self.num_classes = 100
        elif dataset == 'fashion mnist':
            self.num_classes = 10
        self.classifier = nn.Sequential(

            nn.Linear(cfg[-1], 512),  # fc1
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),  # fc2
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, self.num_classes),  # fc3，最终cifar10的输出是10类
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x_):
        x_ = self.conv1(x_)
        x_ = self.relu(self.bn1(x_))
        x_ = self.conv2(x_)
        x_ = self.relu(self.bn2(x_))
        x_ = self.max1(x_)
        x_ = self.layer2(x_)
        x_ = self.max2(x_)
        x_ = self.layer3(x_)
        x_ = self.max3(x_)
        x_ = self.layer4(x_)
        x_ = self.max4(x_)
        x_ = self.layer5(x_)
        x_ = nn.AvgPool2d(2)(x_)
        x_ = x_.view(x_.size(0), -1)
        y_ = self.classifier(x_)
        return y_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # 对卷积层权重进行标准化 normal_(mean= 0, std= )    0.0002    0.0294    0.0416   0.0589
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# a = Vgg(depth=16, dataset='data.cifar10')
# print(a)
# for name, param in a.named_parameters():
#     if 'conv' in name:
#         print(param)
# import torch
# b = torch.rand([1,3,32,32])
# c = a(b)
# print(c)
