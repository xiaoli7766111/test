from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable
import models
import random
import time
from Reg_tool import reg_max, SSL
from BnScaleMax.Main_tool import dataset, sparsity


# Training settings
parser = argparse.ArgumentParser()

# 数据集地址
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
# batch size
parser.add_argument('--batch-size', type=int, default=256, help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=256, help='input batch size for testing')

# epoch
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')

# learning rate , momentum  , weight decay
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')

# cuda  seed
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='LOG_10000/r19/m12', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')

parser.add_argument('--arch', default='Vgg', type=str, help='architecture to use')
parser.add_argument('--depth', default=19, type=int, help='depth of the neural network')

# 重训练的模型
parser.add_argument('--refine', default='logs_baseline/v19/Vgg_cifar10_200.model', type=str, metavar='PATH',
                    help='path to the pruned model to be inc4 tuned')

# the threshold and the pruned threshold
parser.add_argument('--thre', default=0.004, type=int, help='the threshold of network')
parser.add_argument('--prune_thre', default=0.0001, type=int, help='the threshold of weight')
parser.add_argument('--bn_thre', default=0.0002, type=int, help='the threshold of weight')

# 解析参数
args = parser.parse_args(args=[])
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 设置随机种子
if args.seed:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


if not os.path.exists(args.save):
    os.makedirs(args.save)

train_loader, test_loader = dataset.data_set(args.cuda, args.dataset, args.batch_size, args.test_batch_size)

if args.refine:
    checkpoint = torch.load(args.refine)
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

    model.load_state_dict(checkpoint['state_dict'])
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)


def update_bn():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.bn_thre * torch.sign(m.weight.data))  # L1

def train(epoch_):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # ==================================      max     =========================================
        loss_increase = reg_max.bn_change_loss(model, args.thre)
        # ==================================      ssl     =========================================
        # loss_increase = SSL.ssl_loss(model, args.thre)

        loss = f.cross_entropy(output, target) + loss_increase
        # loss = f.cross_entropy(output, target)

        _, pred = output.data.max(1, keepdim=True)

        loss.backward()
        update_bn()
        optimizer.step()

        if batch_idx % args.log_interval == 0:

            print(f'Train Epoch: {epoch_ + 1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.1f}%)]\tLoss: {loss.item():.6f}')


def the_test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            test_loss += f.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    info = f'\nTest set: Average loss: {test_loss:.4f},' \
           f' Accuracy: {correct}/{len(test_loader.dataset)}' \
           f'({100. * correct / len(test_loader.dataset):.1f}%)\n'

    return correct / float(len(test_loader.dataset)), info


def main():
    best_prec_num = 0.

    # 对learning_rate 进行处理
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        t_star = time.time()
        train(epoch)
        wig, fil = sparsity.train_sparsity(model, args.prune_thre)  # 稀疏性
        f_ = sparsity.conv_sparsity(model, args.prune_thre)
        print(f_)
        t_end = time.time()
        print(f"each epoch use time: {t_end - t_star:.3f} s "
              f"\nweight sparsity {wig:.3f}%", f"\t filter sparsity {fil:.3f}%")

        prec, info = the_test()
        print(info)
        savepath = os.path.join(args.save, f"{args.arch}_{args.dataset}_{args.epochs}info.txt")
        with open(savepath, "a") as fp:
            fp.write(str(info) + "\n")
            fp.write(str(wig) + "\t")
            fp.write(str(fil) + "\n")
        is_best = prec >= best_prec_num

        if is_best:
            best_prec_num = prec
            file = f"{args.save}/{args.arch}_{args.dataset}_{args.epochs}.model"
            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec': best_prec_num,
                        'optimizer': optimizer.state_dict()}, file)

        print("Best accuracy: " + str(best_prec_num))


if __name__ == "__main__":
    main()
