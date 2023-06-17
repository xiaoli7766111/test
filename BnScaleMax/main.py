from __future__ import print_function
import torch
import os
import argparse
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable
import models
import random
import numpy as np
from BnScaleMax.Main_tool import dataset, sparsity
import time

# Training settings
parser = argparse.ArgumentParser()

# train test batch_size
parser.add_argument('--batch-size', type=int, default=256, help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=256, help='input batch size for testing')

#  epoch num start and end
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--start-epoch', default=0, type=int)

# learning rate , momentum  , weight decay
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')

# cuda  seed
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# save and print
parser.add_argument('--log-interval', type=int, default=100, help='how many batches to display')
parser.add_argument('--save', default='logs_base/vgg', type=str, help='path to save prune model')

# model layers
parser.add_argument('--arch', default='Vgg', type=str, help='(Vgg LeNet Resnet)')
parser.add_argument('--depth', default=16, type=int, help='depth of network Vgg(19), Resnet(18/50/101)')
parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset ( cifar10 / mnist)')

# the threshold and the pruned threshold
parser.add_argument('--thre', default=0.0004, type=int, help='the threshold of network')
parser.add_argument('--prune_thre', default=0.0001, type=int, help='the threshold of weight')
parser.add_argument('--bn_thre', default=0.0002, type=int, help='the threshold of weight')

args = parser.parse_args(args=[])
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.seed:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


if not os.path.exists(args.save):
    os.makedirs(args.save)

train_loader, test_loader = dataset.data_set(args.cuda, args.dataset, args.batch_size, args.test_batch_size)

model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

model.cuda() if args.cuda else {}
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)


def train(epoch_):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = f.cross_entropy(output, target)

        _, pred = output.data.max(1, keepdim=True)

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch_ + 1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.1f}%)]\tLoss: {loss.item():.6f}')


def the_test():
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            test_loss += f.cross_entropy(output, target, size_average=False).item()
            _, pred = output.data.max(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            total = len(test_loader.dataset)
            pred1 = output.argmax(dim=1)
            correct2 += torch.eq(pred1, target).sum().float().item()

            maxk = max((1, 5))
            y_resize = target.view(-1, 1)
            _, pred = output.topk(maxk, 1, True, True)
            correct1 += torch.eq(pred, y_resize).sum().float().item()
    # ==========
    test_loss /= len(test_loader.dataset)
    info_r = f'\nTest set: Average loss: {test_loss:.4f},\n' \
             f' Top1:Accuracy: {correct}/{len(test_loader.dataset)}({100. * correct / total:.1f}%)' \
             f'   Top5:Accuracy: {correct1 / total}({100. * correct1 / total:.1f}%)\n'

    return correct / float(len(test_loader.dataset)), info_r


def main():
    best_prec_num = 0.

    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        t_star = time.time()
        train(epoch)

        wig, fil = sparsity.train_sparsity(model, args.prune_thre)  # 稀疏性
        t_end = time.time()
        print(f"each epoch use time: {t_end - t_star:.3f} s "
              f"\nweight sparsity {wig:.3f}%", f"\t filter sparsity {fil:.3f}%")
        prec, info = the_test()
        print(info)
        savepath = os.path.join(args.save, f"{args.arch}_{args.dataset}_{args.epochs}info.txt")
        with open(savepath, "a") as fp:
            fp.write(str(info) + "\n")

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
