import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from BnScaleMax.models.Vgg import *
from BnScaleMax.Main_tool import dataset, sparsity

# Prune settings
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset ( data.cifar10 / mnist)')

parser.add_argument('--test-batch-size', type=int, default=256, help='input batch size for testing')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--depth', default=16, type=int, help='depth of network Vgg(19), Resnet(18/50/101)')

parser.add_argument('--model', default='LOG_10000/r16_A/m12/Vgg_cifar10_100.model', type=str,
                    help='path to the model (default: none)')

parser.add_argument('--save', default='LOG_10000/prune/r16_A/m12', type=str,
                    help='path to save pruned model')

args = parser.parse_args(args=[])
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = Vgg(dataset=args.dataset, depth=args.depth)
model.cuda() if args.cuda else {}

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        # print(checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> loaded checkpoint '{args.model}' (epoch {checkpoint['epoch']}) Prec1: {best_prec:f}")
    else:
        print("=> no checkpoint found at '{}'".format(args.model))


defaultcfg = {
    'v16': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    'v19': [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
}

for name, param in model.named_parameters():
    if 'conv' in name:
        param.data = sparsity.zero_out(param, 0.0001).cuda()

filter_tap = sparsity.filter_gl(model)

cfg = defaultcfg['v16']
index = 0

cfg_up = []
cfg_mask = []
for t in range(len(cfg)):
    rank_part = filter_tap[index: (index + cfg[t])]
    index += cfg[t]
    ak2 = (rank_part != 0).sum()
    cfg_mask.append(rank_part.clone())
    cfg_up.append(ak2.item())

print(f'Pre-processing Successful!')


def the_test(model):

    _, test_loader = dataset.data_set(args.cuda, args.dataset, 64, args.test_batch_size)
    model.eval()
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print(f'\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} '
          f'{100. * correct / len(test_loader.dataset):.1f}%\n')

    return correct / float(len(test_loader.dataset))


if __name__ == '__main__':

    acc = the_test(model)
    print("Cfg:", cfg_up)
    newmodel = Vgg(dataset=args.dataset, cfg=cfg_up, depth=args.depth)
    if args.cuda:
        newmodel.cuda()

    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n" + str(cfg_up) + "\n")
        fp.write("Test accuracy: \n" + str(acc))

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]

    layer_id = 0
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().detach().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().detach().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().detach().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            if layer_id == 0:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().detach().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                m1.weight.data = m0.weight.data[:, idx0].clone()
                m1.bias.data = m0.bias.data.clone()
            else:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
            layer_id += 1

    torch.save({'cfg': cfg_up, 'state_dict': newmodel.state_dict()},
               os.path.join(args.save, 'pruned.pth'))

    model = newmodel
    p_acc = the_test(model)
    with open(savepath, "a") as fp:
        fp.write("\nPruned Test Accuracy: \n" + str(p_acc) + "\n")

