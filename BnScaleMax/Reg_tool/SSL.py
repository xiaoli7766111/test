import torch
import numpy as np


def get_conv_weight(model):
    conv_weight_list = []

    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name and 'layer' in name and\
                'layer5.3' not in name or 'conv2' in name:
        # if 'conv' in name and 'weight' in name and 'layer' in name and 'layer5.2' not in name or 'conv2' in name:
        # if 'conv' in name and 'layer' in name and 'weight' in name:    # resnet

            weight = (name, param)
            conv_weight_list.append(weight)

    return conv_weight_list


# filter的group lasso
def filter_gl(model):
    conv_weight_list = get_conv_weight(model)
    filter_gl_list = torch.tensor([]).cuda()

    for name, wight in conv_weight_list:
        ith_filter_reg_loss = torch.sqrt(torch.sum(torch.pow(wight, 2), dim=[1, 2, 3]))
        filter_gl_list = torch.cat([filter_gl_list, ith_filter_reg_loss], dim=0)

    return filter_gl_list


def ssl_loss(model, thr):
    gl = filter_gl(model)
    loss_tol = sum(torch.mul(thr, gl))
    return loss_tol
