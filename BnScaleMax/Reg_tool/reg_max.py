import torch


# 将 weight 存储 比每次循环节省时间
def get_conv_weight(model):
    conv_weight_list = []

    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name and 'layer' in name and 'layer5.3' not in name or 'conv2' in name:
        # if 'conv' in name and 'weight' in name and 'layer' in name and 'layer5.2' not in name or 'conv2' in name:
        # if 'conv' in name and 'layer' in name and 'weight' in name:    # resnet

            weight = (name, param)
            conv_weight_list.append(weight)

    return conv_weight_list


# 存储 bn层 weight
def get_bn_weight(model):
    bn_weight_list = []

    for name, param in model.named_parameters():
        # vgg 19
        if 'bn' in name and 'weight' in name and 'layer' in name and 'layer5.3' not in name or 'bn2.weight' in name:
        # vgg 16
        # if 'bn' in name and 'weight' in name and 'layer' in name and 'layer5.2' not in name or 'bn2.weight' in name:
        # resnet
        # if 'bn' in name and 'weight' in name and 'layer' in name:  # resnet
            weight = (name, param)
            bn_weight_list.append(weight)

    return bn_weight_list


# 计算filter的group lasso
def filter_gl(model):
    conv_weight_list = get_conv_weight(model)  # 得到模型参数
    filter_gl_list = torch.tensor([]).cuda()  # torch 列表操作

    for name, wight in conv_weight_list:
        ith_filter_reg_loss = torch.sqrt(torch.sum(torch.pow(wight, 2), dim=[1, 2, 3]))  # 进行GL平方和开根号
        # torch 添加list
        filter_gl_list = torch.cat([filter_gl_list, ith_filter_reg_loss], dim=0)

    return filter_gl_list  # 64


# calculate bn param
def bn_each_weight(model, thr):

    bn_weight_list = get_bn_weight(model)  # 读入bn信息
    weight_list2 = torch.tensor([]).cuda()

    for t in range(len(bn_weight_list)):   # 13
        rank_part = bn_weight_list[t]  # 截取的L1
        max_num = torch.max(torch.abs(rank_part[1]))    # 选择最大的值作为分母，肯定每次有一个不惩罚
        weight = torch.mul(torch.sub(1, torch.div(torch.abs(rank_part[1]), max_num)), thr)
        weight_list2 = torch.cat([weight_list2, weight], dim=0)

    return weight_list2


# 变动系数
def bn_change_loss(model, thr):

    num = bn_each_weight(model, thr)
    gl = filter_gl(model)                 # 获取 filter 的 gl
    loss_tol = sum(torch.mul(num, gl))    # 进行 对应位置相乘

    return loss_tol
