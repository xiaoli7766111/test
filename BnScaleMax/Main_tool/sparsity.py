import numpy as np
import torch


# =======================================训练后剪枝============================================
def zero_out(p, thre):
    p = p.cpu().detach().numpy()
    zero_out_idx = np.nonzero(abs(p) < thre)  # <thre则为True，返回非零(True)元素的索引
    p[zero_out_idx] = 0
    return torch.tensor(p)


# 训练中计算稀疏：小于阈值的权值数量
def train_sparsity(model, thre):
    # 整体参数的计算
    tol_filter_num = 0
    for name, param in model.named_parameters():
        if 'conv' in name:
            tol_filter_num += param.nelement()

    sparsity_zero = 0   # 网络总体的零值数量
    zero_filter_num, filter_num = 0, 0

    for name, param in model.state_dict().items():
        zero = 0  # 每个filter的零值数量
        if 'conv' in name:  # 仅计算卷积权重的稀疏
            filter_num += param.shape[0]
            for i in range(param.shape[0]):
                p = param[i, :, :, :].cpu().detach().numpy()
                zero += np.count_nonzero(abs(p) < thre)
                # abs<thre  即小于阈值的（要被剪枝）是true，np.count_nonzero就能统计到要被剪枝的权值数量
                sparsity_zero += zero
                if zero == p.size:  # filter整体 的小于权重
                    zero_filter_num += 1
                zero = 0
    # 整体的稀疏性   filter全为 0 的个数
    filter_sapr = 100 * (zero_filter_num / filter_num)
    weight_sapr = 100 * (sparsity_zero / tol_filter_num)
    return weight_sapr, filter_sapr


# 剪枝后计算稀疏：零值权值数量
def get_sparsity(model):
    zero = 0
    # 整体参数的计算
    tol_filter_num = 0
    for name, param in model.named_parameters():
        if 'conv' in name:
            tol_filter_num += param.nelement()
    for name, param in model.state_dict().items():
        if 'conv' in name:  # 仅计算卷积权重的稀疏
            p = param.cpu().detach().numpy()   # p.size filter的权重数量
            zero += np.count_nonzero(p == 0)
    return 100 * (zero / tol_filter_num)


# 训练中计算稀疏：小于阈值的权值数量
def conv_sparsity(model, thre):
    conv_list = []
    filter_num = 0

    for name, param in model.state_dict().items():
        zero = 0  # 每个filter的零值数量
        shape = 0
        sparsity_zero = 0  # 网络总体的零值数量
        if 'conv' in name:  # 仅计算卷积权重的稀疏
            zero_filter_num = 0
            filter_num += param.shape[0]
            for i in range(param.shape[0]):
                # print(param.shape[0])  # 是不是每层的filter数目========================================
                p = param[i, :, :, :].cpu().detach().numpy()
                zero += np.count_nonzero(abs(p) < thre)
                # abs<thre  即小于阈值的（要被剪枝）是true，np.count_nonzero就能统计到要被剪枝的权值数量
                sparsity_zero += zero
                if zero == p.size:  # filter整体 的小于权重
                    zero_filter_num += 1
                zero = 0
                shape = zero_filter_num/param.shape[0]
            conv_list.append(shape)

    return conv_list


def get_conv_weight(model):
    conv_weight_list = []

    for name, param in model.named_parameters():
        if 'conv' in name:
            weight = (name, param)
            conv_weight_list.append(weight)

    return conv_weight_list


# 计算filter的group lasso
def filter_gl(model):
    conv_weight_list = get_conv_weight(model)  # 得到模型参数
    filter_gl_list = torch.tensor([]).cuda()  # torch 列表操作

    for name, wight in conv_weight_list:
        ith_filter_reg_loss = torch.sqrt(torch.sum(torch.pow(wight, 2), dim=[1, 2, 3]))  # 进行GL平方和开根号
        # torch 添加list
        filter_gl_list = torch.cat([filter_gl_list, ith_filter_reg_loss], dim=0)

    return filter_gl_list  # 64