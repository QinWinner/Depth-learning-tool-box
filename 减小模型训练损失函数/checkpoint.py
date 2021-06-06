#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   checkpoint.py
@Time    :   2021/06/01 10:44:33
@Author  :   QinJian 
@Desc    :   None
'''
'''
使用https://blog.csdn.net/ONE_SIX_MIX/article/details/93937091 
https://zhuanlan.zhihu.com/p/134008239?utm_source=wechat_session&utm_medium=social&utm_oi=1133702791620067328
中介绍的chechpoint 来使用时间换取显存，在bert-L 的训练过程中，在batch_size上提升了5倍，速度只降低了约1/5，且精度没有损失。

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.datasets.cifar import CIFAR10
import numpy as np
from progressbar import progressbar


#方法一 https://realmyang.com/squeeze-most-out-of-your-gpu-checkpoint-in-pytorch/
'''
有模型为：
class Dummy(nn.Module):
    def __init__(self, ...):
        ...
    def forward(self, ...):
        ...
dummy=Dummy(...)
平时我们调用模型 ... = dummy(...)

When the input to Checkpoint requires a gradient, the output of Checkpoint requires a gradient and vice versa. So you shouldn’t use Checkpoint on the first sub-module of a network since the input doesn’t require gradient and it will make this sub-module unable to be updated during backpropagation. Although you can manually set “requires_grad” to True of the initial Tensor to solve this problem, by principle, it is not correct.
The batch normalization layers save two variables called “running_mean” and “running_var” to track the mean and variance of your data. Because Checkpointed modules will have its forward() method called twice, you should change the original momentum of batch normalization layers to its squared root to get a correct mean and variance estimation.
To avoid the “requires_grad” warning that is raised by Checkpointed modules during validation, you can make the module checkpointed only when the module is in the training mode. For example:


def memory_efficient_forward(*args, module: nn.Module, training: bool):
    if training:
        def _forward(*_args):
            return module(*_args)
        return cp.checkpoint(_forward, *args)
    else:
        return module(*args)
        
现在调用 memory_efficient_forward(..., dummy,True)






'''










########方法二示意
def conv_bn_relu(in_ch, out_ch, ker_sz, stride, pad):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, bias=False),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU())


class NetA(nn.Module):
    def __init__(self, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        k = 2
        # 32x32
        self.layer1 = conv_bn_relu(3, 32*k, 3, 1, 1)
        self.layer2 = conv_bn_relu(32*k, 32*k, 3, 2, 1)
        # 16x16
        self.layer3 = conv_bn_relu(32*k, 64*k, 3, 1, 1)
        self.layer4 = conv_bn_relu(64*k, 64*k, 3, 2, 1)
        # 8x8
        self.layer5 = conv_bn_relu(64*k, 128*k, 3, 1, 1)
        self.layer6 = conv_bn_relu(128*k, 128*k, 3, 2, 1)
        # 4x4
        self.layer7 = conv_bn_relu(128*k, 256*k, 3, 1, 1)
        self.layer8 = conv_bn_relu(256*k, 256*k, 3, 2, 1)
        # 1x1
        self.layer9 = nn.Linear(256*k, 10)

    def seg0(self, y):
        y = self.layer1(y)
        return y

    def seg1(self, y):
        y = self.layer2(y)
        y = self.layer3(y)
        return y

    def seg2(self, y):
        y = self.layer4(y)
        y = self.layer5(y)
        return y

    def seg3(self, y):
        y = self.layer6(y)
        y = self.layer7(y)
        return y

    def seg4(self, y):
        y = self.layer8(y)
        y = F.adaptive_avg_pool2d(y, 1)
        y = torch.flatten(y, 1)
        y = self.layer9(y)
        return y

    def forward(self, x):
        y = x
        # y = self.layer1(y)
        y = y + torch.zeros(1, dtype=y.dtype, device=y.device, requires_grad=True)
        if self.use_checkpoint:
            # 使用 checkpoint
            y = checkpoint(self.seg0, y)
            y = checkpoint(self.seg1, y)
            y = checkpoint(self.seg2, y)
            y = checkpoint(self.seg3, y)
            y = checkpoint(self.seg4, y)
        else:
            # 不使用 checkpoint
            y = self.seg0(y)
            y = self.seg1(y)
            y = self.seg2(y)
            y = self.seg3(y)
            y = self.seg4(y)

        return y


if __name__ == '__main__':
    net = NetA(use_checkpoint=True).cuda()

    train_dataset = CIFAR10('../datasets/cifar10', True, download=True)
    train_x = np.asarray(train_dataset.data, np.uint8)
    train_y = np.asarray(train_dataset.targets, np.int)

    losser = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), 1e-3)

    epoch = 10
    batch_size = 31
    batch_count = int(np.ceil(len(train_x) / batch_size))

    for e_id in range(epoch):
        print('epoch', e_id)

        print('training')
        net.train()
        loss_sum = 0
        for b_id in progressbar(range(batch_count)):
            optim.zero_grad()

            batch_x = train_x[batch_size*b_id: batch_size*(b_id+1)]
            batch_y = train_y[batch_size*b_id: batch_size*(b_id+1)]

            batch_x =  torch.from_numpy(batch_x).permute(0, 3, 1, 2).float() / 255.
            batch_y =  torch.from_numpy(batch_y).long()

            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            batch_x = F.interpolate(batch_x, (224, 224), mode='bilinear')

            y = net(batch_x)
            loss = losser(y, batch_y)
            loss.backward()
            optim.step()
            loss_sum += loss.item()
        print('loss', loss_sum / batch_count)

        with torch.no_grad():
            print('testing')
            net.eval()
            acc_sum = 0
            for b_id in progressbar(range(batch_count)):
                optim.zero_grad()

                batch_x = train_x[batch_size * b_id: batch_size * (b_id + 1)]
                batch_y = train_y[batch_size * b_id: batch_size * (b_id + 1)]

                batch_x = torch.from_numpy(batch_x).permute(0, 3, 1, 2).float() / 255.
                batch_y = torch.from_numpy(batch_y).long()

                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

                batch_x = F.interpolate(batch_x, (224, 224), mode='bilinear')

                y = net(batch_x)

                y = torch.topk(y, 1, dim=1).indices
                y = y[:, 0]

                acc = (y == batch_y).float().sum() / len(batch_x)

                acc_sum += acc.item()
            print('acc', acc_sum / batch_count)

        ids = np.arange(len(train_x))
        np.random.shuffle(ids)
        train_x = train_x[ids]
        train_y = train_y[ids]