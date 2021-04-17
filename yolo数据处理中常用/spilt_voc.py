#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :set.py
@说明        :voc数据集生产训练集和测试机的列表
@时间        :2020/05/07 21:41:47
@作者        :
@版本        :1.0
'''


import os
import random

trainval_percent = 0.8 # trainval数据集占所有数据的比例
train_percent = 1 # train数据集占trainval数据的比例
xmlfilepath = '/repository01/CellData/细胞分割voc数据集/VOCdevkit/VOC2007/Annotations/'
txtsavepath = '/repository01/CellData/细胞分割voc数据集/VOCdevkit/VOC2007/ImageSets/Main/'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num*trainval_percent)
tr = int(tv*train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
ftrainval = open(txtsavepath+'trainval.txt', 'w')
ftest = open(txtsavepath+'test.txt', 'w')
ftrain = open(txtsavepath+'train.txt', 'w')
fval = open(txtsavepath+'val.txt', 'w')

for i  in list:
    name = total_xml[i].split(".")[0]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
