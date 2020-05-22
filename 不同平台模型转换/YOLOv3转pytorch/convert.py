#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :convert.py
@说明        :用来吧yolo本身的模型转换为pytorch文件
@时间        :2020/05/23 02:24:32
@作者        :秦健
@版本        :1.0
'''


from __future__ import division
from models import *
import torch
import os


if __name__ == "__main__":


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs("checkpoints", exist_ok=True)

    # Initiate model
    model = Darknet("config/yolov3-voc3.cfg").to(device)

    model.load_darknet_weights("weights/yolov3-voc3.weights")

    torch.save(model, "checkpoints/yolov3.pth")
