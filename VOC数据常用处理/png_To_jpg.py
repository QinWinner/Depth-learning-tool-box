#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :png批量转为jpg.py
@说明        :
@时间        :2020/05/07 21:40:09
@作者        :
@版本        :1.0
'''

import os
import string
dirName = "VOCdevkit/VOC2008/JPEGImages/"         #最后要加双斜杠，不然会报错
li=os.listdir(dirName)
for filename in li:
    newname = filename
    newname = newname.split(".")
    if newname[-1]=="JPG":
        newname[-1]="jpg"
        newname = str.join(".",newname)  #这里要用str.join
        filename = dirName+filename
        newname = dirName+newname
        os.rename(filename,newname)
        print(newname,"updated successfully")