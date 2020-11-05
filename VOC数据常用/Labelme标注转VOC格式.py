
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:51:00 2019
@author: Andrea
https://blog.csdn.net/Maisie_Nan/article/details/102662911
"""
 
import os
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split
#1.标签路径
labelme_path = "/home/qinjian/细胞检测平台代码/segmentation_label/segimage1/"              #原始labelme标注数据路径
saved_path = "/home/qinjian/细胞检测平台代码/segmentation_label/VOC/"                #保存路径
 
#2.创建要求文件夹
if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")
    
##3.获取待处理文件
#files = glob(labelme_path + "*.json")
#print(files)
#files = [i.split("/")[-1].split(".json")[0] for i in files]
 
#4.读取标注信息并写入 xml
#for json_file_ in files:
for json_file_ in os.listdir(labelme_path):
    if('.json' not in json_file_):
        continue
    else:
        json_file_ = json_file_.split('.json')[0]
    print(json_file_)
    json_filename = os.path.join(labelme_path , json_file_ + ".json")
    print(json_filename)
    json_file = json.load(open(json_filename,"r",encoding="utf-8"))
    print(os.path.join(labelme_path , json_file_ +".jpg"))
    height, width, channels = cv2.imread(os.path.join(labelme_path , json_file_ +".jpg")).shape
    with codecs.open(saved_path + "Annotations/"+json_file_ + ".xml","w","utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'UAV_data' + '</folder>\n')
        xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>The UAV autolanding</database>\n')
        xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
        xml.write('\t\t<image>flickr</image>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t</source>\n')
        xml.write('\t<owner>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t\t<name>Yuanyiqin</name>\n')
        xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>'+ str(width) + '</width>\n')
        xml.write('\t\t<height>'+ str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for multi in json_file["shapes"]:
            points = np.array(multi["points"])
            xmin = min(points[:,0])
            xmax = max(points[:,0])
            ymin = min(points[:,1])
            ymax = max(points[:,1])
            label = multi["label"]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>'+str(label)+'</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                print(json_filename,xmin,ymin,xmax,ymax,label)
        xml.write('</annotation>')
        
#5.复制图片到 VOC2007/JPEGImages/下
image_files = glob(labelme_path + "*.jpg")
print("copy image files to VOC007/JPEGImages/")
for image in image_files:
    shutil.copy(image,saved_path +"JPEGImages/")
    
#6.split files for txt
txtsavepath = saved_path + "ImageSets/Main/"
ftrainval = open(txtsavepath+'/trainval.txt', 'w')
ftest = open(txtsavepath+'/test.txt', 'w')
ftrain = open(txtsavepath+'/train.txt', 'w')
fval = open(txtsavepath+'/val.txt', 'w')
total_files = glob("./VOC2007/Annotations/*.xml")
total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
#test_filepath = ""
for file in total_files:
    ftrainval.write(file + "\n")
#test
#for file in os.listdir(test_filepath):
#    ftest.write(file.split(".jpg")[0] + "\n")
#split
train_files,val_files = train_test_split(total_files,test_size=0.15,random_state=42)
#train
for file in train_files:
    ftrain.write(file + "\n")
#val
for file in val_files:
    fval.write(file + "\n")
 
ftrainval.close()
ftrain.close()
fval.close()
#ftest.close()
