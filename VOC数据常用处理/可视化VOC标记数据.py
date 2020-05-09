#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :show.py
@说明        :将VOC的标记数据可视化
@时间        :2020/05/07 21:43:34
@作者        :
@版本        :1.0
'''


import os
import cv2
import re
 
pattens = ['name','xmin','ymin','xmax','ymax']
 
def get_annotations(xml_path):
    bbox = []
    with open(xml_path,'r') as f:
        text = f.read().replace('\n','return')
        p1 = re.compile(r'(?<=<object>)(.*?)(?=</object>)')
        result = p1.findall(text)
        for obj in result:
            tmp = []
            for patten in pattens:
                p = re.compile(r'(?<=<{}>)(.*?)(?=</{}>)'.format(patten,patten))
                if patten == 'name':
                    tmp.append(p.findall(obj)[0])
                else:
                    tmp.append(int(float(p.findall(obj)[0])))
            bbox.append(tmp)
    return bbox
 
def save_viz_image(image_path,xml_path,save_path):
    bbox = get_annotations(xml_path)
    image = cv2.imread(image_path)
    for info in bbox:
        cv2.rectangle(image,(info[1],info[2]),(info[3],info[4]),(255,0,0),thickness=2)
        cv2.putText(image,info[0],(info[1],info[2]),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cv2.imwrite(os.path.join(save_path,image_path.split('/')[-1]),image)
 
if __name__ == '__main__':
    image_dir = 'JPEGImages'
    xml_dir = 'Annotations'
    save_dir = 'viz_images'
    image_list = os.listdir(image_dir)
    for i in  image_list:
        image_path = os.path.join(image_dir,i)
        xml_path = os.path.join(xml_dir,i.replace('.jpg','.xml'))
        save_viz_image(image_path,xml_path,save_dir)