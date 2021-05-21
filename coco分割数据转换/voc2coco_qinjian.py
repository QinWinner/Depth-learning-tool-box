# 用于将voc的分割数据转换为coco的标注格式，网上基本都是目标检测的转换，我门这里做的是分割的转换

import cv2
import json
import os
import numpy as np
from tqdm import tqdm
from skimage import measure

START_BOUNDING_BOX_ID = 1  # ann开始的id
Image_ID=1      #image_id 开始的id
PRE_DEFINE_CATEGORIES = {"building": 1}


def convert(images_dir, labels_dir, json_save_path):
    json_dict = {"images": [], "annotations": [],
                 "categories": []}

    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    image_id = Image_ID

    images_name = os.listdir(images_dir)
    images_name = sorted(images_name)  # 对文件按照文件名进行排序

    for image_name in tqdm(images_name):  # 对于每一个文件进行处理image 是文件的名字

        
        
        temp_image = cv2.imread(os.path.join(images_dir, image_name))
        width, height, deep = temp_image.shape
        image = {'file_name': image_name, 'height': height, 'width': width,
                 'id': image_id}
        

        

        label_image = cv2.imread(os.path.join(labels_dir, image_name), -1)
        _, block_im = cv2.threshold(label_image, 1, 255, cv2.THRESH_BINARY)
        _,contours, hierarchy = cv2.findContours(block_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        Flag_labe=False
        for contour in contours:

            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            
            area = cv2.contourArea(contour)
            if area<10.0:                                      #设置目标的最小面积，可以不设置，但是我再转换的过程中发现很多area=0.0 的情况，所以最好还是设置一下
                continue
            
            #rect = cv2.minAreaRect(contour) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            #box = cv2.boxPoints(rect)
			x, y, w, h = cv2.boundingRect(contour)
            annotation = {
                "segmentation": [segmentation],
                "area": area,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [x, y, w, h],
                "category_id": 1,
                "id": bnd_id
            }
            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1
            Flag_labe=True  #说明整个图片有可用的标注的信息

        # 写入文件Images部分
        if(Flag_labe):
            json_dict['images'].append(image)


        image_id=image_id+1

        
    # 写入categories
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_save_path, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

if __name__ == "__main__":
    # convert(images_dir="/home/qinjian/Segmentation/地理遥感图像分割/0.25高清房屋数据/val/images", 
    #         labels_dir="/home/qinjian/Segmentation/地理遥感图像分割/0.25高清房屋数据/val/labels",
    #         json_save_path="/home/qinjian/Segmentation/地理遥感图像分割/0.25高清房屋数据/val/annotation.json")
    convert(images_dir="/home/qinjian/Segmentation/地理遥感图像分割/0.25高清房屋数据/images", 
            labels_dir="/home/qinjian/Segmentation/地理遥感图像分割/0.25高清房屋数据/labels",
            json_save_path="/home/qinjian/Segmentation/地理遥感图像分割/0.25高清房屋数据/annotation.json")