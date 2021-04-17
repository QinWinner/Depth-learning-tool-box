
#coco分割数据的可视化，但是是用点进行可视化的
#

# -*- coding: utf-8 -*-
import os
import sys, getopt
from pycocotools.coco import COCO, maskUtils
import cv2
import numpy as np
 
def mkdir_os(path):
    if not os.path.exists(path):
        os.makedirs(path)
 
def main():

 
    inputfile = '/home/qinjian/Segmentation/地理遥感图像分割/aicrowd房屋分割竞赛/val/images'
    jsonfile = '/home/qinjian/Segmentation/地理遥感图像分割/aicrowd房屋分割竞赛/val/annotation-small.json'
    outputfile = '/home/qinjian/Segmentation/地理遥感图像分割/aicrowd房屋分割竞赛/val/show'
 
    
 
 
    mkdir_os(outputfile)
 
    coco = COCO(jsonfile)
    catIds = coco.getCatIds(catNms=['wires'])  # catIds=1 表示人这一类
    imgIds = coco.getImgIds(catIds=catIds)  # 图片id，许多值
    for i in range(len(imgIds)):
        if i % 100 == 0:
            print(i, "/", len(imgIds))
        img = coco.loadImgs(imgIds[i])[0]
 
        cvImage = cv2.imread(os.path.join(inputfile, img['file_name']), -1)
        cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
        cvImage = cv2.cvtColor(cvImage, cv2.COLOR_GRAY2BGR)
 
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
 
        polygons = []
        color = []
        for ann in anns:
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                        poly_list = poly.tolist()
                        polygons.append(poly_list)
                        if ann['iscrowd'] == 0:
                            color.append([0, 0, 255])
                        if ann['iscrowd'] == 1:
                            color.append([0, 255, 255])
                else:
                    exit()
                    print("-------------")
                    # mask
                    t = imgIds[ann['image_id']]
                    if type(ann['segmentation']['counts']) == list:
                        rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                    else:
                        rle = [ann['segmentation']]
                    m = maskUtils.decode(rle)
 
                    if ann['iscrowd'] == 0:
                        color_mask = np.array([0, 0, 255])
                    if ann['iscrowd'] == 1:
                        color_mask = np.array([0, 255, 255])
 
                    mask = m.astype(np.bool)
                    cvImage[mask] = cvImage[mask] * 0.7 + color_mask * 0.3
 
        point_size = 2
        thickness = 2
        for key in range(len(polygons)):
            ndata = polygons[key]
            cur_color = color[key]
            for k in range(len(ndata)):
                data = ndata[k]
                cv2.circle(cvImage, (int(data[0]), int(data[1])), point_size, (cur_color[0], cur_color[1], cur_color[2]), thickness)
        cv2.imwrite(os.path.join(outputfile, img['file_name']), cvImage)
 
 
if __name__ == "__main__":
   main()