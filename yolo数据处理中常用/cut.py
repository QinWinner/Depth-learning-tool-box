#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :cut.py
@说明        :在运行yolov3算法时，自己的数据集图片太大，如果直接resize成算法输入大小，会破坏原先的RGB结构，所以这里采用裁剪的方式
             原本的数据为yolo格式
			 转化yolo格式为:type x_min y_min x_max y_max
             之后裁剪并且归一化形式为yolo

@时间        :2020/05/08 02:04:59
@作者        :秦健
@版本        :2.0
'''


import os
from cv2 import cv2
from tqdm import tqdm
currpat = os.path.dirname(os.path.abspath(__file__))
os.chdir(currpat)


image_dir = "/repository01/CellData/细胞分割voc数据集/VOCdevkit/VOC2007/JPEGImages/"
labels_dir = "/repository01/CellData/细胞分割voc数据集/VOCdevkit/VOC2007/labels/"
image_out = "/repository01/CellData/细胞分割voc数据集/VOCdevkit/VOC2008/JPEGImages/"
labels_out = "/repository01/CellData/细胞分割voc数据集/VOCdevkit/VOC2008/labels/"
if not os.path.exists(image_out):
    os.makedirs(image_out)
if not os.path.exists(labels_out):
    os.makedirs(labels_out)


image_list = sorted(os.listdir(image_dir))
labels_list = sorted(os.listdir(labels_dir))
print(len(image_list),len(labels_list))
assert len(image_list)==len(labels_list)    #标注和图片应该是一一对应的关系

# 应为两个名字是一样的所以可以借用Sort 排序实现对齐操作


# 下面的用来转换为YOLO的格式
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0# - 1
    y = (box[2] + box[3])/2.0# - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = max(x*dw, 0)  # 保证边界不会小于0
    w = max(w*dw, 0)
    y = max(y*dh, 0)
    h = max(h*dh, 0)
    return (x, y, w, h)


# 这里的n表示图片要切成w_cut*h_cut块。
def cut(w_leng, h_leng):

    for i, image in tqdm(enumerate(image_list)):

        # for label in labels_list:
        # 	label_str = []
        # 	label_ = []

        # 	if(image.split(".")[0] == label.split(".")[0]):
        # 		file1 = open(labels_dir+label, "r", encoding="utf-8")
        # 		label_str = file1.readlines()
        # 		file1.close
        # 	else:
        # 		continue
        label_str = []
        label_ = []
        label = labels_list[i]
        assert label.split(".")[0]==image.split(".")[0] ##图片和标注的名字应该是一一对应的
        file1 = open(labels_dir+label, "r", encoding="utf-8")
        label_str = file1.readlines()
        file1.close

        # 以上代码用来分别读取对应的图片和标记文件
        img = cv2.imread(image_dir+image)
        size = img.shape
        leng = size[0]
        hight = size[1]
        for lab in label_str:
            lab = lab.split()
            no = float(lab[0])
            l1 = float(lab[1])
            l2 = float(lab[2])
            l3 = float(lab[3])
            l4 = float(lab[4])
            w = l3*leng
            h = l4*hight
            x1 = round(l1*leng-w/2, 6)
            y1 = round(l2*hight-h/2, 6)

            x2 = x1+w
            y2 = y1+h
            label_.append([no, x1, y1, x2, y2])
        # 以上代码是YOLO的归一化位置改为绝对值位置。

        cut_w = w_leng
        cut_h = h_leng
        x = cut_w
        y = cut_h

        name = image.split(".")[0]
        number = 0
        while x <= leng:
            while y <= hight:
                for lab_temp in label_:
                    classlfy = int(lab_temp[0])
                    x1 = float(lab_temp[1])  # 左上
                    y1 = float(lab_temp[2])
                    x2 = float(lab_temp[3])  # 右下
                    y2 = float(lab_temp[4])

                    # # 没有重叠的情况下
                    # if((y1 >= y) | (x1 >= x) | (y2 <= y-cut_h) | (x2 <= x-cut_w)):
                    #     continue

                    # 没有重叠的情况下,考虑到边缘太近的不作为有效标注，这里35是应为一个中心粒的大小大概就是这么大
                    if((y1 >= y-35) | (x1 >= x-35) | (y2 <= y-cut_h+35) | (x2 <= x-cut_w+35)):
                        #如果出现一种情况，当标注框很小，刚好处于内外矩形的中间区域，这种情况也是需要保留的
                        if(x2<x and y2<y and x1> x-cut_w and y1>y-cut_h):
                            continue        #目前暂时还不不切这一种情况了
                        else:
                            continue
                    else:  # 有交叉的部分
                        x1 = x1-(x-cut_w)
                        if(x1 < 0):
                            x1 = 0
                        if(x1 > cut_w):
                            x1 = cut_w

                        y1 = y1-(y-cut_h)
                        if(y1 < 0):
                            y1 = 0
                        if(y1 > cut_h):
                            y1 = cut_h

                        x2 = x2-(x-cut_w)
                        if(x2 < 0):
                            x2 = 0
                        if(x2 > cut_w):
                            x2 = cut_w

                        y2 = y2-(y-cut_h)
                        if(y2 < 0):
                            y2 = 0
                        if(y2 > cut_h):
                            y2 = cut_h

                        z1, z2, z3, z4 = convert(
                            (cut_w, cut_h), (x1, x2, y1, y2))

                        cv2.imwrite(image_out+name+"_"+str(number) +
                                    ".jpg", img[y-cut_h:y, x-cut_w:x])  # 切分图片
                        with open(labels_out+name+"_"+str(number)+".txt", "w", encoding="utf-8") as file:
                            file.write(str(classlfy)+" "+str(z1) +
                                       " "+str(z2)+" "+str(z3)+" "+str(z4)+"\n")
                y = y+cut_h
                number = number+1
            x = x+cut_w
            y = cut_h


if __name__ == "__main__":
    cut(1024, 1024)
