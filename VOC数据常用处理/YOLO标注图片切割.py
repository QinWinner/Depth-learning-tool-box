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
@版本        :1.0
'''


import os
from cv2 import cv2

currpat=os.path.dirname(os.path.abspath(__file__))
os.chdir(currpat)


image_dir="JPEGImages/"
labels_dir="labels/"
image_out="JPEGImagesc/"
labels_out="labelc/"
if not os.path.exists(image_out):
        os.makedirs(image_out)
if not os.path.exists(labels_out):
        os.makedirs(labels_out)


image_list = os.listdir(image_dir)
labels_list = os.listdir(labels_dir)


#下面的用来转换为YOLO的格式
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = round(x*dw,6)
    w = round(w*dw,6)
    y = round(y*dh,6)
    h = round(h*dh,6)
    return (x,y,w,h)


# 这里的n表示图片要切成w_cut*h_cut块。 
def cut(w_cut,h_cut):
	number=0
	for image in image_list:
		for label in labels_list:
			label_str=[]
			label_=[]
			img=cv2.imread(image_dir+image)
			if(image.split(".")[0]==label.split(".")[0]):
				file1=open(labels_dir+label,"r",encoding="utf-8")
				label_str=file1.readlines()
				file1.close
			else:
				continue
			##以上代码用来分别读取对应的图片和标记文件

			size=img.shape
			leng=size[0]
			hight=size[1]
			for lab in label_str:
				lab=lab.split()
				no=float(lab[0])
				l1=float(lab[1])
				l2=float(lab[2])
				l3=float(lab[3])
				l4=float(lab[4])
				w=l3*leng
				h=l4*hight
				x1=round(l1*leng+1-w/2,6)
				y1=round(l2*hight+1-h/2,6)

				x2=x1+w
				y2=y1+h
				label_.append([no,x1,y1,x2,y2])
			#以上代码是YOLO的归一化位置改为绝对值位置。

			cut_w=int(leng/w_cut)
			cut_h=int(hight/h_cut)
			x=cut_w
			y=cut_h

			im=cv2.imread(image_dir+image)

			while x<=leng:
				while y<=hight:
					cv2.imwrite(image_out+str(number)+".jpg",im[y-cut_h:y,x-cut_w:x]) #切分图片
					with open(labels_out+str(number)+".txt","w",encoding="utf-8") as file:
						for lab_temp in label_:
							classlfy=int(lab_temp[0])
							x1=float(lab_temp[1])  #左上
							y1=float(lab_temp[2])
							x2=float(lab_temp[3])  #右下
							y2=float(lab_temp[4])

							#没有重叠的情况下
							if( (y1>=y) | (x1>=x) | (y2<=y-cut_h) | (x2<=x-cut_w) ):
								continue
							else:  #有交叉的部分
								x1=x1-(x-cut_w)
								if(x1<0):
									x1=0
								if(x1>cut_w):
									x1=cut_w

								y1=y1-(y-cut_h)
								if(y1<0):
									y1=0
								if(y1>cut_h):
									y1=cut_h

								x2=x2-(x-cut_w)
								if(x2<0):
									x2=0
								if(x2>cut_w):
									x2=cut_w

								y2=y2-(y-cut_h)
								if(y2<0):
									y2=0
								if(y2>cut_h):
									y2=cut_h
								
								z1,z2,z3,z4=convert((cut_w,cut_h),(x1,x2,y1,y2))
								file.write(str(classlfy)+" "+str(z1)+" "+str(z2)+" "+str(z3)+" "+str(z4)+"\n")
					y=y+cut_h
					number=number+1
				x=x+cut_w
				y=cut_h


if __name__ == "__main__":
	cut(4,4) 


			




