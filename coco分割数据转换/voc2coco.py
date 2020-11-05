import json
import numpy as np
from pycocotools import mask
from skimage import measure
import cv2
import os
import sys


if sys.version_info[0] >= 3:
    unicode = str
__author__ = 'hcaesar'
import io

def maskToanno(ground_truth_binary_mask,ann_count,category_id):
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    annotation = {
        "segmentation": [],
        "area": ground_truth_area.tolist(),
        "iscrowd": 0,
        "image_id": ann_count,
        "bbox": ground_truth_bounding_box.tolist(),
        "category_id": category_id,
        "id": ann_count
    }
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)
    return annotation


block_mask_path="/home/qinjian/Segmentation/地理遥感图像分割/aicrowd房屋分割竞赛/房屋/labels/building"
# mouse_mask_path="D:\\6Ddataset\\XieSegmentation\\2mouse\\mouse_mask_thresh"
block_mask_image_files=os.listdir(block_mask_path)
# mouse_mask_image_files=os.listdir(mouse_mask_path)
jsonPath="/home/qinjian/Segmentation/地理遥感图像分割/aicrowd房屋分割竞赛/房屋/blockmouseAll.json"
annCount=0
imageCount=0
path="/home/qinjian/Segmentation/地理遥感图像分割/aicrowd房屋分割竞赛/房屋/images"
rgb_image_files=os.listdir(path)
rgb_image_files=sorted(rgb_image_files)
with io.open(jsonPath, 'w', encoding='utf8') as output:
    # 那就全部写在一个文件夹好了
    # 先写images的信息
    output.write(unicode('{\n'))
    output.write(unicode('"images": [\n'))
    for image in rgb_image_files:
        output.write(unicode('{'))
        annotation = {
            "height": 300,
            "width": 300,
            "id": imageCount,
            "file_name": image
        }
        str_ = json.dumps(annotation,indent=4)
        str_ = str_[1:-1]
        if len(str_) > 0:
            output.write(unicode(str_))
            imageCount = imageCount + 1
            output.write(unicode('},\n'))
    output.write(unicode('],\n'))
    #接下来写cate
    output.write(unicode('"categories": [\n'))
    output.write(unicode('{\n'))
    categories={
        "supercategory": "building",
        "id": 0,
        "name": "building"
    }
    str_ = json.dumps(categories, indent=4)
    str_ = str_[1:-1]
    if len(str_) > 0:
        output.write(unicode(str_))
    output.write(unicode('},\n'))
    # output.write(unicode('{\n'))
    # categories={
    #     "supercategory": "mouse",
    #     "id": 1,
    #     "name": "mouse"
    # }
    # str_ = json.dumps(categories, indent=4)
    # str_ = str_[1:-1]
    # if len(str_) > 0:
    #     output.write(unicode(str_))
    # output.write(unicode('}\n'))
    output.write(unicode('],\n'))
    output.write(unicode('"annotations": [\n'))
    for i in range(len(block_mask_image_files)):
    #for (block_image,mouse_image) in (block_mask_image_files,mouse_mask_image_files):
        block_image=block_mask_image_files[i]
        # mouse_image=mouse_mask_image_files[i]
        #output.write(unicode('{\n'))
        block_im=cv2.imread(os.path.join(block_mask_path,block_image),0)
        # mouse_im = cv2.imread(os.path.join(mouse_mask_path, mouse_image), 0)
        _,block_im=cv2.threshold(block_im,1,1,cv2.THRESH_BINARY)
        # _, mouse_im = cv2.threshold(mouse_im, 100, 1, cv2.THRESH_BINARY)
        block_im=np.array(block_im).astype(np.uint8)
        # mouse_im = np.array(mouse_im).astype(np.uint8)
        block_anno=maskToanno(block_im,annCount,0)
        # mouse_anno = maskToanno(mouse_im, annCount,1)
        str_block = json.dumps(block_anno,indent=4)
        str_block = str_block[1:-1]
        # str_mouse = json.dumps(mouse_anno,indent=4)
        # str_mouse = str_mouse[1:-1]
        if len(str_block) > 0:
            output.write(unicode('{\n'))
            output.write(unicode(str_block))
            output.write(unicode('},\n'))
            # output.write(unicode('{\n'))
            # # output.write(unicode(str_mouse))

            #output.write(unicode('},\n'))
            annCount = annCount + 1
    output.write(unicode(']\n'))
    output.write(unicode('}\n'))

