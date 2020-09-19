'''
找个代码用于把rgb颜色的标记图片转化为我们分割任务常用的one-hot格式 或者是引索图模式
其中，引索图也就是一张位深为8的图片，其中的每一个像素值就是找个地方像素的类别。直接看起来是全黑的一张图片

editer@ qianjian
'''

import cv2
import numpy as np
import os


# 把标注文件转化位one-hot格式的样子
def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x


def conver():
    '''
    设置好对应的像素和类别的对印表就可以进行转换了。

    built-up 红色 密集建筑物 （255，0，0）
    farmland 绿色 农田 （0，255，0）
    forest 浅蓝  森林 （0，255，255）
    meadow 黄色 草地 （255,255,0）
    water 深蓝   水面 （0，0，255）

    其他 黑色（0，0，0）
    '''

    palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 255],
               [255, 255, 0], [0, 0, 255]]
    gt_onehot = mask_to_onehot(gt, palette)  # one-hot 后 gt的shape=[H, W, 6]


##############################################################################
# 下面开始时rgb和引索图的转化 https://www.jianshu.com/p/e4e30463b2a4


VOC_COLORMAP = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 255],
                [255, 255, 0], [0, 0, 255]]

VOC_CLASSES = ['background', 'built-up',
               "farmland", 'forest', 'meadow', 'water']

# 方法一


def color2index(color):
    """
    color: List of [R, G, B]
    """
    if color in VOC_COLORMAP:
        return VOC_COLORMAP.index(color)
    else:
        return 0


def proc_one_img(img):
    h, w = img.shape[:2]
    idx = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            idx[i, j] = color2index(img[i, j].tolist())
    return idx

# 方法二


def color2index2(color):
    """
    color: List of [R, G, B]
    """
    try:
        return VOC_COLORMAP.index(color.tolist())
    except ValueError:
        return 0


def proc_one_img_v2(img):
    """
    img: Np.array with shape HxWxC

    """
    h, w = img.shape[:2]
    mask = np.array(list(map(color2index2, img.reshape(-1, 3)))).reshape(h, w)
    return mask


# 方法三
# 现在我们来定义上述映射矩阵: 注意由于256**3数值较大, 需使用32位整数
mapMatrix = np.zeros(256*256*256, dtype=np.int32)
for i, cm in enumerate(VOC_COLORMAP):
    mapMatrix[cm[0]*65536 + cm[1]*256 + cm[2]] = i


def SegColor2Label(img, mapMatrix):
    """
    img: Shape [h, w, 3]
    mapMatrix: color-> label mapping matrix, 
               覆盖了Uint8 RGB空间所有256x256x256种颜色对应的label

    return: labelMatrix: Shape [h, w], 像素值即类别标签
    """
    data = img.astype('int32')
    indices = data[:, :, 0]*65536 + data[:, :, 1]*256 + data[:, :, 2]
    return mapMatrix[indices]


###################################################
# 下面还有一种建立方法,也可以直接建立one-hot编码
    """其中的color_codes 是一个对应的字典。
    {(0, 0, 255): 0, (0, 147, 108): 1, (0, 207, 48): 2, (0, 234, 21): 3}
    """
def rgb2label(img, color_codes=None, one_hot_encode=False):
    if color_codes is None:
        color_codes = {val: i for i, val in enumerate(
            set(tuple(v) for m2d in img for v in m2d))}
    n_labels = len(color_codes)
    result = np.ndarray(shape=img.shape[:2], dtype=int)
    #result[:, :] = -1 我这里小小改动让他默认不是-1 而是0 背景
    result[:, :] = 0
    for rgb, idx in color_codes.items():
        result[(img == rgb).all(2)] = idx

    if one_hot_encode:
        one_hot_labels = np.zeros((img.shape[0], img.shape[1], n_labels))
        # one-hot encoding
        for c in range(n_labels):
            one_hot_labels[:, :, c] = (result == c).astype(int)
        result = one_hot_labels

    return result, color_codes
    """    
    img = cv2.imread("/home/qinjian/Segmentation/地理遥感图像分割/unet-nested-multiple-classification/qinjian.png")
    img_labels, color_codes = rgb2label(img)
    print(color_codes)  # e.g. to see what the codebook is

    img1 = cv2.imread("/home/qinjian/Segmentation/地理遥感图像分割/unet-nested-multiple-classification/qinjian.png")
    img1_labels, _ = rgb2label(img1, color_codes)  # use the same codebook
    """

##下面用来测试一些图片的信息
def test(path):
    img=cv2.imread(path)
    print(img.shape)
    np.savetxt("a.txt",img[:,:,0])
    np.savetxt("b.txt",img[:,:,1])
    np.savetxt("c.txt",img[:,:,2])
    np.savetxt("d.txt",img[:,:,0]==img[:,:,1].all())


if __name__ == "__main__":
    
    duiying={(0, 0, 0):0, (255, 0, 0):1, (0, 255, 0):2, (0, 255, 255):3,(255, 255, 0):4, (0, 0, 255):5}

    # img=cv2.imread("qinjian.png")
    # mask=proc_one_img(img)
    # print(mask.shape)
    # np.savetxt("a.txt",mask)
    # cv2.imwrite("b.png",mask)

    path="/home/qinjian/Segmentation/地理遥感图像分割/VOC2012/labels/"
    name_lists=os.listdir(path)
    for name in name_lists:

        img=cv2.imread(path+name)
        result, _=rgb2label(img,duiying)
        cv2.imwrite("/home/qinjian/Segmentation/地理遥感图像分割/VOC2012/label-8bit/"+name,result)




    
 