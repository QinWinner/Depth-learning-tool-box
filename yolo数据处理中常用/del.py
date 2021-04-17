import os
from tqdm import tqdm


def del_label():
    xml_save_path = "/repository01/CellData/细胞分割voc数据集/VOCdevkit/VOC2007/Annotations"
    img_save_path = "/repository01/CellData/细胞分割voc数据集/VOCdevkit/VOC2007/JPRGImages"

    xml_list = os.listdir(xml_save_path)
    img_list = os.listdir(img_save_path)
    img_name = []

    for img in img_list:
        name = img.split(".")[0]
        if(name.find(" ") > 0):
            os.remove(img_save_path+"/"+img)
        else:
            img_name.append(name)

    for xml in xml_list:
        name = xml.split(".")[0]
        if (name not in img_name):
            os.remove(xml_save_path+"/"+xml)



# 标注经过切割后有很多是没有标注的图片，这些部分需要进行删除。
def empty_label():
    label_path = "/repository01/CellData/细胞分割voc数据集/VOCdevkit/VOC2008/labels"
    img_path="/repository01/CellData/细胞分割voc数据集/VOCdevkit/VOC2008/JPRGImages"

    for label in tqdm(os.listdir(label_path)):
        with open(os.path.join(label_path,label),"r") as f:
            lines=f.readlines()
            if(lines==[]):
                name=label.split(".")[0]+".jpg"
                os.remove(os.path.join(label_path,label))
                os.remove(os.path.join(img_path,name))


if __name__ == "__main__":
    del_label()