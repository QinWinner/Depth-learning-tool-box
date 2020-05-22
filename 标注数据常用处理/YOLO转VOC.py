import cv2
import os
xml_head = '''<annotation>
    <folder>VOC2007</folder>
    <!--文件名-->
    <filename>{}</filename>.
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>325991873</flickrid>
    </source>
    <owner>
        <flickrid>null</flickrid>
        <name>null</name>
    </owner>    
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>{}</depth>
    </size>
    <segmented>0</segmented>
    '''
xml_obj = '''
    <object>        
        <name>{}</name>
        <pose>Rear</pose>
        <!--是否被裁减，0表示完整，1表示不完整-->
        <truncated>0</truncated>
        <!--是否容易识别，0表示容易，1表示困难-->
        <difficult>0</difficult>
        <!--bounding box的四个坐标-->
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
    '''
xml_end = '''
</annotation>'''
 
labels = ['A','B','C']#label for datasets
 
cnt = 0
with open('list.txt','r') as train_list:
    for lst in train_list.readlines():
        lst = lst.strip()
        jpg = lst#image path
        txt = lst.replace('.jpg','.txt')#yolo label txt path
        xml_path = 'Annotations/'+jpg.replace('.jpg','.xml')#xml save path
 
        obj = ''
 
        img = cv2.imread(jpg)
        img_h,img_w = img.shape[0],img.shape[1]
        head = xml_head.format(str(jpg),str(img_w),str(img_h))
        with open(txt,'r') as f:
            for line in f.readlines():
                yolo_datas = line.strip().split(' ')
                label = int(float(yolo_datas[0].strip()))
                center_x = round(float(str(yolo_datas[1]).strip()) * img_w)
                center_y = round(float(str(yolo_datas[2]).strip()) * img_h)
                bbox_width = round(float(str(yolo_datas[3]).strip()) * img_w)
                bbox_height = round(float(str(yolo_datas[4]).strip()) * img_h)
 
                xmin = str(int(center_x - bbox_width / 2 ))
                ymin = str(int(center_y - bbox_height / 2))
                xmax = str(int(center_x + bbox_width / 2))
                ymax = str(int(center_y + bbox_height / 2))
 
                obj += xml_obj.format(labels[label],xmin,ymin,xmax,ymax)
        with open(xml_path,'w') as f_xml:
            f_xml.write(head+obj+xml_end)
        cnt += 1
        print(cnt)