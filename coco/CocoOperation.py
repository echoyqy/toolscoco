from pycocotools.coco import COCO
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import csv
import os
import glob
import sys
import pandas as pd
from random import sample
import random
from datatool import mkr



def write_xml(ann_path, head, objs, tail, obj_str):
    """
    :param ann_path:
    :param head:
    :param objs:
    :param tail:
    :param obj_str:
    :return:
    """
    f = open(ann_path, "w")
    f.write(head)
    for obj in objs:
        f.write(obj_str % (obj[0], obj[1], obj[2], obj[3], obj[4]))
    f.write(tail)


def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


def save_annotations_and_images(coco, data_dir, dataset, filename, objs, ann_dir, img_dir, head_str, tail_str, obj_str):
    # eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path = os.path.join(ann_dir, filename[:-3] + 'xml')
    img_path = os.path.join(data_dir, dataset, filename)
    print(img_path)
    dst_imgpath = os.path.join(img_dir, filename)

    img = cv2.imread(img_path)
    if (img.shape[2] == 1):
        print(filename + " not a RGB image")
        return
    shutil.copy(img_path, dst_imgpath)

    head = head_str % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tail_str
    write_xml(anno_path, head, objs, tail, obj_str)


def show_img(coco, data_dir, dataset, img, classes, cls_id, classes_names, drinks, show=True):
    I = Image.open('%s/%s/%s' % (data_dir, dataset, img['file_name']))
    # 通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        class_name = classes[ann['category_id']]
        if class_name in classes_names:
            print(class_name)
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                if class_name in drinks:
                    obj = ['drink', xmin, ymin, xmax, ymax]
                else:
                    obj = ['others', xmin, ymin, xmax, ymax]
                objs.append(obj)
                draw = ImageDraw.Draw(I)
                draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()

    return objs


class CocoOperation:
    def __init__(self):
        self.save_path = 'd:/cv/data/voc_rpc2'
        self.img_dir = os.path.join(self.save_path, 'JPEGImages')
        self.anno_dir = os.path.join(self.save_path, 'Annotations')
        self.datasets_list = ['train2019']
        self.data_dir = 'd:/cv/data/rpc-archive/retail_product_checkout'
        self.headstr = """\
                <annotation>
                    <folder>VOC</folder>
                    <filename>%s</filename>
                    <source>
                        <database>My Database</database>
                        <annotation>COCO</annotation>
                        <image>flickr</image>
                        <flickrid>NULL</flickrid>
                    </source>
                    <owner>
                        <flickrid>NULL</flickrid>
                        <name>company</name>
                    </owner>
                    <size>
                        <width>%d</width>
                        <height>%d</height>
                        <depth>%d</depth>
                    </size>
                    <segmented>0</segmented>
                """

        self.objstr = """\
            <object>
                <name>%s</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>%d</xmin>
                    <ymin>%d</ymin>
                    <xmax>%d</xmax>
                    <ymax>%d</ymax>
                </bndbox>
            </object>
        """

        self.tailstr = '''\
        </annotation>
        '''

    def coco_to_voc(self):
        mkr(self.save_path)
        mkr(self.img_dir)
        mkr(self.anno_dir)
        for dataset in self.datasets_list:
            # ./COCO/annotations/instances_train2014.json
            annFile = '{}/annotations/instances_{}.json'.format(self.data_dir, dataset)

            # COCO API for initializing annotated data
            coco = COCO(annFile)
            cats = coco.loadCats(coco.getCatIds())
            classes_names = [cat['name'] for cat in cats]
            drinks = [cat['name'] for cat in cats if
                      cat['supercategory'] == 'instant_drink' or cat['supercategory'] == 'drink']
            '''
            COCO 对象创建完毕后会输出如下信息:
            loading annotations into memory...
            Done (t=0.81s)
            creating index...
            index created!
            至此, json 脚本解析完毕, 并且将图片和对应的标注数据关联起来.
            '''
            # show all classes in coco
            classes = id2name(coco)
            print(classes)
            # show class_ids
            classes_ids = coco.getCatIds(catNms=classes_names)
            print(classes_ids)
            for cls in classes_names:
                # Get ID number of this class
                cls_id = coco.getCatIds(catNms=[cls])
                img_ids = coco.getImgIds(catIds=cls_id)
                if cls in drinks:
                    img_ids = random.sample(img_ids, 10)
                else:
                    img_ids = random.sample(img_ids, 1)
                print(cls, len(img_ids))
                # imgIds=img_ids[0:10]
                for imgId in tqdm(img_ids):

                    img = coco.loadImgs(imgId)[0]
                    filename = img['file_name']
                    # print(filename)
                    objs = show_img(coco, self.data_dir, dataset, img, classes, cls_id, classes_names, drinks, show=False)
                    print(objs)
                    save_annotations_and_images(coco, self.data_dir, dataset, filename, objs, self.anno_dir, self.img_dir,
                                                self.headstr, self.tailstr, self.objstr)


class PascalVOC2CSV(object):
    def __init__(self, xml=[], ann_path=r'D:/cv/data/voc_rpc6/annotations4.csv', classes_path=r'D:/cv/data/voc_rpc6/classes.csv'):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param ann_path: ann_path
        :param classes_path: classes_path
        '''
        self.xml = xml
        self.ann_path = ann_path
        self.classes_path = classes_path
        self.label = []
        self.annotations = []

        self.data_transfer()
        self.write_file()

    def data_transfer(self):
        for num, xml_file in enumerate(self.xml):
            try:
                # print(xml_file)
                # 进度输出
                sys.stdout.write('\r>> Converting image %d/%d' % (
                    num + 1, len(self.xml)))
                sys.stdout.flush()

                # xml_file = 'd:/cv/data/voc_rpc6/change_Annotations/20180903-15-35-43-2666.xml'
                with open(xml_file, 'r') as fp:

                    for p in fp:
                        if '<filename>' in p:
                            # self.filen_ame = p.split('>')[1].split('<')[0]

                            self.filen_ame = xml_file.split('\\')[1].split('.')[0]+'.jpg'

                        if '<object>' in p:
                            # 类别
                            d = [next(fp).split('>')[1].split('<')[0] for _ in range(9)]
                            self.supercategory = d[0]
                            if self.supercategory not in self.label:
                                self.label.append(self.supercategory)

                            # 边界框
                            x1 = int(d[-4])
                            y1 = int(d[-3])
                            x2 = int(d[-2])
                            y2 = int(d[-1])

                            self.annotations.append(
                                [os.path.join('./data/JPEGImages', self.filen_ame), x1, y1, x2, y2,
                                 self.supercategory])
            except:
                continue

        sys.stdout.write('\n')
        sys.stdout.flush()

    def write_file(self, ):
        with open(self.ann_path, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(self.annotations)

        class_name = sorted(self.label)
        class_ = []
        for num, name in enumerate(class_name):
            class_.append([name, num])
        with open(self.classes_path, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(class_)


def split_train_val(csv_path, choice_number):
    df = pd.read_csv(csv_path, header=None)
    other_list = df[(df[5] == 'others')].index.tolist()
    drink_list = df[(df[5] == 'drink')].index.tolist()
    try:
        final_other = sample(other_list, choice_number)
    except:
        final_other = []
    final_drink = sample(drink_list, choice_number)
    select_index = final_other + final_drink
    # data_df = df['num'].loc[select_index]
    data_df = df[df.index.isin(select_index)]
    # 划分训练集和测试集
    train = data_df.sample(frac=0.8, random_state=0, axis=0)
    val = data_df[~data_df.index.isin(train.index)]
    train.to_csv(r'D:/cv/data/voc_rpc6/train_annotations.csv', index=None)
    val.to_csv(r'D:/cv/data/voc_rpc6/val_annotations.csv', index=None)


if __name__ == '__main__':
    """
    把coco数据集写为voc数据集格式
    """
    # coco = CocoOperation()
    # coco.coco_to_voc()
    xml_file = glob.glob(r'd:/cv/data/voc_rpc6/change_Annotations/*.xml')
    PascalVOC2CSV(xml_file)
    # split_train_val(r'D:/cv/data/voc_rpc6/annotations.csv', 1955)


