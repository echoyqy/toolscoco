#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 将一个文件夹下图片按比例分在两个文件夹下，比例改0.7这个值即可
import os
import random
from shutil import copy2
from datatool import mkr
import pandas as pd


def imagenet_train_val_division(dest_dir, goods_name, orig_dir):
    """
    用于imagenet数据集选取特定的种类，同时划分训练集和测试集
    :param dest_dir: 目标数据集位置
    :param goods_name: 待选择商品种类名
    :param orig_dir: 原始数据集位置
    :return:
    """
    for item in goods_name:
        # dir = os.path.join('D:\\cv\\data\\transform-data2\\rpc\\train', item, 'images')
        total_files = os.listdir(orig_dir)
        num_train = len(total_files)
        print("num_train: " + str(num_train))
        index_list = list(range(num_train))
        print(index_list)
        random.shuffle(index_list)
        num = 0
        trainDir = os.path.join(dest_dir, 'train', item)
        mkr(trainDir)
        validDir = os.path.join(dest_dir, 'val', item)
        mkr(validDir)
        for i in index_list:
            fileName = os.path.join(orig_dir, total_files[i])
            if num < num_train * (2/3):
                print(str(fileName))
                copy2(fileName, trainDir)
            else:
                copy2(fileName, validDir)
            num += 1


def voc_csv_division(ori_path, train_csv_path, val_csv_path, ratio, exlude_str=None):

    """
    输入CSV文件进行训练集和验证集的划分
    :param exlude_str: 删除包含特定字符串的行
    :param ratio: 验证集所占比列
    :param ori_path: 原始CSV路径
    :param train_csv_path: 训练CSV保存路径
    :param val_csv_path: 验证csv保存路径
    :return:
    如：
    ori_path = r'D:/cv/data/voc_rpc6/annotations3.csv'
    train_csv_path = r'D:/cv/data/voc_rpc6/val_annotations.csv'
    val_csv_path = r'D:/cv/data/voc_rpc6/train_annotations.csv'
    """
    df = pd.read_csv(ori_path, encoding='utf-8', header=None)
    if exlude_str:
        df = df[~ df[0].str.contains(exlude_str)]
    # df.drop_duplicates(keep='first', inplace=True)  # 去重，只保留第一次出现的样本
    df = df.sample(frac=ratio)  # 全部打乱
    cut_idx = int(round(0.1 * df.shape[0]))
    df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
    df_test.to_csv(val_csv_path, index=None)
    df_train.to_csv(train_csv_path, index=None)
    print(df.shape, df_test.shape, df_train.shape)  # (3184, 12) (318, 12) (2866, 12)


