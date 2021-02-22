import os
import shutil
import pandas as pd
import csv


def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


def csv_contact(Folder_Path, SaveFile_Path, SaveFile_Name):
    """
    合并一个文件夹下的所有CSV文件
    :param Folder_Path: 原始CSV文件所在文件夹
    :param SaveFile_Path: 目标存储文件夹
    :param SaveFile_Name: 存储文件名
    :return:
    """
    # Folder_Path = r'D:\cv\data\voc_rpc2\csv'  # 要拼接的文件夹及其完整路径，注意不要包含中文
    # SaveFile_Path = r'D:\cv\data\voc_rpc2\csv'  # 拼接后要保存的文件路径
    # SaveFile_Name = r'all3.csv'  # 合并后要保存的文件名

    # 修改当前工作目录
    os.chdir(Folder_Path)
    # 将该文件夹下的所有文件名存入一个列表
    file_list = os.listdir()

    # 读取第一个CSV文件并包含表头
    df = pd.read_csv(Folder_Path + '\\' + file_list[0])  # 编码默认UTF-8，若乱码自行更改

    # 将读取的第一个CSV文件写入合并后的文件保存
    df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8", index=False)

    # 循环遍历列表中各个CSV文件名，并追加到合并后的文件
    for i in range(1, len(file_list)):
        df = pd.read_csv(Folder_Path + '\\' + file_list[i])
        df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, encoding='utf-8', index=False, header=False, mode='a+')


def copy_file_from_csv(original_path, target_path, csv_path):
    """
    目前csv考虑的是文件路径，故使用路径中的文件名，保存xml文件
    根据CSV文件中的路径拷贝文件到指定文件夹
    :param original_path: 图片源文件地址
    :param target_path: 文件存储地址
    :param csv_path: csv文件路径
    :return:
    """
    with open(csv_path, "rt", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        rows = [row for row in reader]
        mkr(target_path)
        for row in rows:
            name = row[0].split('/')
            fig_name = name[3].split('.')
            xml_path = fig_name[0] + '.xml'
            full_path = os.path.join(original_path, xml_path)
            shutil.copy(full_path, target_path + '/')
