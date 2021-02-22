import os
import shutil

source_path = os.path.abspath(r'D:\cv\data\voc_rpc2\JPEGImages')  # 源文件夹
target_path = os.path.abspath(r'D:\cv\data\voc_rpc3\JPEGImages')  # 目标文件夹

if not os.path.exists(target_path):  # 目标文件夹不存在就新建
    os.makedirs(target_path)

if os.path.exists(source_path):  # 源文件夹存在才执行
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)

    for root, dirs, files in os.walk(source_path):
        for file in files:
            src_file = os.path.join(root, file)
            shutil.copy(src_file, target_path)
            print(src_file)

print('复制完成')