#!/usr/bin/env python
# -*- coding: gb18030 -*-
# @Time    : 2017/3/22 18:29
# @Author  : taoyongbo
# @Site    : 
# @File    : common_utils.py
# @desc    :

# 创建logger
import linecache
import logging
import os

import sys


def get_logger_obj(log_file_name):
    # 创建一个logger
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.DEBUG)
    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(log_file_name + ".log")
    fh.encoding = 'gb18030'
    fh.setLevel(logging.DEBUG)
    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = get_logger_obj('/search/odin/taoyongbo/sogou-ml/code/query-category')

# 获取目录下的所有文件夹和文件
def get_files(path):
    global allFileNum

    # 所有文件夹，第一个字段是次目录的级别
    dirList = []
    # 所有文件
    fileList = []
    # 返回一个列表，其中包含在目录条目的名称(google翻译)
    files = os.listdir(path)
    for f in files:
        if (os.path.isdir(path + '/' + f)):
            # 排除隐藏文件夹。因为隐藏文件夹过多
            if (f[0] == '.'):
                pass
            else:
                # 添加非隐藏文件夹
                dirList.append(f)
        if (os.path.isfile(path + '/' + f)):
            # 添加文件
            fileList.append(path + '/' + f)
    # 当一个标志使用，文件夹列表第一个级别不打印
    i_dl = 0
    for dl in dirList:
        if (i_dl == 0):
            i_dl = i_dl + 1
    return fileList,dirList
    # for fl in fileList:
    #     # 打印文件
    #     print '-' * (int(dirList[0])), fl
    #     # 随便计算一下有多少个文件
    #     allFileNum = allFileNum + 1

def print_exception():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
