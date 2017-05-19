#!/usr/bin/env python
# -*- coding: gb18030 -*-
# @Time    : 2017/3/22 18:29
# @Author  : taoyongbo
# @Site    : 
# @File    : common_utils.py
# @desc    :

# ����logger
import linecache
import logging
import os

import sys


def get_logger_obj(log_file_name):
    # ����һ��logger
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.DEBUG)
    # ����һ��handler������д����־�ļ�
    fh = logging.FileHandler(log_file_name + ".log")
    fh.encoding = 'gb18030'
    fh.setLevel(logging.DEBUG)
    # �ٴ���һ��handler���������������̨
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # ����handler�������ʽ
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # ��logger���handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = get_logger_obj('/search/odin/taoyongbo/sogou-ml/code/query-category')

# ��ȡĿ¼�µ������ļ��к��ļ�
def get_files(path):
    global allFileNum

    # �����ļ��У���һ���ֶ��Ǵ�Ŀ¼�ļ���
    dirList = []
    # �����ļ�
    fileList = []
    # ����һ���б����а�����Ŀ¼��Ŀ������(google����)
    files = os.listdir(path)
    for f in files:
        if (os.path.isdir(path + '/' + f)):
            # �ų������ļ��С���Ϊ�����ļ��й���
            if (f[0] == '.'):
                pass
            else:
                # ��ӷ������ļ���
                dirList.append(f)
        if (os.path.isfile(path + '/' + f)):
            # ����ļ�
            fileList.append(path + '/' + f)
    # ��һ����־ʹ�ã��ļ����б��һ�����𲻴�ӡ
    i_dl = 0
    for dl in dirList:
        if (i_dl == 0):
            i_dl = i_dl + 1
    return fileList,dirList
    # for fl in fileList:
    #     # ��ӡ�ļ�
    #     print '-' * (int(dirList[0])), fl
    #     # ������һ���ж��ٸ��ļ�
    #     allFileNum = allFileNum + 1

def print_exception():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
