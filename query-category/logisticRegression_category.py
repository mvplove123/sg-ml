#!/usr/bin/env python
# -*- coding: gb18030 -*-
# @Time    : 2017/5/17 15:07
# @Author  : taoyongbo
# @Site    : 
# @File    : logisticRegression_category.py
# @desc    :
import datetime
import time

import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from common_utils import logger, get_files, print_exception
from ml_utils import model_fit, matrix_fit
import multiprocessing

now = datetime.datetime.now().strftime('%Y%m%d_%H:%M')
report_folder_path = '/search/odin/taoyongbo/sogou-ml/data/report/'


def suffix_data(data):
    new_data = []
    field = data.split(' ')
    if len(field) == 1:
        new_data.append(data + '_o')
    elif len(field) == 2:
        new_data.append(field[0] + '_p')
        new_data.append(field[1] + '_l')
    else:
        new_data.append(field[0] + '_p')
        new_data.append(field[len(field) - 1] + '_l')
        for i in field[1:len(field) - 1]:
            new_data.append(i + '_m')

    return ' '.join(new_data)


def execute_task(all_data, fileList, idf_vector, shuffle_num):
    try:
        print('begin execute')
        start_time = time.time()

        report_log_path = report_folder_path + now + '_' + str(shuffle_num) + '_report_log'

        report_log = open(report_log_path, mode='w', encoding='gb18030')

        train_test = shuffle(all_data, n_samples=shuffle_num)
        logger.info('shuffle_num:{shuffle_num} shuffle finished'.format(shuffle_num=shuffle_num))
        for file in fileList:
            logger.info(file)
            mi_tf_feature = pd.read_csv(file, sep='\t', encoding='gb18030')

            final_feature_names = mi_tf_feature['name'].values

            X_train, y_train, X_test, y_test = matrix_fit(train_test=train_test, feature_names=final_feature_names,
                                                          idf_vector=idf_vector)

            methods = ['LR']

            model_fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, methods=methods,
                      report_log=report_log, is_gridSearch=True)

        logger.info('report output,use_time:{time}'.format(time=time.time() - start_time))

        report_log.close()
    except Exception as e:
        print_exception()


def main():
    """
    特征入库
    :return: 
    """
    try:
        start_time = time.time()

        logger.info('begin load data')

        all_data = pd.read_pickle('/search/odin/taoyongbo/sogou-ml/data/all_data.pickel')
        fileList, dirList = get_files('/search/odin/taoyongbo/sogou-ml/data/tf_mi_feature_select/')
        idf_vector = joblib.load('/search/odin/taoyongbo/sogou-ml/model/idf_vetor')

        logger.info(
            'load data finished,use_time:{time},begin transform feature matrix'.format(time=time.time() - start_time))

        n_samples = [5000000, 10000000]

        # pool = multiprocessing.Pool(processes=5)

        for i in n_samples:
            logger.info('feature num{num} begin train'.format(num=i))
            execute_task(all_data, fileList, idf_vector, i)
            # pool.apply_async(execute_task, (all_data, fileList, idf_vector, i,))
            # pool.close()
            # pool.join()
    except Exception as e:
        print_exception()


if __name__ == '__main__':
    main()
