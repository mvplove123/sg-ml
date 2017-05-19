#!/usr/bin/env python
# -*- coding: gb18030 -*-
# @Time    : 2017/5/4 14:55
# @Author  : taoyongbo
# @Site    : 
# @File    : query-category-ml.py
# @desc    :
import pandas as pd
import numpy as np
import time

import sys

from logging import getLogger
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report

# sys.path.insert(0, "/search/odin/taoyongbo/sogou-ml/sogou-ml/")


sogou_category_names = ['宾馆饭店', '餐饮服务', '场馆会所', '地名', '房地产', '公司企业', '购物场所', '交通出行', '金融银行', '旅游景点', '其它', '汽车服务',
                        '体育场馆', '新闻媒体', '休闲娱乐', '学校科研', '医疗卫生', '邮政电信', '政府机关']

input_path = '/search/odin/taoyongbo/sogou-ml/data/segment_nav_standard_poi'
output_path = '/search/odin/taoyongbo/sogou-ml/result/query-category-report'

category_name_dict = {}
for i, v in enumerate(sogou_category_names):
    category_name_dict[v] = i


def covert_label(category):
    return category_name_dict[category]


def calclate_dscore(docs_values, len_doc=3):
    dw = len(docs_values)
    pw = dw / len_doc
    max_freq = np.max(docs_values)
    min_freq = np.min(docs_values)
    tw = max_freq / min_freq
    score = (1 + pw * tw) * dw
    return score


def get_feature_names_score(feature_names, transpose_term_lil_data, doc_len):
    feature_names_score_dict = {}
    for i, v in enumerate(feature_names):
        score = calclate_dscore(transpose_term_lil_data[i], doc_len)
        feature_names_score_dict[v] = score
    return feature_names_score_dict


def query_category():
    df = pd.read_csv(input_path, sep='\t', names=['query', 'category', 'data'], encoding='gb18030')
    df = df.dropna()

    vectorizer = CountVectorizer()
    df['label'] = df['category'].apply(func=covert_label)
    train, test = train_test_split(df, train_size=0.7)
    term_matrix = vectorizer.fit_transform(train['data'])
    feature_names = vectorizer.get_feature_names()
    transpose_term_matrix = term_matrix.transpose()
    transpose_term_lil_data = transpose_term_matrix.tolil().data
    doc_len = len(feature_names)
    feature_names_score_dict = get_feature_names_score(feature_names=feature_names,
                                                       transpose_term_lil_data=transpose_term_lil_data, doc_len=doc_len)
    sort_feature_names_dict = sorted(feature_names_score_dict.items(), key=lambda d: d[1], reverse=True)


    run_category(sort_feature_names_dict, train=train, test=test)


def run_category(sort_feature_names_dict, train, test):
    with open(output_path, mode='w', encoding='gb18030') as result_lines:
        for k in range(50000, 51000, 1000):
            filter_feature_names = [v[0] for i, v in enumerate(sort_feature_names_dict) if i <= k]
            train_y = train.label.values
            test_y = test.label.values

            step1 = ('count_vectorizer', CountVectorizer(vocabulary=filter_feature_names))
            step2 = ('tf_transformer', TfidfTransformer())  # 该类会统计每个词语的tf-idf权值
            step3 = ('blb_clf', BernoulliNB())

            pipeline = Pipeline(steps=[step1, step2, step3])

            parameters = {
                'tf_transformer__norm': ['l1'],
                'blb_clf__alpha': [0.5],
            }

            grid_search = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
            grid_search.fit(train['data'], train_y)
            test_y_pre = grid_search.predict(test['data'])
            test_report = classification_report(test_y, test_y_pre, target_names=sogou_category_names)
            result_lines.write(
                'feature_names num:{num},the best score:{score}'.format(num=k, score=grid_search.best_score_))
            result_lines.write('\n')
            result_lines.write('the best param:{param}'.format(param=grid_search.best_params_))
            result_lines.write('\n')
            result_lines.write('test_report:{test_report}'.format(test_report=test_report))
            result_lines.write('\n\n')



if __name__ == '__main__':
    start_time = time.time()
    query_category()


    x=BernoulliNB()
    x.fit()