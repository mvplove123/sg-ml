#!/usr/bin/env python
# -*- coding: gb18030 -*-
# @Time    : 2017/5/18 20:17
# @Author  : taoyongbo
# @Site    : 
# @File    : feature_select.py
# @desc    : 特征选择

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from common_utils import get_logger_obj

feature_select_logger = get_logger_obj('feature_select')

sogou_category_names = ['宾馆饭店', '餐饮服务', '场馆会所', '地名', '房地产', '公司企业', '购物场所', '交通出行', '金融银行', '旅游景点', '其它', '汽车服务',
                        '体育场馆', '新闻媒体', '休闲娱乐', '学校科研', '医疗卫生', '邮政电信', '政府机关']
sogou_category_names.sort()


def calclate_dscore(docs_values, len_doc=3):
    """
    计算频次分数
    :param docs_values: 
    :param len_doc: 
    :return: 
    """
    dw = len(docs_values)
    pw = dw / len_doc
    max_freq = np.max(docs_values)
    min_freq = np.min(docs_values)
    tw = max_freq / min_freq
    score = (1 + pw * tw) * dw
    return score


def calclate_mi(term_info, totalcount_terms, categorys_count):
    """
    计算互信息
    :param term_info: 
    :param totalcount_terms: 
    :param categorys_count: 
    :return: 
    """
    term_totalcount = term_info.loc['count']
    term_categorys_count = term_info.iloc[2:].values
    pwc = term_categorys_count / categorys_count
    pw = term_totalcount / totalcount_terms
    lg = (pwc / pw) + 1
    mi = pwc * np.log(lg.tolist())
    mi_var = np.var(mi)
    return mi_var


def suffix_data(data):
    """
    前中后缀处理
    :param data: 
    :return: 
    """
    new_data = []
    field = data.split(' ')
    if len(field) == 1:
        new_data.append(data + '_o')
    elif len(field) == 2:
        new_data.append(field[0] + '_p')
        new_data.append(field[1] + '_l')
    else:
        new_data.append(field[0] + '_p')
        for i in field[1:len(field) - 1]:
            new_data.append(i + '_m')
        new_data.append(field[len(field) - 1] + '_l')
    return ' '.join(new_data)


def feature_select():
    """
    特征选择
    :return: 
    """
    feature_select_logger.info('load all data')
    df = pd.read_csv('/search/odin/taoyongbo/sogou-ml/data/segment_all_poi', sep='\t',
                     names=['query', 'category', 'data'], encoding='gb18030')
    df = df.dropna()

    df['new_data'] = df['data'].apply(func=suffix_data)

    vectorizer_total = CountVectorizer(token_pattern=r"\b\w+\b")

    total_term_matrix = vectorizer_total.fit_transform(df['new_data'])


    #idf 持久化
    total_idf = TfidfTransformer(norm=None)
    total_idf_vector = total_idf.fit(total_term_matrix)
    with open('/search/odin/taoyongbo/sogou-ml/model/idf_vetor',encoding='gb18030',mode='w') as idf_writer:
        for feature_name, idf in zip(vectorizer_total.get_feature_names(), total_idf.idf_):
            idf_str = '\t'.join((feature_name,str(idf)))+'\n'
            idf_writer.write(idf_str)



    feature_names = vectorizer_total.get_feature_names()
    transpose_term_matrix = total_term_matrix.transpose()
    transpose_term_lil_data = transpose_term_matrix.tolil().data
    doc_len = len(feature_names)

    feature_select_logger.info('get each term frequency')

    # 统计每个term的词频总数
    dd = DataFrame(transpose_term_matrix.sum(axis=1), index=vectorizer_total.get_feature_names(),
                   columns=['count']).sort_values(['count'], ascending=False)
    dd.index.name = 'term'
    dd.reset_index(inplace=True)

    feature_select_logger.info('get term frequency in each category merge to dataframe')

    # 统计每个term 在各个分类的词频总数 并合并在一个dataframe
    for i in sogou_category_names:
        category_query = df.loc[df['category'] == i, :]
        vectorizer = CountVectorizer()
        term_matrix = vectorizer.fit_transform(category_query['new_data'])

        transpose_term_matrix = term_matrix.T
        category_sum = transpose_term_matrix.sum(axis=1)
        category_df = pd.DataFrame(category_sum, columns=[i], index=vectorizer.get_feature_names())
        category_df.index.name = 'term'
        category_df.reset_index(inplace=True)
        dd = dd.merge(category_df, how='left', left_on=['term'], right_on=['term'])
    # 部分词未出现在当前类别下，填充0
    dd.fillna(0, inplace=True)

    # 统计每个类别的词频数
    df.reset_index(inplace=True)
    term_num = DataFrame(total_term_matrix.sum(axis=1), columns=['term_num'])

    # 统计每个类别的词频数
    result = pd.concat([df, term_num], axis=1).groupby(['category']).term_num.sum().sort_index()

    # 统计矩阵的总词频数
    totalcount_terms = transpose_term_matrix.sum()

    feature_select_logger.info('compute mutual information variance ')

    # 计算互信息方差
    dd['var'] = dd.apply(func=calclate_mi, args=(totalcount_terms, result.values), axis=1)

    # 按方差降序排序
    dd.sort_values(by=['var'], ascending=False, inplace=True)

    # 重置行索引
    dd.reset_index(drop=True, inplace=True)

    dd.to_pickle('/search/odin/taoyongbo/sogou-ml/data/mi_feature')
    feature_select_logger.info('compute tf info ')



    # 计算文档频次
    feature_names_dict = {}
    for i, v in enumerate(feature_names):
        score = calclate_dscore(transpose_term_lil_data[i], doc_len)
        feature_names_dict[v] = score

    # 文档频次降序排序并转化dataframe
    sort_feature_names_dict = sorted(feature_names_dict.items(), key=lambda d: d[1], reverse=True)
    feature_df = DataFrame(sort_feature_names_dict, columns=['name', 'score'])
    feature_select_logger.info('df to disk')

    # 频次持久化
    feature_df.to_pickle('/search/odin/taoyongbo/sogou-ml/data/tf_feature')

    # 持久化交集特征值
    feture_num = [300000, 500000]
    feature_tm_df_path = '/search/odin/taoyongbo/sogou-ml/data/tf_mi_feature_select/'
    for i in feture_num:
        feature_df_select = feature_df.loc[:i]
        feature_mi_select = dd.loc[:i, ['term', 'var']]
        current_feature_dataframe = feature_df_select.merge(feature_mi_select, left_on=['name'], right_on=['term'],
                                                     how='inner')
        feature_len = len(current_feature_dataframe)
        file_name = str(i) + '_' + str(feature_len) + '_feature_tm_df'
        current_feature_dataframe.to_csv(feature_tm_df_path + file_name, sep='\t', encoding='gb18030', index=False)

    feature_select_logger.info('intersection for tf,var and save to disk')


if __name__ == '__main__':
    feature_select()
