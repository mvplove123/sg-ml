#!/usr/bin/env python
# -*- coding: gb18030 -*-
# @Time    : 2017/5/17 15:15
# @Author  : taoyongbo
# @Site    : 
# @File    : ml_utils.py
# @desc    :
import pandas as pd
import numpy as np
import time
from pandas import DataFrame, Series
from scipy.spatial import distance
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import metrics, svm
from sklearn.externals import joblib

# 矩阵转换
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from common_utils import logger, print_exception

model_pkl_path = '/search/odin/taoyongbo/sogou-ml/model/'


# 计算相关性
def compute_similarity(source, target):
    similarity = []
    array_lengh = len(source)
    for i in range(array_lengh):
        result = 1 - distance.cosine(source[i], target[i])
        similarity.append([result])
        if result >= 0 and result <= 0.2:
            similarity.append('无关')
        else:
            similarity.append('相关')
    return similarity


def create_idf_dict():
    idf_vector_dict = {}
    with open('/search/odin/taoyongbo/sogou-ml/model/idf_vetor', encoding='gb18030', mode='r') as idf_lines:
        for idf_line in idf_lines:
            fields = idf_line.strip().split('\t')
            feature_name = fields[0]
            idf_value = float(fields[1])
            idf_vector_dict[feature_name] = idf_value
    return idf_vector_dict


def feature_matrix_transform(vocabulary, fit_data, idf_vector):
    term_vectorizer = CountVectorizer(vocabulary=vocabulary, token_pattern=r"\b\w+\b")
    term_tf_matrix = term_vectorizer.fit_transform(fit_data)
    term_tf_idf_matrix = term_tf_matrix.multiply(idf_vector)
    return term_tf_idf_matrix


def dump_model(clf, model_name):
    joblib.dump(clf, model_pkl_path + model_name)


# 报告分析生成

def analyze_report(clf, X_train, y_train, X_test, y_test, is_gridSearch):
    y_test_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)
    test_score = metrics.accuracy_score(y_true=y_test, y_pred=y_test_pred)

    train_score = metrics.accuracy_score(y_true=y_train, y_pred=y_train_pred)

    train_best_params = ''
    train_estimator_params = ''
    dev_best_score_str = ''
    if is_gridSearch:
        train_best_params = '最佳训练参数:{value}\n'.format(value=clf.best_params_)
        train_estimator_params = '最佳估计参数:{value}\n'.format(value=clf.best_estimator_)

        dev_best_score_str = '最佳训练模型dev准确率:{value}\n'.format(value=clf.best_score_)

    train_score_str = '训练数据准确率:{value}\n'.format(value=train_score)
    test_score_str = '测试数据准确率:{value}\n'.format(value=test_score)

    train_report = metrics.classification_report(y_true=y_train, y_pred=y_train_pred)
    test_report = metrics.classification_report(y_true=y_test, y_pred=y_test_pred)

    train_peport_head = '训练数据详细分类报告'
    test_report_head = '测试数据详细分类报告'

    output_report = '\n'.join(
        (train_best_params, train_estimator_params, dev_best_score_str, train_score_str, test_score_str,
         train_peport_head, train_report, test_report_head, test_report,
         ))

    return output_report


def output_report(report_log, model, clf, X_train, y_train, X_test, y_test, is_gridSearch):
    """
    模型报告输出
    :param report_log: 
    :param model: 
    :param clf: 
    :param X_train: 
    :param y_train: 
    :param X_test: 
    :param y_test: 
    :param is_gridSearch: 
    :return: 
    """
    # 模型持久化
    dump_model_name = '_'.join(
        (model + '_', str(X_train.shape[0]), str(X_test.shape[0]), str(X_train.shape[1]), 'pkl'))
    dump_model(clf, dump_model_name)

    summary = '训练模型:{model},全集:{total_size},训练集:{train_size},测试集:{test_size},特征数:{feature_size}\n'. \
        format(total_size=X_train.shape[0] + X_test.shape[0], model=model,
               train_size=X_train.shape[0], test_size=X_test.shape[0],
               feature_size=X_train.shape[1])

    report = analyze_report(clf, X_train, y_train, X_test, y_test, is_gridSearch)

    report_log.write(summary)
    report_log.write(report)
    report_log.write('\n\n\n')
    report_log.flush()


# 模型选择
def model_fit(X_train, y_train, X_test, y_test, methods, report_log, is_gridSearch):
    try:

        logger.info('begin model_selection for th')
        # Logistic Regression
        if 'LR' in methods:
            model_name = 'LR'

            start_time = time.time()
            if is_gridSearch:

                classifier = LogisticRegression(solver='sag')
                parameters = {'C': [1, 1e01, 1e02],
                              # 'loss ': ['hinge', 'log'],
                              }
                clf = simple_classification(X_train=X_train, y_train=y_train, classifier=classifier, parameters=parameters)
            else:
                classifier = LogisticRegression(C=10, solver='sag', n_jobs=-1)
                clf = classifier.fit(X_train, y_train)

            logger.info('model_selection Logistic Regression fit finished,use_time:{time},begin predict'.format(
                time=time.time() - start_time))
            output_report(report_log=report_log, model=model_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                          clf=clf, is_gridSearch=is_gridSearch)

            # NLB
            if 'NLB' in methods:
                start_time = time.time()
                if is_gridSearch:
                    classifier = BernoulliNB()
                    parameters = {'alpha': [1e-01, 1, 1e01]}
                    clf = simple_classification(X_train=X_train, y_train=y_train, classifier=classifier,
                                                parameters=parameters)

                else:
                    classifier = BernoulliNB(alpha=1)
                    clf = classifier.fit(X_train, y_train)

                logger.info('model_selection BernoulliNB fit finished,use_time:{time},begin predict'.format(
                    time=time.time() - start_time))
                output_report(report_log=report_log, model='LR', X_train=X_train, y_train=y_train, X_test=X_test,
                              y_test=y_test,
                              clf=clf, is_gridSearch=is_gridSearch)
            # SVM
            if 'SVM' in methods:
                classifier = svm.SVC(kernel='rbf', probability=True, class_weight={1: 1})
                parameters = {'C': [0.8]}
                print('\n\n\nresult for SVC')
                clf = simple_classification(X_train=X_train, y_train=y_train, classifier=classifier, parameters=parameters)

            # Decision Tree
            if 'DT' in methods:
                model_name = 'DT'
                start_time = time.time()
                if is_gridSearch:
                    classifier = BernoulliNB()
                    parameters = {'max_depth': [5, 10], 'min_samples_leaf': [3, 5, 10]}
                    clf = simple_classification(X_train=X_train, y_train=y_train, classifier=classifier,
                                                parameters=parameters)
                else:
                    logger.info('DT begin fit')
                    classifier = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5)
                    clf = classifier.fit(X=X_train, y=y_train)

                logger.info('model_selection Decision Tree fit finished,use_time:{time},begin predict'.format(
                    time=time.time() - start_time))
                output_report(report_log=report_log, model=model_name, X_train=X_train, y_train=y_train, X_test=X_test,
                              y_test=y_test,clf=clf, is_gridSearch=is_gridSearch)

            # Random Forest
            if 'RF' in methods:
                classifier = RandomForestClassifier()
                parameters = {'n_estimators': [50, 100], 'min_samples_leaf': [3, 5, 10], 'max_depth': [5, 10]}
                print('\n\n\nresult for Random Forest')
                clf = simple_classification(X_train=X_train, y_train=y_train, classifier=classifier, parameters=parameters)

            # Extra Trees
            if 'ET' in methods:
                classifier = ExtraTreesClassifier(class_weight={0: 1, 1: 5})
                parameters = {}
                print('\n\n\nresult for Extra Trees')
                clf = simple_classification(X_train=X_train, y_train=y_train, classifier=classifier, parameters=parameters)

            if 'GBDT' in methods:
                classifier = GradientBoostingClassifier()
                parameters = {'n_estimators': [100], 'min_samples_leaf': [5, 7, 10], 'max_depth': [5, 7, 10]}
                print('\n\n\nresult for Gradient Boosted Regression Trees')
                clf = simple_classification(X_train=X_train, y_train=y_train, classifier=classifier, parameters=parameters)

            # k-nearest neighbors
            if 'KNN' in methods:
                classifier = KNeighborsClassifier()
                parameters = {'n_neighbors': [3, 4, 5, 6, 7]}
                print('\n\n\nresult for k-nearest neighbors ')
                clf = simple_classification(X_train=X_train, y_train=y_train, classifier=classifier, parameters=parameters)

            if 'NN' in methods:
                scaler = StandardScaler()
                scaler.fit(X_train)
                # X_train = scaler.transform(X_train)
                # X_test = scaler.transform(X_test)
                classifier = MLPClassifier(hidden_layer_sizes=(50))
                parameters = {'alpha': [0.0001, 0.001, 0.01], 'hidden_layer_sizes': [(50, 10)]}
                print('\n\n\nresult for MLPClassifier ')
                clf = simple_classification(X_train=X_train, y_train=y_train, classifier=classifier, parameters=parameters)
    except Exception as e:
        print_exception()

# 工作流
def simple_classification(X_train, y_train, classifier, parameters):
    # 通用的分类器流程
    clf = GridSearchCV(classifier, parameters, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def matrix_fit(train_test, feature_names, idf_vector):
    """
    特征转换
    :param all_data: 
    :param mi_tf_feature: 
    :param idf_vector: 
    :param report_log: 
    :return: 
    """

    transform_time = time.time()

    train, test = train_test_split(train_test, train_size=0.7)
    y_train, y_test = train.category, test.category

    # 训练集（训练，测试）
    X_train = feature_matrix_transform(vocabulary=feature_names, fit_data=train['new_data'], idf_vector=idf_vector)
    # 测试集（测试）
    X_test = feature_matrix_transform(vocabulary=feature_names, fit_data=test['new_data'], idf_vector=idf_vector)

    logger.info(
        'transform feature matrix finished,use_time:{time},begin model fit'.format(time=time.time() - transform_time))

    return X_train, y_train, X_test, y_test
