# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import math
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time


# 预处理数据集
def data_load(file_name):
    data = pd.read_csv(file_name)
    # x，y分别代表数据和标签
    x = data.values[:, :-1]
    y = data.values[:, -1]

    # 以7:3的比例随机划分训练集与测试集
    train_num = random.sample(range(0, 3167), 2218)
    test_num = list(set(range(0, 3167)).difference(set(train_num)))

    train_data = np.array(x)[train_num]
    train_label = np.array(y)[train_num]
    test_data = np.array(x)[test_num]
    test_label = np.array(y)[test_num]

    # 标准化数据
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    # PCA降维处理到11维
    pca_std = PCA(n_components=11).fit(train_data)
    train_data = pca_std.transform(train_data)
    test_data = pca_std.transform(test_data)

    return train_data, train_label, test_data, test_label


# 求高斯分布需要的参数并存入字典
def get_para(train_data, train_label):
    male_list = []
    female_list = []
    male = female = 0
    # 男声和女声的序号
    train_length = len(train_label)
    for i in range(train_length):
        if train_label[i] == 'male':
            male_list.append(i)
            male += 1
        else:
            female_list.append(i)
            female += 1

    para_for_cal = {}
    for i in range(11):

        # male
        fea_data = train_data[male_list, i]
        mean = fea_data.mean()  # 计算出第i个属性的均值
        std = fea_data.std()    # 计算出第i个属性的标准差
        para_for_cal[(i, 'male')] = (mean, std)

        # female
        fea_data = train_data[female_list, i]
        mean = fea_data.mean()  # 计算出第i个属性的均值
        std = fea_data.std()    # 计算出第i个属性的标准差
        para_for_cal[(i, 'female')] = (mean, std)

    return para_for_cal, male, female, train_length


# 一维正态分布函数
def gaussian(x, mean, std):
    return 1 / (math.sqrt(math.pi * 2) * std) * math.exp((-(x - mean) ** 2) / (2 * std * std))


# 计算后验概率P(feature = x|C)
def possibility(feature_Index, x, C, para_for_cal):
    fea_para = para_for_cal[(feature_Index, C)]
    mean = fea_para[0]
    std = fea_para[1]
    ans = gaussian(x, mean, std)
    return ans


# 高斯贝叶斯过程
def bayes(X, para_for_cal, male, female, train_length):
    # 求先验概率
    male_para = male / train_length
    female_para = female / train_length
    # 朴素贝叶斯
    result = []
    x_length = len(X)
    for i in range(x_length):
        ans_male = math.log(male_para)
        ans_female = math.log(female_para)
        x_length_i = len(X[i])
        for j in range(x_length_i):
            ans_male += math.log(possibility(j, X[i][j], 'male', para_for_cal))
            ans_female += math.log(possibility(j, X[i][j], 'female', para_for_cal))
        if ans_male > ans_female:
            result.append('male')
        else:
            result.append('female')
    return result


# 主模块
def main_():
    train_data, train_label, test_data, test_label = data_load('voice.csv')                     # 加载数据集
    para_for_cal,male,female,train_length = get_para(train_data, train_label)                   # 求高斯分布参数
    predict_label = bayes(test_data, para_for_cal, male, female, train_length)                  # 通过贝叶斯进行预测
    matrix = confusion_matrix(test_label, predict_label, labels=['male', 'female'])             # 得出混淆矩阵
    lst = [matrix[0][0]/(1584-male), matrix[1][1]/(1584-female)]
    return lst


if __name__ == '__main__':
    correct_rate = []
    time_start = time.time()
    correct_rate = main_()
    time_end = time.time()
    print('male：  correct rate: %.4f      mistaken rate: %.4f' %(correct_rate[0],1-correct_rate[0]))
    print('female：correct rate: %.4f      mistaken rate: %.4f' %(correct_rate[1],1-correct_rate[1]))
    print('time elapsed: %fs' %(time_end - time_start))
