import numpy as np
import re
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# 读取数据集
def data_in():
    doc_list = []       # 数据集
    class_list = []     # 标签集
    for i in range(1, 26):
        word_list = parse(open('data/spam/%d.txt' % i, 'r').read())
        doc_list.append(word_list)
        class_list.append(1)   # 1垃圾邮件
        word_list = parse(open('data/ham/%d.txt' % i, 'r').read()) 
        doc_list.append(word_list)
        class_list.append(0)   # 0非垃圾邮件
    return doc_list, class_list


# 划分训练集和测试集
def random_split():
    train = list(range(50))
    test = []
    for i in range(15):  # 7:3随机划分训练集和测试集
        index = int(random.uniform(0, len(train)))  # 从去掉抽出的下标的训练集中随机取数字作为测试集的下标
        test.append(train[index])
        del (train[index])
    return train, test


# 将字符串拆分为小写字母的单词
def parse(bigString):
    token_list = re.split(r'\W+', bigString)  # 用重复任意次的非字符作为切分标志
    return [tok.lower() for tok in token_list if len(tok) > 2]  # 转为小写


# 生成词汇表
def create_vocab(doc_list):  # 每封邮件的字符串列表
    vocab_set = set([])  # 不重复的词汇表
    for document in doc_list:
        vocab_set = vocab_set | set(document)  # 取并集保证不重复
    return list(vocab_set)


# 字符串转换为列表
def word_to_vec(vocab_list, input_set):
    turn_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            turn_vec[vocab_list.index(word)] = 1
    return turn_vec


# 贝叶斯过程
def bayes_train(trainMat, trainClasses):
    num_train = len(trainMat)
    num_words = len(trainMat[0])
    p_abusive = sum(trainClasses) / float(num_train)
    # 单词出现次数初始化为1，避免出现0的情况
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p0_denomi = 2.0
    p1_denomi = 2.0
    for i in range(num_train):
        if trainClasses[i] == 1:
            p1_num += trainMat[i]    # 矩阵相加，得到的矩阵表示单词出现情况，p1_num：在每个训练文件中，如果某单词出现过，在对应的地方加一
            p1_denomi += sum(trainMat[i])     # p1_denomi：计算总共出现的不重复的单词数目
        else:
            p0_num += trainMat[i]
            p0_denomi += sum(trainMat[i])
    p1_vec = np.log(p1_num / p1_denomi)   # 垃圾邮件条件概率
    p0_vec = np.log(p0_num / p0_denomi)   # 取对数，便于计算
    return p0_vec, p1_vec, p_abusive


# 比较概率大小进行分类
def classifyNB(vec_classify, p0_vec, p1_vec, p_abusive):
    p1 = sum(vec_classify*p1_vec)+np.log(p_abusive)
    p0 = sum(vec_classify*p0_vec)+np.log(1.0-p_abusive)
    if p1 > p0:
        return 1
    else:
        return 0


# 绘制混淆矩阵
def cm_plot(matrix):
    cm = matrix
    plt.matshow(cm, cmap=plt.cm.Reds)
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('Real Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()


# 主函数
def main_():
    doc_list, class_list = data_in()
    vocab_list = create_vocab(doc_list)
    train, test = random_split()
    trainMat = []
    trainClasses = []
    for docIndex in train:
        trainMat.append(word_to_vec(vocab_list, doc_list[docIndex]))  # 利用词袋模型得到训练矩阵
        trainClasses.append(class_list[docIndex])  # 增加标签
    p0V, p1V, pSpam = bayes_train(np.array(trainMat), np.array(trainClasses))  # 进行贝叶斯过程
    pre_label = []  # 预测标签
    test_true = []  # 实际标签
    for docIndex in test:
        word_vector = word_to_vec(vocab_list, doc_list[docIndex])  # 用词袋模型转化测试集
        pre_label.append(classifyNB(np.array(word_vector), p0V, p1V, pSpam))
        test_true.append(class_list[docIndex])
    matrix = confusion_matrix(test_true, pre_label)  # 得到混淆矩阵
    accuracy = (matrix[0][0] + matrix[1][1]) / len(test)
    return accuracy


# 训练n次求平均正确率
if __name__ == '__main__':
    accuracy = []
    times = int(input("训练次数："))
    for i in range(times):
        accuracy.append(main_())
    accuracy = np.array(accuracy)
    print('垃圾邮件识别平均正确率: %.2f' %(accuracy.mean()))