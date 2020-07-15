import numpy as np
import datetime
import struct
from multiprocessing import Process, Manager, Lock


# 读取数据集中的图片文件，返回一个28*28矩阵，每个行向量代表一张图
def read_image(file_now):
    file_in = open(file_now, "rb").read()
    number_prev, image_account, row, col = struct.unpack_from('>IIII', file_in, 0)
    offset = struct.calcsize('>IIII')
    # 读取之后的图片信息
    image_size = row * col
    nx_image = '>' + str(image_size) + 'B'
    images = np.empty((image_account, image_size))  # 新建大矩阵存图片信息
    for i in range(image_account):
        images[i] = np.array(struct.unpack_from(nx_image, file_in, offset))
        offset += struct.calcsize(nx_image)
    return images


# 读取与图片相对应的储存标签的文件，返回一个一维数组，数组的元素值即为对应图片的数字值
def read_label(file_now):
    file_in = open(file_now, "rb").read()
    number_prev, labels_num = struct.unpack_from('>II', file_in, 0)
    offset = struct.calcsize('>II')
    # 读取到的标签信息
    nx_label = '>' + str(labels_num) + 'B'
    labels = np.array(struct.unpack_from(nx_label, file_in, offset))
    return labels


# K近邻算法得到测试图片的预测值
def KNN(test_image, test_label, train_image, train_labels, k, process_pre, test_number, process_number, error_lock,
        errors_list):
    for test_count in range(process_pre, test_number, process_number):
        # 读取训练集的行数
        train_image_num = train_image.shape[0]
        # 通过欧式距离的大小来判断图片的相似度
        all_distances = (np.sum((np.tile(test_image[test_count], (train_image_num, 1)) - train_image) ** 2,
                                axis=1)) ** 0.5
        # 按all_distances中元素进行升序排序后得到其对应索引的列表
        sorted_distance_index = all_distances.argsort()
        # 选择距离最小的k个样本，看一下它们中大部分都是哪个数字的样本
        classCount = np.zeros((10), dtype=int)
        for i in range(k):
            vote_label = train_labels[sorted_distance_index[i]]
            classCount[vote_label] += 1
        # 出现最多的数字样本为预测值
        result_label = -1
        time_max = 0
        for i in range(10):
            if classCount[i] >= time_max:
                time_max = classCount[i]
                result_label = i
        print("进程", process_pre + 1, ":第", test_count + 1, "张测试图片：", "预测值:", result_label, "真实值:",
              test_label[test_count], end='')
        if (result_label != test_label[test_count]):
            save_error(test_count, error_lock, errors_list)
            print('…………（错误！）', end='')
        print(' ')


# 保存并记录预测错误的测试样本
def save_error(err_count, error_lock, errors_list):
    with error_lock:  # 因为errors_list是存放在进程公共通信区的，所以需要加把锁
        errors_list.append(err_count)
        

# 程序入口
def main():
    t1 = datetime.datetime.now()  # 计时开始
    k = int(input('选择最邻近的K个值，K='))
    test_number = int(input('选择测试样本的数目(1-10000):'))
    process_number = int(input('选择计算进程数:'))
    # 读取文件
    train_image = read_image('MNIST_data\\train-images.idx3-ubyte')
    train_label = read_label('MNIST_data\\train-labels.idx1-ubyte')
    test_image = read_image('MNIST_data\\t10k-images.idx3-ubyte')
    test_label = read_label('MNIST_data\\t10k-labels.idx1-ubyte')
    # 设置进程共享区域，访问锁
    errors = Manager()
    error_lock = Lock()
    errors_list = errors.list([])
    # 批量新建和开始进程
    process_list = []
    for i in range(process_number):
        p = Process(target=KNN, args=(test_image, test_label, train_image, train_label, k, i, test_number, process_number, error_lock, errors_list))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()
    # 打印预测结果
    print("\nk值为:  ", k)
    print("样本数: ", test_number)
    print("进程数: ", process_number)
    print("错误数: ", len(errors_list))
    print("错误率= {:.2f}%".format(len(errors_list) / float(test_number) * 100))
    t2 = datetime.datetime.now()
    print('耗时= ', t2 - t1)


if __name__ == "__main__":
    main()