import numpy as np
from numpy import *
import random

MAXITERA = 5


def load_data(file_path):
    dataMat = [];
    labelMat = []
    fr = open(file_path)
    for line in fr.readlines():
        lineArr = line.strip().split(' ')  # 去除制表符，将数据分开

        dataMat.append([float(lineArr[1]), float(lineArr[2]), float(lineArr[3]), float(lineArr[4])])  # 数组矩阵
        if lineArr[5] == '"setosa"':
            labelMat.append(1)  # 标签
        else:
            labelMat.append(-1)  # 标签
    return dataMat, labelMat


class SVM():
    def __init__(self, data, label, max_itera, C, tolar):

        self.data = mat(data)
        self.label = mat(label)
        # print("----------{}".format(self.label.shape))
        self.max_itera = max_itera
        self.C = C
        self.num = len(self.data)       # the num of dataset
        self.w = mat(zeros([self.data.shape[1], 1]))
        self.b = 0
        self.alpha = mat(np.random.randn(self.num)) #mat(zeros([1, self.num]))      # size is num of dataset
        self.tolar = tolar

    # type 1:line kernel; 2:RBF Kernel
    def kernel_fuction(self, x1, x2, type = 1):

        if type == 1:
            return x1 * x2.T

    def select_j(self, i):
        j = i
        while j == i:
            j = int(random.uniform(0, self.num))
        return j

    def calculate_gx(self, i):
        a = self.kernel_fuction(self.data, self.data[i], 1)
        b = multiply(self.alpha, self.label)
        print("a shape {}\n b shape {}".format(a.shape, b.shape))
        return float(sum(b * a)) + self.b

    def get_L_H(self, alpha_1, alpha_2, i, j):
        if self.label[:, i] == self.label[:, j]:
            L = max(0, alpha_2 - alpha_1)
            H = min(self.C, self.C + alpha_2 + alpha_1)
        else:
            L = max(0, alpha_1 + alpha_2 - self.C)
            H = min(self.C, alpha_2 + alpha_1)
        return L, H

    def SMO(self):
        for idx in range(self.max_itera):
            for i in range(self.num):
                gx_i = self.calculate_gx(i)    # calculate the g(x)
                print("Gxi {}".format(self.label[:, i]))
                E_i = gx_i - self.label[:, i]     # calculate the cost according label
                # print("Ei shape{}".format(self.label[:, i]))
                # optimization constraints
                if (E_i * self.label[:, i] > self.tolar and self.alpha[:, i] < self.C) \
                        or (E_i * self.label[:, i] < self.tolar and self.alpha[:, i] > 0):
                    j = self.select_j(i)
                    gx_j = self.calculate_gx(j)
                    E_j = gx_j = self.label[:, j]

                    alphaold_i = self.alpha[:, i]
                    alphaold_j = self.alpha[:, j]
                    L, H = self.get_L_H(alphaold_i, alphaold_j, i, j)
                    if L == H:
                        print('L=H')
                        continue

                    eta = self.kernel_fuction(self.data[i], self.data[j], 1) + self.kernel_fuction(self.data[j], self.data[j], 1) \
                        - 2 * self.kernel_fuction(self.data[i], self.data[j], 1)
                    self.alpha[:, j] = alphaold_j + self.label[:, j] * (E_j - E_i) / eta

                    if self.alpha[:, j] > H:
                        self.alpha[:, j] = H
                    if self.alpha[:, j] < L:
                        self.alpha[:, j] = L

                    self.alpha[:, i] = alphaold_i + self.label[:, i] * self.label[:, j] * (alphaold_j - self.alpha[:, j])

                    b1_new = -E_i - self.label[:, i] * self.kernel_fuction(self.data[i], self.data[i], 1) * (self.alpha[:, i] - alphaold_i) \
                        - self.label[:, j] * self.kernel_fuction(self.data[j], self.data[i]) * (self.alpha[:, j] - alphaold_j) + self.b

                    b2_new = -E_j - self.label[:, i] * self.kernel_fuction(self.data[i], self.data[j]) * (self.alpha[:, i] - alphaold_i) \
                        - self.label[:, j] * self.kernel_fuction(self.data[j], self.data[j]) * (self.alpha[:, j] - alphaold_j) + self.b
                    if self.alpha[:, i] > 0 and self.alpha[:, i] < self.C:
                        self.b = b1_new
                    if self.alpha[:, j] > 0 and self.alpha[:, j] < self.C:
                        self.b = b2_new
                    else:
                        self.b = (b1_new + b2_new)/2

            print("iater {}".format(idx))

    def calculate_w(self):
        self.w = mat(zeros((self.data.shape[1], 1)))
        for i in range(self.num):
            # print("{} {}".format(self.alpha.shape, self.data[i].shape))
            self.w += multiply(self.alpha[:, i] * self.label[:, 1], self.data[i]).T

    def predict(self, test_data, label):
        pre_value = 0
        error = 0
        self.calculate_w()
        l = len(test_data)
        print(self.w)
        for i in range(l):
            pre_value = self.w * test_data[i] + self.b
            if sign(int(pre_value)) != sign(int(label[:, i])):
                error += 1
        print("the acc:{}".format(len((l - error) / l)))


if __name__ == '__main__':
    data, label = load_data("iris.txt")
    model = SVM(data, label, 10, 0.6, 0.01)
    model.SMO()
    model.predict(data, label)

    # x1 = np.eye(5, 4)
    # x2 = np.eye(1, 4)
    # print(x1)
    # print(x2.shape)
    # print(x1 * x2)
