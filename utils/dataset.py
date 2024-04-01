import os

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.io import loadmat

from plot.stft import MyPlot


class Signal:
    def __init__(self, path, fs=1024):
        self.path = path
        self.fs = fs
        dic = {
            'Aligned': 0,
            'Bearing': 1,
            'Bowed': 2,
            'Broken': 3,
            'Normal': 4,
            'Parallel': 5,
            'SWF': 6,
            'Unbalance': 7,
        }
        fileName = self.path.split('\\')[-1]
        self.label = -1
        for key in dic:
            if key in fileName:
                self.label = dic[key]
                break
        df = self.loadMat()
        self.data = df.loc[['TimeData/Motor/S_x', 'TimeData/Motor/R_y', 'TimeData/Motor/T_z'],]  # 选择三个轴的数据

    def loadMat(self):
        rawData = loadmat(self.path)
        dataLabels = [v[0][0] for v in rawData['data'][0]]
        data = [[v[1].T[0]] for v in rawData['data'][0]]
        # print("data[0]=", data[0])
        slicesLabels = [str(self.label) + "_" + str(i) for i in range(len(data[0][0]) // self.fs)]  # 生成切片标签
        slices = []
        for v in data:  # 切片
            temp = []
            for i in range(len(v[0]) // self.fs):
                temp.append(np.array(v[0][i * self.fs:(i + 1) * self.fs]))
            slices.append(temp)
        # print("dataLabels=", dataLabels)
        df = DataFrame(slices, index=dataLabels, columns=slicesLabels, )
        # print("df=", df)
        return df

    def saveSTFT(self):
        # 获取行标签
        for label in self.data.index:
            temp = label.replace('/', '_')
            for column in self.data.columns:
                data = self.data.loc[label, column]
                plot = MyPlot(data, temp + '_' + column)
                savePath = self.path.replace('.mat', '')
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                plot.saveSTFT(path=savePath + '\\')
        # plot = MyPlot(, self.label)

        pass

    def plotWaveform(self):
        pass


class Signals:
    def __init__(self, path):
        paths = [path + '\\' + f for f in os.listdir(path) if f.endswith('.mat')]  # 获取所有.mat文件
        self.signals = [Signal(path) for path in paths]  # 读取所有.mat文件
        self.labels = [signal.label for signal in self.signals]  # 获取所有信号的标签
        self.df = self.signals[0].data  # 初始化DataFrame
        for i in range(1, len(self.signals)):  # 合并所有信号的数据
            self.df = pd.merge(self.df, self.signals[i].data, left_index=True, right_index=True, how='outer')

    def saveSTFT(self):
        for signal in self.signals:
            signal.saveSTFT()


if __name__ == '__main__':
    print('test')
    dataSet = Signals('../data')
    print(dataSet.df)
    dataSet.saveSTFT()
    # print(dataSet.df[0])
