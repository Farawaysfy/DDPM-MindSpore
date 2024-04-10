import os

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib.pyplot import plot
from pandas import DataFrame
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose, ToTensor, Lambda
from tqdm import tqdm

from utils.myPlot import MyPlot
from torch.utils.data import Dataset


class Signal:
    def __init__(self, path, fs=5120, slice_length=1024, slice_type='cut'):
        self.path = path
        self.slice_length = slice_length
        self.slice_type = slice_type
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
        # self.stftData = self.saveSTFT()  # 保存STFT图的路径

    def loadMat(self):
        rawData = loadmat(self.path)
        dataLabels = [v[0][0] for v in rawData['data'][0]]
        data = [[v[1].T[0]] for v in rawData['data'][0]]
        # print("data[0]=", data[0])
        slices = []
        if self.slice_type == 'cut':
            slices_labels = [str(self.label) + "_" + str(i) for i in
                             range(len(data[0][0]) // self.slice_length)]  # 生成切片标签
            for v in data:  # 切片
                temp = []
                for i in range(len(v[0]) // self.slice_length):
                    temp.append(np.array(v[0][i * self.slice_length:(i + 1) * self.slice_length]))
                slices.append(temp)
        else:
            slices_labels = [str(self.label) + "_" + str(i) for i in range(len(data[0][0]) - self.slice_length)]
            for v in data:
                temp = []
                left, right = 0, self.slice_length
                while right < len(v[0]):  # 信号重合度为0.5
                    temp.append(np.array(v[0][left:right]))
                    left += self.slice_length // 2
                    right += self.slice_length // 2
                slices.append(temp)
        # print("dataLabels=", dataLabels)
        df = DataFrame(slices, index=dataLabels, columns=slices_labels, )
        # print("df=", df)
        return df

    def saveSTFT(self):
        # 获取行标签
        stftLabels = self.data.index
        slicesLabels = self.data.columns
        slices = []
        for label in self.data.index:
            temp = []
            savePath = self.path.replace('.mat', '') + '\\' + "stft" + str(self.slice_length) + '\\' + label.replace(
                '/', '_')

            if not os.path.exists(savePath):
                os.makedirs(savePath)
            else:
                # 删除文件夹下所有文件
                for root, dirs, files in os.walk(savePath):
                    for name in files:
                        os.remove(os.path.join(root, name))

            for column in tqdm(self.data.columns, desc="正在处理" + savePath + "的STFT图像"):
                data = self.data.loc[label, column]
                plot = MyPlot(data, column, fs=self.fs)

                temp.append(plot.saveSTFT(path=savePath + '\\'))

                # print("现在处理的图像为：", temp[-1])
            slices.append(temp)
        return DataFrame(slices, index=stftLabels, columns=slicesLabels)

    def plotWaveform(self):
        pass


class Signals:
    def __init__(self, path, fs=1024, slice_length=512, slice_type='cut'):
        paths = [path + '\\' + f for f in os.listdir(path) if f.endswith('.mat')]  # 获取所有.mat文件
        if slice_type == 'cut':
            print("当前信号分割模式为切片式（不重复）")
        else:
            print("当前信号分割模式为窗口式（重复）")
        self.signals = [Signal(path, fs, slice_length, slice_type) for path in paths]  # 读取所有.mat文件
        self.labels = [signal.label for signal in self.signals]  # 获取所有信号的标签
        self.df = self.signals[0].data  # 初始化DataFrame
        for i in range(1, len(self.signals)):  # 合并所有信号的数据
            self.df = pd.merge(self.df, self.signals[i].data, left_index=True, right_index=True, how='outer')

    def saveSTFT(self):
        for signal in self.signals:
            signal.saveSTFT()


class PictureData(VisionDataset):

    def __init__(self, path, shape, batch_size: int, data_type: str, slice_length=512, transform=None, merged=True):
        if transform is None:
            transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
        super().__init__(transform=transform)
        self.shape = shape
        self.batch_size = batch_size
        self.data_type = data_type
        self.paths = []
        for root, dirs, files in os.walk(path):
            for name in dirs:
                if data_type in name and name.endswith(str(slice_length)):
                    self.paths.append(os.path.join(root, name))
        self.merged = merged
        self.data_tensor, self.target_tensor = self.getDataSet()

        # self.dataSet = self.getDataSet()

    def process(self):
        for path in self.paths:
            #   将三张图像合并成一个
            root = path

            # 获取所有文件夹
            dirs = os.listdir(root)[-3:]
            length = len(os.listdir(os.path.join(root, dirs[0])))

            for i in tqdm(range(length), desc="正在合并" + path + "的图像"):
                img1 = cv2.imread(os.path.join(path, dirs[0], os.listdir(os.path.join(root, dirs[0]))[i]))
                img2 = cv2.imread(os.path.join(path, dirs[1], os.listdir(os.path.join(root, dirs[1]))[i]))
                img3 = cv2.imread(os.path.join(path, dirs[2], os.listdir(os.path.join(root, dirs[2]))[i]))
                img = np.vstack((img1, img2, img3))

                if img.shape[2] == 3 or img.shape[2] == 4:  # 彩色图像
                    # 将img由三通道转换为单通道
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # img黑白颜色调换

                img = 255 - img
                img = cv2.resize(img, self.shape[1:])

                # cv2.imshow("img", img)
                # cv2.waitKey(0)
                # 保存合并后的图像
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(path + '\\' + str(i) + '.png', img)

    def getDataSet(self):
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
        data = []
        target = []
        for path in self.paths:
            label = -1
            for key in dic:
                if key in path:
                    label = dic[key]
                    break
            root = path
            # 获取所有png
            files = os.listdir(root)
            if self.merged:  # 使用合并的图片
                for file in files[:-3]:
                    if not file.endswith('.png'):
                        continue
                    img = cv2.imread(os.path.join(root, file), cv2.IMREAD_UNCHANGED)
                    # cv2.imshow('img', img)
                    # cv2.waitKey(0)
                    # print("size: ", img.shape)

                    # if img.shape[2] == 3 or img.shape[2] == 4:  # 彩色图像
                    #     # 将img由三通道转换为单通道
                    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    if img.shape[1:] != self.shape[1:]:
                        img = cv2.resize(img, self.shape[1:])
                    # cv2.imshow('gray', img)
                    # # 显示当前图像
                    # cv2.waitKey(0)
                    # print("gray size: ", img.shape)

                    # 将img转换为tensor
                    img = torch.tensor(img, dtype=torch.float32)
                    img = torch.reshape(img, (1, *self.shape[1:]))
                    data.append(img)
                    # 获取标签
                    target.append(label)
            else:  # 使用单独的图片
                for file in files[-3:]:
                    for _, _, sub_files in os.walk(os.path.join(root, file)):
                        for sub_file in sub_files:
                            img = cv2.imread(os.path.join(root, file, sub_file), cv2.IMREAD_UNCHANGED)
                            cv2.imshow('img', img)
                            cv2.waitKey(0)
                            print("size: ", img.shape)

                            if img.shape[2] == 3 or img.shape[2] == 4:  # 彩色图像
                                # 将img由三通道转换为单通道
                                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                            if img.shape[1:] != self.shape[1:]:
                                img = cv2.resize(img, self.shape[1:])
                            cv2.imshow('gray', img)
                            # 显示当前图像
                            cv2.waitKey(0)
                            print("gray size: ", img.shape)

                            img = 255 - img
                            cv2.imshow('dst', img)
                            cv2.waitKey(0)
                            # 将img转换为tensor
                            img = torch.tensor(img, dtype=torch.float32)
                            img = torch.reshape(img, (1, *self.shape[1:]))
                            data.append(img)
                            # 获取标签
                            target.append(label)
        data = torch.stack(data)
        return data, target

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.target_tensor[idx]


def get_dataloader(path, batch_size: int, slice_length=512) -> DataLoader:
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = PictureData(path, get_img_shape(), batch_size,
                          'stft', slice_length=slice_length, transform=transform, merged=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_img_shape():
    return 1, 512, 512


if __name__ == '__main__':
    print('test')
    # dataSet = Signals('../data', slice_length=512)
    # dataSet.saveSTFT()
    # print(dataSet.df)
    # print(dataSet.df[0])
    dataset = PictureData('../data', get_img_shape(), 32, 'stft', slice_length=512, merged=False)
    # dataset.process()
    # print(dataset.paths)
    # dataset.getDataSet()
