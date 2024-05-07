import os

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.io import loadmat
from torch import float32, tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from tqdm import tqdm

from utils.FFTPlot import FFTPlot, processImg


def createFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # 删除文件夹下所有文件
        for root, dirs, files in os.walk(path):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))


class Signal:
    def __init__(self, path, fs=5120, slice_length=1024, slice_type='cut', add_noise=False):
        self.path = path
        self.slice_length = slice_length
        self.slice_type = slice_type
        self.fs = fs
        self.add_noise = add_noise
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
        # fileName = self.path.split('\\')[-1]
        self.label = -1
        for key in dic:
            if key in path:
                self.label = dic[key]
                break
        df = self.loadMat()
        self.data = df.loc[['TimeData/Motor/S_x', 'TimeData/Motor/R_y', 'TimeData/Motor/T_z'],]  # 选择三个轴的数据
        # self.stftData = self.saveSTFT()  # 保存STFT图的路径

    def loadMat(self):
        rawData = loadmat(self.path)
        dataLabels = [v[0][0] for v in rawData['data'][0]]
        data = [[v[1].T[0]] for v in rawData['data'][0]]
        data = np.array(data)  # 将数据转换为numpy数组
        if self.add_noise:
            original, data = generate_mixed_signal_data(data)
            # 绘制原始信号和带噪声的信号
            # plt.figure(figsize=(18, 6))
            # plt.plot(data[0][0], label='Noisy Signal', linestyle='--', linewidth=1)
            # plt.plot(original[0][0], label='Original Signal', linewidth=1)
            # plt.xlabel("Sample Index")
            # plt.ylabel("Amplitude")
            # plt.legend()
            # plt.show()
            # plt.close()
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
        elif self.slice_type == 'window':
            for v in data:
                temp = []
                left, right = 0, self.slice_length
                while right < len(v[0]):  # 信号重合度为0.5
                    temp.append(np.array(v[0][left:right]))
                    left += self.slice_length // 2
                    right += self.slice_length // 2
                slices.append(temp)
            slices_labels = [str(self.label) + "_" + str(i) for i in range(len(slices[0]))]
        else:
            for v in data:  # 存储原始数据
                slices.append([v[0]])
            slices_labels = [str(self.label) + "_" + str(i) for i in range(len(slices[0]))]
        # print("dataLabels=", dataLabels)
        df = DataFrame(slices, index=dataLabels, columns=slices_labels, )
        # print("df=", df)
        return df

    def saveSTFT(self):
        for label in self.data.index:
            savePath = os.path.join(self.path.replace('.mat', ''), "stft" + str(self.slice_length), label.replace(
                '/', '_'))

            if not os.path.exists(savePath):
                os.makedirs(savePath)
            else:
                # 删除文件夹下所有文件
                for root, dirs, files in os.walk(savePath):
                    for name in files:
                        os.remove(os.path.join(root, name))

            for column in tqdm(self.data.columns, desc="正在处理" + savePath + "的STFT图像"):
                data = self.data.loc[label, column]
                plot = FFTPlot(data, column, fs=self.fs)
                plot.saveSTFT(path=savePath)

    def saveWaveform(self):
        for label in self.data.index:
            temp = []
            savePath = os.path.join(self.path.replace('.mat', ''), "waveform" + str(self.slice_length), label.replace(
                '/', '_'))

            createFolder(savePath)

            for column in tqdm(self.data.columns, desc="正在处理" + savePath + "的波形图像"):
                data = self.data.loc[label, column]
                plot = FFTPlot(data, column, fs=self.fs)
                plot.saveWaveform(path=savePath)


class Signals(Dataset):
    def __init__(self, path, fs=5120, slice_length=512, slice_type='cut', add_noise=False):
        self.signals = [Signal(os.path.join(path, f), fs, slice_length, slice_type, add_noise) for f in os.listdir(path)
                        if
                        f.endswith('.mat')]
        self.labels = [signal.label for signal in self.signals]
        self.df = pd.concat([signal.data for signal in self.signals], axis=1)
        self.slice_type = slice_type
        self.data, self.target = self.makeDataSets()

    def makeDataSets(self):
        dic = {
            'TimeData/Motor/S_x': 0,
            'TimeData/Motor/R_y': 1,
            'TimeData/Motor/T_z': 2,
        }
        data = []
        target = []
        for key in dic:
            selected_column = self.df.loc[key]
            for j in range(len(selected_column)):
                # 将selected_column[j]转换为(1, x)的形状
                temp = selected_column[j].copy()
                temp = temp.reshape(1, len(temp))
                data.append(temp)
                target.append(int(selected_column.index[j].split('_')[0]))
        # 将data，target转换为numpy数组
        # 对齐数据, 根据最短数据进行截取
        # if self.slice_type != 'cut' and self.slice_type != 'window':
        #     min_length = min([len(data[i][0]) for i in range(len(data))])
        #     for i in range(len(data)):
        #         data[i] = data[i][:, :min_length]
        data = np.array(data)
        target = np.array(target)
        return data, target

    def saveFigure(self, type='stft'):

        for signal in self.signals:
            if type == 'stft':
                signal.saveSTFT()
            elif type == 'waveform':
                signal.saveWaveform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


class PictureData(VisionDataset):

    def __init__(self, path, shape, data_type: str, slice_length=512):
        super().__init__(root=path)
        self.shape = shape
        self.paths = []
        for root, dirs, files in os.walk(path):
            for name in dirs:
                if data_type in name and name.endswith(str(slice_length)):
                    self.paths.append(os.path.join(root, name))
        # self.merged = merged
        self.data, self.target = self.getDataSet()

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
            label = next((dic[key] for key in dic if key in path), -1)
            # 获取文件夹下所有png文件
            files = [file for file in os.listdir(path) if not file.endswith('.png')]
            # files.sort()
            png_files = [os.path.join(file, sub_file) for file in files[-3:] for
                         _, _, sub_files in os.walk(os.path.join(path, file)) for
                         sub_file in sub_files]

            for file in png_files:
                img = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
                img = processImg(self.shape, img)
                img_tensor = torch.tensor(img, dtype=float32).permute(2, 0, 1)
                data.append(img_tensor)
                target.append(label)
        return data, target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def showFigure(self, idx):
        img = tensor2img(self.data[idx])  # 将tensor转换为numpy, 格式为BRG
        cv2.imshow('img_tensor', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class GeneralFigures(VisionDataset):

    def __init__(self, path, shape, target=0):
        super().__init__(root=path)
        self.path = path
        pngs = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.png')]
        self.data, self.target = [], [target for _ in range(len(pngs))]
        for png in pngs:
            img = cv2.imread(png, cv2.IMREAD_COLOR)
            img = processImg(shape, img)
            img_tensor = torch.tensor(img, dtype=float32).permute(2, 0, 1)
            self.data.append(img_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def tensor2img(tensor):  # 将tensor转换为numpy，CHW -> HWC
    tensor = tensor.detach().to('cpu')  # Detach tensor before converting to numpy
    img = tensor.numpy()
    if img.shape[0] == 3 or img.shape[0] == 4:  # 彩色图像, 通道数为3或4, 通道顺序为RGB或RGBA
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    return img.astype(np.uint8)


def tensor2signal(tensor):  # 将tensor转换为numpy
    tensor = tensor.detach().to('cpu')  # Detach tensor before converting to numpy
    return tensor.numpy()[0]


def generate_mixed_signal_data(signals: np.ndarray, frequency_range=(10, 1000),
                               amplitude_range=(1, 10), snr_range=(-20, -5)):
    """
    生成混合信号数据集，包括正弦波形和复合波形，具有不同的频率和幅值，以及不同的信噪比。
    :param signals: 原始信号数据集
    :param frequency_range: 频率范围，以Hz为单位
    :param amplitude_range: 幅值范围
    :param snr_range: 信噪比范围，以dB为单位
    :return: 原始信号和带噪声的信号
    """
    signals = np.zeros((10000, 512), dtype=np.float16) if signals is None else signals
    num_samples, sample_length = len(signals), len(signals[0][0])
    signals.reshape(-1, sample_length)
    noisy_signals = np.zeros((num_samples, 1, sample_length), dtype=np.float16)

    for i in range(num_samples):
        signal = signals[i]
        # 计算信号功率
        signal_power = np.mean(signal ** 2)

        # 随机选择信噪比
        snr_db = np.random.uniform(*snr_range)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # 生成噪声并添加到信号上
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape).astype(np.float16)
        noisy_signal = signal + noise
        signals[i] = signal
        noisy_signals[i] = noisy_signal

    return signals, noisy_signals


def get_dataloader(path, batch_size: int, slice_length=512) -> DataLoader:
    dataset = PictureData(path, get_shape(),
                          'stft', slice_length=slice_length, merged=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_signal_dataloader(path, batch_size: int, slice_length=512, slice_type='window', add_noise=False) -> DataLoader:
    dataset = Signals(path, slice_length=slice_length, slice_type=slice_type, add_noise=add_noise)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_shape():  # 获取输入的形状
    return 1, 1, 512


def make_noise(original_signal: tensor, frequency_range=(10, 1000), amplitude_range=(1, 10),
               snr_range=(-20, -5)):
    """
    生成混合信号数据集，包括正弦波形和复合波形，具有不同的频率和幅值，以及不同的信噪比。
    :param original_signal: 原始信号数据集
    :param frequency_range: 频率范围，以Hz为单位
    :param amplitude_range: 幅值范围
    :param snr_range: 信噪比范围，以dB为单位
    :return: 原始信号和带噪声的信号
    """
    num_samples, sample_length = original_signal.shape[0], original_signal.shape[2]
    noises = []
    for i in range(num_samples):
        cur_signal = original_signal[i][0].cpu().detach().numpy()
        # cur_plot = FFTPlot(cur_signal, 'original', fs=5120)
        # cur_plot.showOriginal()
        # 计算信号功率
        signal_power = np.mean(cur_signal ** 2)

        # 随机选择信噪比
        snr_db = np.random.uniform(*snr_range)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # 生成噪声
        temp = np.random.normal(0, np.sqrt(noise_power), cur_signal.shape).astype(np.float16)
        # noise = FFTPlot(temp, 'noise', fs=5120)
        # noise.showOriginal()
        noises.append([temp])

    noises = np.array(noises)
    return tensor(noises, dtype=float32)


if __name__ == '__main__':
    print('test')
    dataSet = Signals('../data', slice_length=512, slice_type='cut')

    # noise, data = generate_mixed_signal_data(dataSet.data)
    make_noise(dataSet.data)

    # noise_dataset = Signals('../data', slice_length=512, slice_type='cut', add_noise=True)
    # noise_fft = FFTPlot(noise_dataset.data[0][0], 'noisy', fs=5120)
    # noise_fft.showOriginal()
    # print(dataSet.df)
    # data, _ = dataSet.__getitem__(0)

    # print(dataSet.df[0])
    # dataset = PictureData('../data', get_img_shape(), 32, 'stft', slice_length=5120, merged=False)
    # dataset.showFigure(0)
    # dataset.process()
    # print(dataset.paths)
    # dataset.getDataSet()
