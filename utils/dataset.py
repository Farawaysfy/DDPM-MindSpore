import os

import cv2
import numpy as np
import pandas as pd
import pywt
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from pykalman import KalmanFilter
from scipy.io import loadmat, savemat
from torch import float32, tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from tqdm import tqdm

from utils.fft_plot import FFTPlot, processImg


def createFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # 删除文件夹下所有文件
        import shutil

        # Delete all files and directories under the given path
        for root, dirs, files in os.walk(path):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                shutil.rmtree(os.path.join(root, name))


class Signal:
    def __init__(self, path, fs=5120, slice_length=1024, slice_type='cut', add_noise=False, windows_rate=0.5):
        self.path = path
        self.slice_length = slice_length
        self.slice_type = slice_type
        self.fs = fs
        self.add_noise = add_noise
        self.windows_rate = windows_rate
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
        self.label = ""
        for key in dic:
            if key in path:
                self.label = key
                break
        df = self.loadMat()
        self.data = df.loc[['TimeData/Motor/S_x'],]  # 选择x轴的数据, 'TimeData/Motor/R_y', 'TimeData/Motor/T_z'

    def loadMat(self):
        rawData = loadmat(self.path)
        data = np.array([[v[1].T[0]] for v in rawData['data'][0]])
        data = np.array([v[:, int(len(v[0]) * 0.2):int(len(v[0]) * 0.8)] for v in data])
        slices, slices_labels = self.slice_data(data)
        return pd.DataFrame(slices, index=[v[0][0] for v in rawData['data'][0]], columns=slices_labels)

    def slice_data(self, data):
        if self.slice_type == 'cut':
            return self.cut_data(data), [self.label + "_" + str(i) for i in range(len(data[0][0]) // self.slice_length)]
        elif self.slice_type == 'window':
            return self.window_data(data), [self.label + "_" + str(i) for i in range(len(self.window_data(data)[0]))]
        else:
            return [[v[0]] for v in data], [self.label + "_" + str(i) for i in range(len(data))]

    def cut_data(self, data):
        return [
            [v[0][i * self.slice_length:(i + 1) * self.slice_length] if not self.add_noise else
             generate_mixed_signal_data(v[0][i * self.slice_length:(i + 1) * self.slice_length].reshape(1, 1, -1),
                                        np.random.randint(-35, -25))[1].reshape(1, -1)
             for i in range(len(v[0]) // self.slice_length)] for v in data]

    def window_data(self, data):
        return [
            [v[0][left:right] if not self.add_noise else
             generate_mixed_signal_data(v[0][left:right].reshape(1, 1, -1),
                                        np.random.randint(-35, -25))[1].reshape(1, -1)
             for left in range(0, len(v[0]), int(self.slice_length * self.windows_rate)) for right in
             range(self.slice_length, len(v[0]) + 1, int(self.slice_length * self.windows_rate))] for v in data]

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
            savePath = os.path.join(self.path.replace('.mat', ''), "waveform" + str(self.slice_length), label.replace(
                '/', '_'))

            createFolder(savePath)

            for column in tqdm(self.data.columns, desc="正在处理" + savePath + "的波形图像"):
                data = self.data.loc[label, column]
                plot = FFTPlot(data, column, fs=self.fs)
                plot.saveWaveform(path=savePath)


class Signals(Dataset):
    def __init__(self, path, fs=5120, slice_length=512, slice_type='cut',
                 add_noise=False, windows_rate=0.5, delete_labels=None):
        self.signals = [Signal(os.path.join(path, f), fs, slice_length, slice_type, add_noise, windows_rate) for f in
                        os.listdir(path)
                        if
                        f.endswith('.mat')]
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
        # 去掉选定信号,'Aligned','Parallel','Unbalance'

        if delete_labels is not None:
            for label in delete_labels:
                self.signals = [signal for signal in self.signals if label != signal.label]
            # 更新 dic
            for label in delete_labels:
                dic.pop(label)
            # 对dic重新赋值
            dic = {key: i for i, key in enumerate(dic)}
        self.labels = dic
        self.df = pd.concat([signal.data for signal in self.signals], axis=1)
        self.slice_type = slice_type
        self.data, self.target = self.makeDataSets()

    def makeDataSets(self):
        dic = {
            'TimeData/Motor/S_x': 0,
            # 'TimeData/Motor/R_y': 1,
            # 'TimeData/Motor/T_z': 2,
        }
        data = []
        target = []
        for key in dic:
            selected_column = self.df.loc[key]
            for j in range(len(selected_column)):
                # 将selected_column[j]转换为(1, x)的形状
                data.append(selected_column[j].reshape(1, -1))
                # 获取标签
                target.append(self.labels[selected_column.index[j].split('_')[0]])
        if self.slice_type != 'cut' and self.slice_type != 'window':
            # 统一数据长度, 使其具有相同的长度
            min_length = min([len(v[0]) for v in data])
            data = [v[:, :min_length] for v in data]
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

    def save(self, path, file_name='signals'):
        # 保存数据到mat文件
        data = {'data': self.data.reshape(-1, get_shape()[-1]), 'target': self.target}
        savePath = os.path.join(path, file_name + '.mat')
        savemat(savePath, data)


class Signal_fft(Signals):
    def __init__(self, path, fs=5120, slice_length=512, slice_type='cut',
                 add_noise=False, windows_rate=0.5):
        super().__init__(path, fs, slice_length, slice_type, add_noise, windows_rate)
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
                # Perform FFT, take absolute value, reshape and append to data
                data.append(np.abs(np.fft.fft(selected_column[j])).reshape(1, -1))
                # 获取标签,并将标签转换为数字
                target.append(eval(selected_column.index[j].split('_')[0]))
        data = np.array(data)
        target = np.array(target)
        return data, target


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


def generate_mixed_signal_data(signals: np.ndarray, snr=-30):
    """
    生成混合信号数据集
    :param signals: 原始信号数据集
    :param snr: 信噪比, 默认为-30,需要统一信噪比
    :return: 原始信号和带噪声的信号
    """
    signals = np.zeros((10000, 512), dtype=np.float16) if signals is None else signals
    num_samples, sample_length = len(signals), len(signals[0][0])
    signals.reshape(-1, sample_length)
    noisy_signals = np.zeros((num_samples, 1, sample_length), dtype=np.float16)

    for i in range(num_samples):
        signal = signals[i]
        # 计算信号功率
        signal_power = np.mean(signal ** 2) if np.mean(signal ** 2) > 1 else 1
        # 给定信噪比，计算噪声功率
        # snr = np.random.randint(-35, -30)  # 信噪比在-20到-10之间随机取值
        snr_linear = 10 ** (snr / 10)
        noise_power = signal_power / snr_linear

        # 生成噪声并添加到信号上
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape).astype(np.float16)
        signals[i] = signal
        noisy_signals[i] = signal + noise

    return signals, noisy_signals


def make_noise(t: tensor) -> tensor:
    """
    生成高斯白噪声
    :param t:当前步
    :return:噪声的信号
    """
    t = t.detach().cpu().numpy()
    snr = -10 - np.power(t, 0.5)  # 计算信噪比, 该公式应当根据实际情况调整
    noises = []
    snr_linears = 10 ** (snr / 10)
    if isinstance(snr_linears, np.ndarray):
        for snr_linear in snr_linears:
            noise_power = 1 / snr_linear
            noises.append(np.random.randn(get_shape()[-1]) * np.sqrt(noise_power))
    else:
        noise_power = 1 / snr_linears
        noises.append(np.random.randn(get_shape()[-1]) * np.sqrt(noise_power))
    noises = np.array(noises)
    noises = noises.reshape(-1, 1, len(noises[0]))
    return tensor(noises, dtype=float32)


def get_dataloader(path, batch_size: int, slice_length=512) -> DataLoader:
    dataset = PictureData(path, get_shape(),
                          'stft', slice_length=slice_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_signal_dataloader(path, batch_size: int, slice_length=512, slice_type='window',
                          add_noise=False, window_ratio=0.5) -> DataLoader:
    dataset = Signals(path, slice_length=slice_length, slice_type=slice_type,
                      add_noise=add_noise, windows_rate=window_ratio)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_shape():  # 获取输入的形状
    return 1, 1, 512


def WD_signal(signal: np.ndarray) -> np.ndarray:
    """
    处理信号，默认使用小波去噪
    :param signal: 输入信号
    :return: 处理后的信号
    """
    signal = signal.reshape(-1, signal.shape[2])
    for i in range(len(signal)):
        coeffs = pywt.wavedec(signal[i], 'db1', level=1)
        # 将小波系数的绝对值进行阈值处理
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal[i])))
        denoised_coeffs = [coeffs[0]]
        for j in range(1, len(coeffs)):
            denoised_coeffs.append(pywt.threshold(coeffs[j], uthresh, mode='soft'))
        signal[i] = pywt.waverec(denoised_coeffs, 'db1')
    return signal.reshape(-1, 1, signal.shape[1])


class KalmanFilterSignal:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x


def KM_signal(signal: np.ndarray) -> np.ndarray:
    """
    使用卡尔曼滤波器对信号进行处理
    :param signal: 输入信号
    :return: 处理后的信号
    """
    signal = signal.reshape(-1, signal.shape[2])
    F = np.array([[1]])
    H = np.array([[1]])
    Q = np.array([[0.0001]])
    R = np.array([[0.001]])
    x0 = np.array([[0]])
    P0 = np.array([[1]])
    kf = KalmanFilterSignal(F, H, Q, R, x0, P0)
    for i in range(len(signal)):
        kf.predict()
        signal[i] = kf.update(signal[i])
    return signal.reshape(-1, 1, signal.shape[1])


if __name__ == '__main__':
    print('test')
    dataSet = Signals('../data', slice_type='origin')
    data, noise = generate_mixed_signal_data(dataSet.data)
    save_path = '../work_dirs/noisy'
    kf_signal = KM_signal(noise.copy())  # 使用卡尔曼滤波器处理信号
    wd_signal = WD_signal(noise.copy())  # 使用小波去噪处理信号
    # 绘制原始信号和带噪声的信号
    for i in tqdm(range(len(data)), desc='正在绘制信号图像'):
        fig, ax = plt.subplots(4, 1, figsize=(320, 48))
        ax[0].plot(data[i][0], label='Original Signal', linewidth=1)
        ax[0].set_title('Original Signal')
        ax[1].plot(noise[i][0], label='Noisy Signal', linestyle='--', linewidth=1)
        ax[1].set_title('Noisy Signal')
        ax[2].plot(kf_signal[i][0], label='Kalman Filter Signal', linestyle='--', linewidth=1)
        ax[2].set_title('Kalman Filter Signal')
        ax[3].plot(wd_signal[i][0], label='Wavelet Denoising Signal', linestyle='--', linewidth=1)
        ax[3].set_title('Wavelet Denoising Signal')
        plt.savefig(os.path.join(save_path, str(i) + '.png'))
        plt.close()
