import os.path

import cv2
import pywt
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt


#  fs:时间序列的采样频率,  nperseg:每个段的长度，默认为256(2^n)   noverlap:段之间重叠的点数。如果没有则noverlap=nperseg/2

# window ： 字符串或元祖或数组，可选需要使用的窗。
# #如果window是一个字符串或元组，则传递给它window是数组类型，直接以其为窗，其长度必须是nperseg。
# 常用的窗函数有boxcar，triang，hamming， hann等，默认为Hann窗。

# nfft ： int，可选。如果需要零填充FFT，则为使用FFT的长度。如果为 None，则FFT长度为nperseg。默认为无

# detrend ： str或function或False，可选
# 指定如何去除每个段的趋势。如果类型参数传递给False，则不进行去除趋势。默认为False。

# return_onesided ： bool，可选
# 如果为True，则返回实际数据的单侧频谱。如果 False返回双侧频谱。默认为 True。请注意，对于复杂数据，始终返回双侧频谱。

# boundary ： str或None，可选
# 指定输入信号是否在两端扩展，以及如何生成新值，以使第一个窗口段在第一个输入点上居中。
# 这具有当所采用的窗函数从零开始时能够重建第一输入点的益处。
# 有效选项是['even', 'odd', 'constant', 'zeros', None].
# 默认为‘zeros’,对于补零操作[1, 2, 3, 4]变成[0, 1, 2, 3, 4, 0] 当nperseg=3.

# 填充： bool，可选
# 指定输入信号在末尾是否填充零以使信号精确地拟合为整数个窗口段，以便所有信号都包含在输出中。默认为True。
# 填充发生在边界扩展之后，如果边界不是None，则填充为True，默认情况下也是如此。

# axis ： int，可选
# 计算STFT的轴; 默认值超过最后一个轴(即axis=-1)。

class FFTPlot:
    def __init__(self, data: np.ndarray, title: str, fs=5120):
        self.data = data
        self.title = title
        self.fs = fs

    def showOriginal(self, path: str = './'):
        plt.plot(self.data)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.title(self.title + '_Original')
        plt.show()
        plt.close()

    def saveOriginal(self, path: str = './'):
        plt.plot(self.data)
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.title(self.title + '_Original')
        plt.savefig(os.path.join(path, self.title + '_original.png'), pad_inches=0, bbox_inches='tight', format='png')
        plt.close()

    def saveSTFT(self, path: str = './'):
        nperseg = len(self.data)
        noverlap = int(nperseg / 4 * 3)
        f, t, nd = signal.stft(self.data, fs=self.fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None,
                               detrend=False, return_onesided=True, boundary='zeros', padded=False,
                               axis=-1, scaling='spectrum')
        plt.pcolormesh(t, f, np.abs(nd), cmap='viridis')   #vmin=0, vmax=0.5,

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(path, self.title + '_STFT.png'), pad_inches=0, bbox_inches='tight', format='png')
        plt.close()

    def showSTFT(self):
        nperseg = len(self.data)
        noverlap = int(nperseg / 4 * 3)
        f, t, nd = signal.stft(self.data, fs=self.fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None,
                               detrend=False, return_onesided=True, boundary='zeros', padded=False,
                               axis=-1, scaling='spectrum')
        plt.pcolormesh(t, f, np.abs(nd), vmin=0, vmax=0.5, cmap='viridis')
        plt.colorbar()
        plt.title(self.title + '_STFT')
        plt.ylabel('frequency')
        plt.xlabel('time')
        plt.show()
        plt.close()

    def saveWaveform(self, path: str = './'):  # 保存波形图
        sr = 128  # 1.sampling rate
        wavename = 'morl'  # 2.母小波名称
        totalscal = 150  # 3.totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
        fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
        cparam = 2 * fc * totalscal  # 常数c
        scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
        [cwtmatr, frequencies] = pywt.cwt(self.data, scales, wavename, 1.0 / sr)  # 4.y为将要进行cwt变换的一维输入信号
        t = np.arange(0, self.data.shape[0] / sr, 1.0 / sr)
        plt.contourf(t, frequencies, abs(cwtmatr))
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(path, self.title + '_waveform.png'), pad_inches=0, bbox_inches='tight', format='png')
        plt.close()

    def showWaveform(self):
        # 二维时频图
        # 1.2.3为参数，y为参数
        sr = 128  # 1.sampling rate
        wavename = 'morl'  # 2.母小波名称
        totalscal = 150  # 3.totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
        fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
        cparam = 2 * fc * totalscal  # 常数c
        scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
        [cwtmatr, frequencies] = pywt.cwt(self.data, scales, wavename, 1.0 / sr)  # 4.y为将要进行cwt变换的一维输入信号
        t = np.arange(0, self.data.shape[0] / sr, 1.0 / sr)
        plt.contourf(t, frequencies, abs(cwtmatr))
        plt.title(self.title + '_Waveform')
        plt.ylabel('frequency')
        plt.xlabel('time')
        plt.show()
        plt.close()


def processImg(shape, img):
    # cv2.imshow('img', img)
    # print("size: ", img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换颜色空间
    img = cv2.resize(img, shape[1:], interpolation=cv2.INTER_AREA)  # 作用是将图片缩放到指定大小
    # 改变图像大小
    # cv2.imshow('resize', img)
    # # 显示当前图像
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
    return img


if __name__ == '__main__':
    plot = FFTPlot(np.random.randn(512), 'test')
    plot.saveSTFT()
    # plot.showSTFT()
    # plot.showOriginal()
