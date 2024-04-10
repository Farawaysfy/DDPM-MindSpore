import cv2
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

class MyPlot:
    def __init__(self, data: np.ndarray, title: str, fs=5120):
        self.title = title
        nperseg = len(data)
        noverlap = int(nperseg / 4 * 3)
        self.f, self.t, self.nd = signal.stft(data, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None,
                                              detrend=False, return_onesided=True, boundary='zeros', padded=False,
                                              axis=-1, scaling='spectrum')

    def saveSTFT(self, path: str = './'):
        plt.figure(figsize=(10, 10))  # 增加图像大小以提高清晰度
        plt.pcolormesh(self.t, self.f, np.abs(self.nd), vmin=0, vmax=0.5, cmap='viridis')
        # plt.colorbar()
        # plt.title(self.title + '_STFT')
        # plt.ylabel('frequency')
        # plt.xlabel('time')
        # plt.yscale('log')  # 使用对数刻度来增强低频特征的可视化
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(path + self.title + '_stft.png', pad_inches=0, bbox_inches='tight', format='png')
        plt.close()
        return path + self.title + '_stft.png'
        # plt.show()

    def showSTFT(self):
        plt.figure(figsize=(10, 6))  # 增加图像大小以提高清晰度
        plt.pcolormesh(self.t, self.f, np.abs(self.nd), vmin=0, vmax=0.5, cmap='viridis')
        plt.colorbar()
        plt.title(self.title + '_STFT')
        plt.ylabel('frequency')
        plt.xlabel('time')
        plt.yscale('log')  # 使用对数刻度来增强低频特征的可视化
        plt.show()
        plt.close()

    def saveWaveform(self, path: str = './'):  # 保存波形图
        plt.plot(self.t, self.f)
        # plt.title(self.title + '_Waveform')
        # plt.ylabel('frequency')
        # plt.xlabel('time')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(path + self.title + '_waveform.png', pad_inches=0, bbox_inches='tight', format='png')
        plt.close()

    def showWaveform(self):
        plt.plot(self.t, self.f)
        plt.title(self.title + '_Waveform')
        plt.ylabel('frequency')
        plt.xlabel('time')
        plt.tight_layout()
        plt.show()


def processImg(shape, img):
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # print("size: ", img.shape)

    # if img.shape[2] == 3 or img.shape[2] == 4:  # 彩色图像
    #     # 将img由三通道转换为单通道
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if img.shape[1:] != shape[1:]:
        img = cv2.resize(img, shape[1:])
    # 改变图像大小
    # cv2.imshow('resize', img)
    # # 显示当前图像
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
    return img


def recoverImg(shape, img):
    img = 255 - img
    # cv2.imshow('dst', img)
    # cv2.waitKey(0)
    pass


if __name__ == '__main__':
    plot = MyPlot(np.random.randn(5120), 'test')
    # plot.saveSTFT()
    plot.showSTFT()
