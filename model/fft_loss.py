import torch


def compute_fft(signal):
    # 计算信号的FFT
    # signal应该是一个二维的tensor，其中每一行代表一个数据序列
    fft_result = torch.fft.fft(signal)
    return fft_result


class FFTLoss(torch.nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, signal1, signal2):
        fft1, fft2 = compute_fft(signal1), compute_fft(signal2)
        # 计算FFT结果之间的损失
        # 这里使用均方误差
        loss = torch.mean((fft1 - fft2).abs() ** 2)
        return loss
