import torch


def compute_fft(signal):
    # 计算信号的FFT
    # signal应该是一个二维的tensor，其中每一行代表一个数据序列
    fft_result = torch.fft.fft(signal)
    return fft_result


class FFTLoss(torch.nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, signal1, signal2):
        fft1, fft2 = compute_fft(signal1), compute_fft(signal2)
        # 计算FFT结果之间的损失
        loss_real = self.mse_loss(fft1.real, fft2.real)
        loss_imag = self.mse_loss(fft1.imag, fft2.imag)
        # 这里使用mse loss
        return (loss_real + loss_imag) / 2


class CombinedLoss(torch.nn.Module):
    def __init__(self, weight_huber=0.5):
        super(CombinedLoss, self).__init__()
        self.weight_huber = weight_huber if 1 >= weight_huber >= 0 else 0.5
        # huber loss的权重
        self.huber_loss = torch.nn.HuberLoss()
        self.fft_loss = FFTLoss()

    def forward(self, signal_output, signal_target):
        # 计算huber loss
        loss_huber = self.huber_loss(signal_output, signal_target)
        # 计算fft loss
        loss_fft = self.fft_loss(signal_output, signal_target)
        total_loss = self.weight_huber * loss_huber + (1 - self.weight_huber) * loss_fft
        return total_loss
