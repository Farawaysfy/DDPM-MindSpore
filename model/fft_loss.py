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
        # 这里使用huber loss
        loss = torch.nn.functional.huber_loss(fft1, fft2)
        return loss


class CombinedLoss(torch.nn.Module):
    def __init__(self, weight_huber=1.0):
        super(CombinedLoss, self).__init__()
        self.weight_huber = weight_huber
        self.huber_loss = torch.nn.HuberLoss()
        self.fft_loss = FFTLoss()

    def forward(self, signal_output, signal_target):
        # 计算huber loss
        loss_huber = self.huber_loss(signal_output, signal_target)
        # 计算fft loss
        loss_fft = self.fft_loss(signal_output, signal_target)
        total_loss = self.weight_huber * loss_huber + (1 - self.weight_huber) * loss_fft
        return total_loss
