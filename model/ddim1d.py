import torch

from model.ddim import DDIM


class DDIM1D(DDIM):

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        super().__init__(device, n_steps, min_beta, max_beta)

    def sample_forward(self, x, t, eps=None):  # x是一维信号， eps是噪声, t是随机数，此步是加噪声
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x

        return res
