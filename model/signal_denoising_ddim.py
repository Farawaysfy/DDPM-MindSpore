import torch
from torch import tensor
from tqdm import tqdm

from model.ddim import DDIM
from utils.dataset import make_noise


class Signal_denoising(DDIM):

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        super().__init__(device, n_steps, min_beta, max_beta)

    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1)
        if eps is None:
            eps = make_noise(x) + x
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res

    def sample_backward(self,
                        original_signal,
                        net,
                        device,
                        simple_var=True,
                        ddim_step=20,
                        eta=1):
        if simple_var:
            eta = 1
        ts = torch.linspace(self.n_steps, 0,
                            (ddim_step + 1)).to(device).to(torch.long)
        x = original_signal  # 输入原始带噪声信号，开始去噪
        batch_size = len(x)
        net = net.to(device)

        x = tensor(x).to(device)
        for i in range(1, ddim_step + 1):
            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1
            ab_cur = self.alpha_bars[cur_t]
            ab_prev = self.alpha_bars[prev_t] if prev_t >= 0 else 1
            t_tensor = torch.tensor([cur_t] * batch_size,
                                    dtype=torch.long).to(device).unsqueeze(1)
            eps = net(x, t_tensor)
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = make_noise(x).to(device)
            first_term = (ab_prev / ab_cur) ** 0.5 * x
            second_term = ((1 - ab_prev - var) ** 0.5 -
                           (ab_prev * (1 - ab_cur) / ab_cur) ** 0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev) ** 0.5 * noise
            else:
                third_term = var ** 0.5 * noise
            x = first_term + second_term + third_term
            # 将x缩放到-1到1之间,缩放!不是裁剪, 防止x的值过大或过小
            while x.max() > 1 or x.min() < -1:
                x = (x - x.min()) / (x.max() - x.min()) * 2 - 1
        return x
