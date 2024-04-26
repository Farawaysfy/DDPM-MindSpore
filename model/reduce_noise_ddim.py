import torch
from torch import tensor
from tqdm import tqdm

from model.ddim import DDIM


class Reduce_noise(DDIM):

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        super().__init__(device, n_steps, min_beta, max_beta)

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
        for i in tqdm(range(1, ddim_step + 1),
                      f'DDIM sampling with eta {eta} simple_var {simple_var}'):
            cur_t = ts[i - 1] - 1

            t_tensor = torch.tensor([cur_t] * batch_size,
                                    dtype=torch.long).to(device).unsqueeze(1)
            eps = net(x, t_tensor)

            x -= eps
        return x
