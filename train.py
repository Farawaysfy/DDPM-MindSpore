import os
import time

import torch
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm

from model.ddpm import DDPM, build_network, convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg, \
    unet_res_cfg
import cv2
import numpy as np
import einops

from utils.dataset import get_dataloader, get_img_shape
from torch.utils.tensorboard import SummaryWriter

batch_size = 64
n_epochs = 500


def train(ddpm: DDPM, net, device='cuda', ckpt_path='./model/model.pth',
          path='E:\\sfy\\xiaolunwen\\alg\\DDPM-MindSpore\\data'):
    print('batch size:', batch_size)

    writer = SummaryWriter(log_dir='./run', filename_suffix=str(n_epochs), flush_secs=5)
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(path, batch_size, 512)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0

        for x, _ in tqdm(dataloader, desc='Epoch {}'.format(e)):
            current_batch_size = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size,)).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))
            loss = loss_fn(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
            writer.add_scalar('loss', loss.item(), e)
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]


def sample_imgs(ddpm,
                net,
                output_path,
                n_sample=81,
                device='cuda',
                simple_var=True):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())  # 1, 3, 28, 28
        imgs = ddpm.sample_backward(shape,
                                    net,
                                    device=device,
                                    simple_var=simple_var).detach().cpu()
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        imgs = einops.rearrange(imgs,
                                '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_sample ** 0.5))

        imgs = imgs.numpy().astype(np.uint8)

        cv2.imwrite(output_path, imgs)


if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)
    n_steps = 1000
    config_id = 4
    device = 'cuda'
    model_path = './model/model_unet_res.pth'
    data_path = './data'

    config = configs[config_id]
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)

    train(ddpm, net, device=device, ckpt_path=model_path, path=data_path)

    net.load_state_dict(torch.load(model_path))
    sample_imgs(ddpm, net, 'work_dirs/diffusion.png', device=device)
