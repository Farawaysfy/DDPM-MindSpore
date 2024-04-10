import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from model.ddpm import DDPM, build_network, convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg, \
    unet_res_cfg
from utils.dataset import get_dataloader, get_img_shape, tensor2img

batch_size = 64
n_epochs = 500


def train(ddpm: DDPM, net, device='cuda', ckpt_path='./model/model.pth',
          path='E:\\sfy\\xiaolunwen\\alg\\DDPM-MindSpore\\data', slice_length=512):
    print('batch size:', batch_size)

    writer = SummaryWriter(log_dir='./run/04102105', filename_suffix=str(n_epochs), flush_secs=5)
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(path, batch_size, slice_length)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    tic = time.time()
    i = 0
    for e in range(n_epochs):
        total_loss = 0

        for x, _ in tqdm(dataloader, desc='Epoch {}'.format(e)):
            current_batch_size = x.shape[0]
            x = x.to(device)  # Convert x to float32
            img_to_write = tensor2img(x[0])
            writer.add_image('origin', img_to_write, i, dataformats='HWC')  # tensor的形状是CHW, 对应的是channel, height, width
            t = torch.randint(0, n_steps, (current_batch_size,)).to(device)  # 生成一个0到n_steps之间的随机数
            eps = torch.randn_like(x).to(device)  # 作用是生成一个与x同样shape的随机数，服从标准正态分布
            x_t = ddpm.sample_forward(x, t, eps)  # 生成一个x_t， x_t是x的一个前向样本, 相当于给原始图片加噪声
            #  写入加噪声的图片
            x_t_img = tensor2img(x_t[0])
            writer.add_image('add_noise', x_t_img, i, dataformats='HWC')

            eps_theta = net(x_t, t.reshape(current_batch_size, 1))
            # 生成一个eps_theta, eps_theta是x_t的一个前向样本，相当于给加噪声的图片去噪声？
            #  写入处理完的图片
            eps_theta_img = tensor2img(eps_theta[0])

            writer.add_image('eps_theta', eps_theta_img, i, dataformats='HWC')

            loss = loss_fn(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
            writer.add_scalar('step loss', loss.item(), i)
            i += 1
        total_loss /= len(dataloader.dataset)
        writer.add_scalar('epochs loss', total_loss, e)
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        # print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
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

        imgs = imgs.numpy().astype(np.uint8)  # Convert tensor to numpy
        if imgs.shape[2] == 4:
            imgs = imgs[:, :, :3]  # Remove alpha channel

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

    train(ddpm, net, device=device, ckpt_path=model_path, path=data_path, slice_length=512)

    net.load_state_dict(torch.load(model_path))
    sample_imgs(ddpm, net, 'work_dirs/diffusion.png', device=device)
