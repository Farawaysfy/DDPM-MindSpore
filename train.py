import os

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.ddpm import DDPM, build_network, convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg, \
    unet_res_cfg
from model.vit import VisionTransformer
from utils.dataset import get_dataloader, get_img_shape, tensor2img

batch_size = 64
n_epochs = 500
_exp_name = "sample"


def config_read(config):
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    device = config['device']
    learning_rate = config['lr']
    wd = config['weight_decay']
    seed = config['seed']
    shape = config['shape']
    return batch_size, n_epochs, device, learning_rate, wd, seed, shape


# fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_model(config, dataset):
    batch_size, n_epochs, device, learning_rate, wd, seed, shape = config_read(config)
    same_seeds(seed)
    in_c, img_size, _ = shape
    model = VisionTransformer(img_size=img_size, in_c=in_c).to(device)
    stale = 0
    best_acc = 0
    patience = 300

    train_loader, valid_loader = train_test_split(dataset, test_size=0.2)  # split dataset

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # imgs = imgs.half()
            # print(imgs.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            # break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # update logs
        if valid_acc > best_acc:
            with open(f"./{_exp_name}_log.txt", "a"):
                print(
                    f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name}_log.txt", "a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(),
                       f"{_exp_name}_best.ckpt")  # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break


def vit_train(dataset, config=None):
    # X_train, X_valid, y_train, y_valid= train_test_split(all_data_set)
    #  hyper parameters
    config = {
        "batch_size": 16,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # "device": "cpu",
        "n_epochs": 100,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "seed": 66
    } if config is None else config
    train_model(config, dataset)


def train(ddpm: DDPM, net, device='cuda', ckpt_path='./model/model.pth',
          path='E:\\sfy\\xiaolunwen\\alg\\DDPM-MindSpore\\data', slice_length=512):
    print('batch size:', batch_size)

    writer = SummaryWriter(log_dir='./run/04151225', filename_suffix=str(n_epochs), flush_secs=5)
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(path, batch_size, slice_length)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

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

            eps_img = tensor2img(eps[0])
            writer.add_image('eps', eps_img, i, dataformats='HWC')
            x_t = ddpm.sample_forward(x, t, eps)  # 生成一个x_t， x_t是x的一个前向样本, 相当于给原始图片加噪声
            #  写入加噪声的图片
            x_t_img = tensor2img(x_t[0])
            writer.add_image('add_noise', x_t_img, i, dataformats='HWC')

            eps_theta = net(x_t, t.reshape(current_batch_size, 1))
            # 生成一个eps_theta, eps_theta是x_t的一个前向样本,预测噪声
            #  写入处理完的图片
            eps_theta_img = tensor2img(eps_theta[0])

            writer.add_image('eps_theta', eps_theta_img, i, dataformats='HWC')

            loss = loss_fn(eps_theta, eps)  # 计算eps_theta和eps的损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
            writer.add_scalar('step loss', loss.item(), i)
            i += 1
        total_loss /= len(dataloader.dataset)
        writer.add_scalar('epochs loss', total_loss, e)
        torch.save(net.state_dict(), ckpt_path)
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
                                '(b1 b2) c h w -> (b1 h) (b2 w) c',  # 将图片拼接成一个大图, chw -> hwc
                                b1=int(n_sample ** 0.5))

        imgs = imgs.numpy().astype(np.uint8)  # Convert tensor to numpy
        # imgs = np.transpose(imgs, (1, 2, 0))
        # if imgs.shape[2] == 4:
        #     imgs = imgs[:, :, :3]  # Remove alpha channel

        # imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)    # RGB是hwc的顺序，而cv2是chw的顺序
        imgs = cv2.cvtColor(imgs, cv2.COLOR_GRAY2BGR)  # gray是hwc的顺序，而cv2是chw的顺序

        cv2.imwrite(output_path, imgs)


if __name__ == '__main__':
    # os.makedirs('work_dirs', exist_ok=True)
    # n_steps = 1000
    # config_id = 4
    # device = 'cuda'
    # model_path = './model/model_unet_res.pth'
    # data_path = './data'
    #
    # config = configs[config_id]
    # net = build_network(config, n_steps)
    # ddpm = DDPM(device, n_steps)
    #
    # train(ddpm, net, device=device, ckpt_path=model_path, path=data_path, slice_length=512)
    #
    # net.load_state_dict(torch.load(model_path))
    # sample_imgs(ddpm, net, 'work_dirs/diffusion.png', n_sample=1, device=device)
    dataset = get_dataloader('./data', batch_size, 512)
    vit_train(dataset)
