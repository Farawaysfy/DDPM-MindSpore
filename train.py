import os

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.ddim import DDIM
from model.ddpm import DDPM, build_network, convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg, \
    unet_res_cfg, convnet1d_big_cfg, convnet1d_medium_cfg, convnet1d_small_cfg
from model.vit import VisionTransformer
from utils.FFTPlot import FFTPlot
from utils.dataset import get_shape, get_signal_dataloader, tensor2signal, createFolder, PictureData, GeneralFigures, \
    tensor2img

batch_size = 512
n_epochs = 50
_exp_name = "sample"


def config_read(config):
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    device = config['device']
    learning_rate = config['lr']
    wd = config['weight_decay']
    seed = config['seed']
    shape = config['shape']
    num_classes = config['num_classes']
    return batch_size, n_epochs, device, learning_rate, wd, seed, shape, num_classes


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
    batch_size, n_epochs, device, learning_rate, wd, seed, shape, num_classes = config_read(config)
    same_seeds(seed)
    in_c, img_size, _ = shape
    model = VisionTransformer(img_size=img_size, in_c=in_c, num_classes=num_classes).to(device)
    stale = 0
    best_acc = 0
    patience = 300

    data, labels = [], []
    for d, l in dataset:
        data.append(d)
        labels.append(l)
    data = torch.stack(data)
    labels = torch.tensor(labels)
    writer = SummaryWriter(log_dir='run/04191518stft512_300', filename_suffix=str(n_epochs), flush_secs=5)
    train_data, valid_data, train_labels, valid_labels = train_test_split(data, labels, test_size=0.2)  # split dataset
    train_dataset = TensorDataset(train_data, train_labels)
    valid_dataset = TensorDataset(valid_data, valid_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}, training"):
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
        # print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{n_epochs}, validation"):
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

        # add to tensorboard
        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('train_acc', train_acc, epoch)
        # writer.add_scalar('valid_loss', valid_loss, epoch)
        # writer.add_scalar('valid_acc', valid_acc, epoch)
        writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, epoch)
        writer.add_scalars('acc', {'train': train_acc, 'valid': valid_acc}, epoch)
        # Print the information.
        # print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

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
        "batch_size": 64,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # "device": "cpu",
        "n_epochs": 300,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "seed": 66,
        "shape": (3, 256, 256),
        "num_classes": 8
    } if config is None else config
    train_model(config, dataset)


def plot_signal(signal, title, subplot_position):
    plt.subplot(subplot_position[0], subplot_position[1], subplot_position[2])
    plt.plot(tensor2signal(signal[0]))
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('amplitude')


def train(ddpm: DDPM, net, device='cuda', ckpt_path='./model/model.pth',
          path='./data', slice_length=512):
    writer = SummaryWriter(log_dir='./run/04231713', filename_suffix=str(n_epochs), flush_secs=5)
    n_steps = ddpm.n_steps
    # dataloader = get_dataloader(path, batch_size, slice_length)
    dataloader = get_signal_dataloader(path, batch_size, slice_length)
    net = net.to(device).double()  # 将网络放到GPU上, 并且将网络的参数类型设置为double
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adamax(net.parameters(), 1e-3)

    i = 0
    for e in range(n_epochs):
        total_loss = 0

        for x, _ in tqdm(dataloader, desc='Epoch {}'.format(e)):
            current_batch_size = x.shape[0]
            x = x.to(device)  # Convert x to float32
            t = torch.randint(0, n_steps, (current_batch_size,)).to(device)  # 生成一个0到n_steps之间的随机数
            eps = torch.randn_like(x).to(device)  # 作用是生成一个与x同样shape的随机数，服从标准正态分布
            x_t = ddpm.sample_forward1D(x, t, eps)  # 生成一个x_t， x_t是x的一个前向样本, 相当于给原始输入加噪声

            eps_theta = net(x_t, t.reshape(current_batch_size, 1))

            loss = loss_fn(eps_theta, eps)  # 计算eps_theta和eps的损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size

            # 信号diffusion的过程
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            fig.tight_layout(h_pad=5, w_pad=5)

            plot_signal(x, 'x', (2, 2, 1))
            plot_signal(x_t, 'x_t', (2, 2, 2))
            plot_signal(eps, 'eps', (2, 2, 3))
            plot_signal(eps_theta, 'eps_theta', (2, 2, 4))
            writer.add_figure('signal process', plt.gcf(), i)
            writer.add_scalar('step loss', loss.item(), i)
            i += 1

        total_loss /= len(dataloader.dataset)
        writer.add_scalar('epochs loss', total_loss, e)
        torch.save(net.state_dict(), ckpt_path)
    print('Done')


configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg, convnet1d_big_cfg, convnet1d_medium_cfg, convnet1d_small_cfg
]


def sample_imgs(ddpm, net, output_path, n_sample=81, device='cuda', simple_var=True):
    net = net.to(device).eval()
    with torch.no_grad():
        shape = (n_sample, *get_shape())
        imgs = ddpm.sample_backward(shape, net, device=device, simple_var=simple_var).detach().cpu()
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        imgs = einops.rearrange(imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_sample ** 0.5))
        imgs = tensor2img(imgs)
        cv2.imwrite(output_path, imgs)


def sample_signals(ddpm, net, n_sample=81, device='cuda', simple_var=True):
    net = net.to(device).eval()
    with torch.no_grad():
        shape = (n_sample, get_shape()[1], get_shape()[2])

        signals = ddpm.sample_backward(shape, net, device=device,
                                       simple_var=simple_var)  # 生成信号, /ddpm.n_steps没有道理
        signals = signals.detach().cpu().numpy()
        # 当signals的范围超过-1到1之间时，对signals的绝对值进行对数运算， 使得范围缩小
        # 记录信号的正负
        while signals.max() > 10 or signals.min() < -10:
            signals = np.sign(signals) * np.log10(np.abs(signals))
        while signals.max() > 1 or signals.min() < -1:
            signals = signals / 10
        # signals = signals.clip(-1, 1)
        # 将信号的形状从(n_sample, 1, n_steps)转换为(n_sample, n_steps)
        # signals = signals.reshape(n_sample, -1)
        # 每条信号生成一张图片，将多张图片拼接成一张大图
        createFolder('work_dirs/original')
        createFolder('work_dirs/stft')
        createFolder('work_dirs/wf')
        for i in tqdm(range(n_sample), desc='sample signals'):
            fft = FFTPlot(signals[i][0], 'signal {}'.format(i + 1))
            fft.saveOriginal('work_dirs/original')
            fft.saveSTFT('work_dirs/stft')
            fft.saveWaveform('work_dirs/wf')


def vit_predict(model_path, data, device='cuda'):
    net = VisionTransformer(img_size=256, in_c=3, num_classes=8).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    res = []
    with torch.no_grad():
        for d in tqdm(data, desc=f'predicting {model_path}'):
            data = d.unsqueeze(0).to(device)
            logits = net(data)
            res.append(logits.argmax(dim=-1).item())
    return res


def predict():
    classify_model_path = './model/stft_512_vit_sample_best300_64batch.ckpt'
    stft_data = GeneralFigures('./work_dirs/stft', (3, 256, 256))
    stft_res = vit_predict(classify_model_path, stft_data.data, device=device)
    wf_data = GeneralFigures('./work_dirs/wf', (3, 256, 256))
    classify_model_path = './model/wf_512_vit_sample_best300_64batch.ckpt'
    wf_res = vit_predict(classify_model_path, wf_data.data, device=device)
    # 预测矩阵
    stft_res = np.array(stft_res).reshape(1000, 1)
    wf_res = np.array(wf_res).reshape(1000, 1)
    compare = stft_res == wf_res
    res = np.concatenate((stft_res, wf_res, compare), axis=1)
    # 统计预测结果
    same = np.sum(compare)
    percent = same / 1000
    # 将结果写入csv
    np.savetxt('work_dirs/result.csv', res, delimiter=',', fmt='%d')
    print(f'预测结果：{same}个相同，占比{percent}')


if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)
    n_steps = 1000
    config_id = 4
    device = 'cuda'
    model_path = './model/model_1d_cnn_signal_big_steps10000.pth'
    data_path = './data'

    config = configs[config_id]
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)

    train(ddpm, net, device=device, ckpt_path=model_path, path=data_path, slice_length=512)
    ddim = DDIM(device, n_steps)
    net.load_state_dict(torch.load(model_path))
    sample_signals(ddim, net, n_sample=1000, device=device)
    # sample_imgs(ddim, net, 'work_dirs/diffusion.png', n_sample=81, device=device)
    # dataset = PictureData('./data', get_shape(),
    #                       'stft', slice_length=512)
    # dataset.showFigure(10)
    # vit_train(dataset)
