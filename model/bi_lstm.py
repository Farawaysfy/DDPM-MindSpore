# coding:utf8
from utils.dataset import get_shape


class BI_LSTM_Config:
    def __init__(self):
        # 训练配置
        # self.seed = 22
        # self.batch_size = 64
        # self.lr = 1e-3
        # self.weight_decay = 1e-4
        # self.num_epochs = 100
        # self.early_stop = 512
        # self.max_seq_length = 128
        # self.save_path = '../model_parameters/BiLSTM_SA.bin'

        # 模型配置
        self.lstm_hidden_size = 128
        self.dense_hidden_size = 128
        self.num_layers = 1
        self.num_outputs = get_shape()[2]
