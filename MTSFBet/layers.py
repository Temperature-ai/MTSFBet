# --------------------------------------------------------------------- #
# Multi-Task Siamese FCN-BiGRU
# 多任务孪生神经网络，任务1：手势分类，任务2：身份认证
# 相关组件
# --------------------------------------------------------------------- #
import torch
from torch.nn.functional import pad
from torch.nn import Dropout, Linear, MaxPool1d, Conv1d, BatchNorm1d, GRU, \
     AdaptiveAvgPool1d, Flatten, Module, Sequential, ReLU, AvgPool1d, Tanh, Softmax

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BiGRU(Module):
    def __init__(self, kernel_size, filter_size):
        super(BiGRU, self).__init__()
        self.BiGRU   = GRU(6, int(16*kernel_size), 1, bidirectional=True)
        self.tanh    = Tanh()
        self.dropout = Dropout(0.5)
        self.gloalavgpooling1d = AdaptiveAvgPool1d(1)

    def forward(self, x):
        output, hidden = self.BiGRU(x)
        x = self.tanh(output)
        x = self.dropout(x)
        x = x.permute(0,2,1)
        x = self.gloalavgpooling1d(x)
        x = torch.squeeze(x, dim=2)
        return x

class FCN(Module):
    def __init__(self, input_size, output_size, num_kernels, strides):
        super(FCN, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.num_kernels = num_kernels
        self.strides     = strides
        self.FCN_block_no_str = Sequential(
            Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=num_kernels, padding='same'),
            BatchNorm1d(output_size),
            ReLU()
        )
        self.FCN_block_str = Sequential(
            Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=num_kernels, stride=strides, padding=int((num_kernels*1)/2)),
            BatchNorm1d(output_size),
            ReLU()
        )

    def forward(self, x):
        if self.strides>=2 :
            x = self.FCN_block_str(x)
        else:
            x = self.FCN_block_no_str(x)
        return x

class FPN(Module):
    def __init__(self,kernel_size, filter_size):
        super(FPN, self).__init__()
        self.FCN_block_1 = Sequential(
            FCN(6,  int(32*kernel_size),  int(9*filter_size), 1),
            FCN(int(32*kernel_size), int(64*kernel_size),  int(9*filter_size), 1),
            FCN(int(64*kernel_size), int(128*kernel_size), int(9*filter_size), 1),
        )

        self.FCN_block_2 = Sequential(
            FCN(int(128*kernel_size), int(64*kernel_size), int(7*filter_size), 2),
            FCN(int(64*kernel_size),  int(32*kernel_size), int(7*filter_size), 2),
            MaxPool1d(kernel_size=2, stride=2),
        )

        self.FCN_block_3 = Sequential(
            FCN(int(128*kernel_size), int(64*kernel_size), int(7*filter_size), 2),
            FCN(int(64*kernel_size),  int(32*kernel_size), int(5*filter_size), 2),
        )
        self.maxpool1d = MaxPool1d(kernel_size=2, stride=2)

        self.FCN_block_4 = Sequential(
            FCN(int(128*kernel_size), int(128*kernel_size), int(7*filter_size), 1),
            FCN(int(128*kernel_size), int(64*kernel_size),  int(5*filter_size), 1),
            FCN(int(64*kernel_size),  int(32*kernel_size),  int(3*filter_size), 2)
        )

        self.FCN_block_5 = Sequential(
            FCN(int(128*kernel_size), int(128*kernel_size), int(5*filter_size), 1),
            FCN(int(128*kernel_size), int(64*kernel_size),  int(3*filter_size), 1),
            FCN(int(64*kernel_size),  int(32*kernel_size),  int(1*filter_size), 1)
        )

        self.GlobalAveragePooling1D = AdaptiveAvgPool1d(1)

    def forward(self, x):
        x   = x.permute(0,2,1)
        x   = self.FCN_block_1(x)
        x_0 = self.FCN_block_2(x)

        x_1 = self.maxpool1d(x)
        x_1_x = self.FCN_block_3(x_1)

        x_2 = self.maxpool1d(x_1)
        x_2_x = self.FCN_block_4(x_2)

        x_3 = self.maxpool1d(x_2)
        x_3_x = self.FCN_block_5(x_3)

        x_2_x = x_3_x + x_2_x
        x_1_x = x_1_x + x_2_x
        x     = x_1_x + x_0

        x     = self.GlobalAveragePooling1D(x)
        x     = torch.squeeze(x, dim=2)

        return x

class Task_1_output_end(Module):
    def __init__(self, kernel_size, filter_size, num_classes):
        super(Task_1_output_end, self).__init__()
        self.num_calsses = num_classes
        self.Task_block = Sequential(
            Linear(int(64*kernel_size), int(256*kernel_size)),
            ReLU(),
            Linear(int(256*kernel_size), num_classes)
        )

    def forward(self, x):
        x = self.Task_block(x)
        return x

class Task_2_output_end(Module):
    def __init__(self, kernel_size, filter_size):
        super(Task_2_output_end, self).__init__()
        self.Task_block_conv = Sequential(
            Conv1d(in_channels=1,  out_channels=int(16*kernel_size), kernel_size=int(3*filter_size), stride=1, padding='valid'),
            Conv1d(in_channels=int(16*kernel_size), out_channels=int(16*kernel_size), kernel_size=int(3*filter_size), stride=1, padding='valid'),
            Conv1d(in_channels=int(16*kernel_size), out_channels=int(8*kernel_size),  kernel_size=int(1*filter_size), stride=1, padding='valid'),
            BatchNorm1d(int(8*kernel_size)),
            ReLU(),
            AvgPool1d(kernel_size=2, stride=2))
        self.GRU_block = GRU(int(8*kernel_size), int(16*kernel_size), 1, batch_first=True, bidirectional=True)
        self.Task= Sequential(
            Tanh(),
            Linear(int(32*kernel_size), int(128*kernel_size)),
            ReLU()
        )
        self.gloalavgpooling1d = AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=2)
        x = x.permute(0, 2, 1)
        x = self.Task_block_conv(x)
        x = x.permute(0, 2, 1)
        x = self.GRU_block(x)
        x = self.Task(x[0])
        x = x.permute(0, 2, 1)
        x = self.gloalavgpooling1d(x)
        x = torch.squeeze(x, dim=2)
        return x