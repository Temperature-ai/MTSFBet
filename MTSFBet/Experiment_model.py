import torch
from torch.nn import Module, CosineSimilarity
from MTSFBet.layers import BiGRU, FPN, Task_1_output_end, Task_2_output_end
from torchinfo import summary

class Feature_extractor(Module):
    def __init__(self ,kernel_size, filter_size):
        super(Feature_extractor, self).__init__()
        self.BiGRU = BiGRU(kernel_size, filter_size)
        self.FPN   = FPN(kernel_size, filter_size)

    def forward(self, x):
        x_1 = self.BiGRU(x)
        x_2 = self.FPN(x)
        feature = torch.cat([x_1, x_2], dim=1)
        return feature

class MTSFBet_single_branch(Module):
    def __init__(self, kernel_size, filter_size, num_classes):
        super(MTSFBet_single_branch, self).__init__()
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        self.num_classes       = num_classes
        self.Feature_extractor = Feature_extractor(kernel_size, filter_size)
        self.Task_1            = Task_1_output_end(kernel_size, filter_size, num_classes)
        self.Task_2            = Task_2_output_end(kernel_size, filter_size)

    def forward(self, x):
        x = self.Feature_extractor(x)
        Task_2 = self.Task_2(x)
        return x, Task_2

class MTSFBet(Module):
    def __init__(self, kernel_size, filter_size, num_classes):
        super(MTSFBet, self).__init__()
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        self.num_classes = num_classes
        self.Branch = MTSFBet_single_branch(kernel_size, filter_size, num_classes)
        self.cos_sim = CosineSimilarity()

    def forward_once(self, x):
        x, Task_2 = self.Branch(x)
        return x, Task_2

    def forward(self, input_1, input_2):
        x_1, Task_2_1 = self.forward_once(input_1)
        x_2, Task_2_2 = self.forward_once(input_2)
        Task_1 = self.cos_sim(x_1, x_2)
        Task_2 = self.cos_sim(Task_2_1, Task_2_2)
        return Task_1, Task_2

if __name__ == '__main__':
    net = MTSFBet(34)
    print(net)