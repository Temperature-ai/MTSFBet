import torch
from torch import log, pow, mean
from torch.nn import Module, CrossEntropyLoss

class Loss(Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, x, y):
        y_lable         = y[:, 0]
        y_pred_task_1   = x[0]
        y_pred_task_1_2 = x[1]
        loss_1          = self.cross_entropy(y_pred_task_1, y_lable) + self.cross_entropy(y_pred_task_1_2, y_lable)

        y_Identity    = y[:, 1]
        y_pred_task_2 = x[2]
        y_pred_task_2 = torch.clamp(y_pred_task_2, 1e-7, 1.0-1e-7)
        loss_2        = y_Identity * (-log(pow(y_pred_task_2, 1.5))) + (1.0 - y_Identity) * (-log(pow(1.0 - y_pred_task_2, 1.5)))

        return mean(loss_1 + loss_2)

    # 自定义softmax运算
    def my_softmax(self,X, dim):
        X_exp = X.exp()
        partition = X_exp.sum(dim=dim, keepdim=True)
        return X_exp / partition  # 这里使用了numpy的广播机制

    # 定义损失函数
    def cross_entropy(self,y_pre, y):
        # y_pre: 预测值未经过softmax处理，shape is (batch_size, class_num)
        # y: 真值，shape is (batch_size)
        y_pre = self.my_softmax(y_pre, dim=1)
        return - torch.log(y_pre.gather(1, y.view(-1, 1))).mean()
        # 返回计算的loss均值
