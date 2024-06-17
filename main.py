import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from MTSFBet.model import MTSFBet
from MTSFBet.Lossfunction import Loss
from Tools.dataloader import Data
from Tools.run import train, val, val_best
from Tools.Custom_Scheduler import warm_up_cosine_lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':
    # 超参数设置
    Batch_size = 64
    lr = 1e-3
    epochs = 200
    num_classes = 20
    size        = 's'
    opti        = 'Adam'
    schedul     = 'warm_up_cosine_lr_scheduler'
    save_path   = './data/part_data/model_s_20.pt'
    # 获取数据
    x_trian = r'./data/part_data/x_train.npy'
    y_train = r'./data/part_data/y_train.npy'
    x_test  = r'./data/part_data/x_test.npy'
    y_test  = r'./data/part_data/y_test.npy'

    train_data, test_data = Data(x_trian, y_train, x_test, y_test, Batch_size)
    # 建立模型
    if size == 's':
        kernel_size = 1
        filter_size = 1
    if size == 'l':
        kernel_size = 3
        filter_size = 1
    model = MTSFBet(kernel_size, filter_size, num_classes).to(device)
    # model.load_state_dict(torch.load(save_path), strict=False)
    # 损失函数
    Loss = Loss()
    # 优化器
    if opti=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if opti=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    if schedul == 'warm_up_cosine_lr_scheduler':
        scheduler = warm_up_cosine_lr_scheduler(optimizer, warm_up_epochs = 3, eta_min = 1e-5)
    print('Batch_size={} || lr={} || epochs={} || num_classes={} || size={} || optim={} || scheduler={} || save_path={} '.format(Batch_size, lr, epochs, num_classes, size, opti, schedul, save_path))
    
    print('============================== Training Begins =======================================')
    acc_flag = 0
    for epoch in range(1, epochs+1):
        train(model, train_data, optimizer, Loss, epoch, scheduler, device, epochs)
        temp_loss = val(model, test_data, Loss, device)
        if temp_loss > acc_flag :
            torch.save(model.state_dict(),save_path)
            acc_flag = temp_loss
            print('============================== Model improved =======================================')
    
    print('Best Score:')
    val_best(model, test_data, Loss, device,save_path)
    print("Done!")