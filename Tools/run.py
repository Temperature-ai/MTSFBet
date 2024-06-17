import numpy as np
import torch
from tqdm import tqdm

def train(model, data_loader, optimizer, Loss, epoch, scheduler, device, epochs):
    # 设置训练状态
    model.train()
    loop = tqdm(enumerate(data_loader), total =len(data_loader))
    train_loss = 0
    # 循环读取DataLoader中的全部数据
    for step, (data_1, data_2, label) in loop:
        loop.set_description(f'Epoch [{epoch}/{epochs}]')
        # 将数据放到GPU用于后续计算
        data_1, data_2, label = data_1.to(device), data_2.to(device),label.to(device)
        # 将优化器的梯度清0
        optimizer.zero_grad()
        # 将数据输入给模型
        output = model(data_1, data_2)
        # 设置损失函数
        loss = Loss(output, label)
        # 将loss反向传播给网络
        loss.backward()
        # 使用优化器更新模型参数
        optimizer.step()
        scheduler.step()
        # 累加训练损失
        train_loss += loss.item() * data_1.size(0)
        loop.set_postfix(loss=loss.item())
    train_loss = train_loss / len(data_loader.dataset)
    print('Training Loss = {:.2f}'.format(train_loss))

def val(model, data_loader, Loss, device):
    # 设置验证状态
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels_1 = []
    pred_labels_2 = []
    t_2_pred      = []
    lable_2       = []
    # 不设置梯度
    with torch.no_grad():
        for data_1, data_2, label in data_loader:
            data_1, data_2, label = data_1.to(device), data_2.to(device), label.to(device)
            output = model(data_1, data_2)
            loss = Loss(output, label)
            val_loss += loss.item()*data_1.size(0)
            # 手势正确率计算
            gesture_label = label[:,0]
            t_1_1, t_1_2, t_2 = output
            preds_1 = torch.argmax(t_1_1, 1)
            preds_2 = torch.argmax(t_1_2, 1)
            gt_labels.append(gesture_label.cpu().data.numpy())
            pred_labels_1.append(preds_1.cpu().data.numpy())
            pred_labels_2.append(preds_2.cpu().data.numpy())
            # id正确率计算
            id_lable = label[:,1]
            t_2_pred.append(t_2.cpu().data.numpy())
            lable_2.append(id_lable.cpu().data.numpy())
    # 计算验证集的平均损失
    val_loss = val_loss/len(data_loader.dataset)
    gt_labels_1, pred_labels_1 = np.concatenate(gt_labels), np.concatenate(pred_labels_1)
    gt_labels_2, pred_labels_2 = np.concatenate(gt_labels), np.concatenate(pred_labels_2)

    t_2_pred, lable_2 = np.concatenate(t_2_pred), np.concatenate(lable_2)
    t_2_pred[t_2_pred >= 0.5] = 1
    t_2_pred[t_2_pred < 0.5] = 0
    # 计算准确率
    acc_gs = (np.sum(gt_labels_1 == pred_labels_1) + np.sum(gt_labels_2 == pred_labels_2)) / (len(pred_labels_1) + len(pred_labels_2))
    acc_id = np.sum(lable_2 == t_2_pred) / len(t_2_pred)
    print('Validation Loss = {:.2f} || Accuracy_Task_1 = {:.2f}% || Accuracy_Task_2 = {:.2f}%'.format(val_loss, acc_gs*100, acc_id*100))

    acc = acc_gs*0.65 + acc_id*0.35
    return acc

def val_best(model, data_loader, Loss, device, save_path):
    model.load_state_dict(torch.load(save_path), strict=False) # 将与训练权重载入模型

    # 设置验证状态
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels_1 = []
    pred_labels_2 = []
    t_2_pred      = []
    lable_2       = []
    # 不设置梯度
    with torch.no_grad():
        for data_1, data_2, label in data_loader:
            data_1, data_2, label = data_1.to(device), data_2.to(device), label.to(device)
            output = model(data_1, data_2)
            loss = Loss(output, label)
            val_loss += loss.item()*data_1.size(0)
            # 手势正确率计算
            gesture_label = label[:,0]
            t_1_1, t_1_2, t_2 = output
            preds_1 = torch.argmax(t_1_1, 1)
            preds_2 = torch.argmax(t_1_2, 1)
            gt_labels.append(gesture_label.cpu().data.numpy())
            pred_labels_1.append(preds_1.cpu().data.numpy())
            pred_labels_2.append(preds_2.cpu().data.numpy())
            # id正确率计算
            id_lable = label[:,1]
            t_2_pred.append(t_2.cpu().data.numpy())
            lable_2.append(id_lable.cpu().data.numpy())
    # 计算验证集的平均损失
    val_loss = val_loss/len(data_loader.dataset)
    gt_labels_1, pred_labels_1 = np.concatenate(gt_labels), np.concatenate(pred_labels_1)
    gt_labels_2, pred_labels_2 = np.concatenate(gt_labels), np.concatenate(pred_labels_2)

    t_2_pred, lable_2 = np.concatenate(t_2_pred), np.concatenate(lable_2)
    t_2_pred[t_2_pred >= 0.5] = 1
    t_2_pred[t_2_pred < 0.5] = 0
    # 计算准确率
    acc_gs = (np.sum(gt_labels_1 == pred_labels_1) + np.sum(gt_labels_2 == pred_labels_2)) / (len(pred_labels_1) + len(pred_labels_2))
    acc_id = np.sum(lable_2 == t_2_pred) / len(t_2_pred)
    print('Validation Loss = {:.2f} || Accuracy_Task_1 = {:.2f}% || Accuracy_Task_2 = {:.2f}%'.format(val_loss, acc_gs*100, acc_id*100))