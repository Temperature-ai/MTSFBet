import torch
from MTSFBet.Experiment_model import MTSFBet
import numpy as np
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def data_process(inputs):
        data_1 = inputs[0:1000]
        data_2 = inputs[1000:2000]
        data_3 = inputs[2000:3000]
        data_4 = inputs[3000:4000]
        data_5 = inputs[4000:5000]
        data_6 = inputs[5000:6000]
        temp = np.stack([data_1, data_2, data_3, data_4, data_5, data_6])
        data = np.transpose(temp, (1, 0))
        data = torch.from_numpy(np.float32(np.expand_dims(data,0)))
        data = data.to(device)
        return data
    
if __name__ == '__main__':
    size        = 's'
    num_classes = 20
    if size == 's':
        kernel_size = 1
        filter_size = 1
    if size == 'l':
        kernel_size = 3
        filter_size = 1   
    save_path = './data/part_data/model_l_20.pt'
    
    # loda data 
    x_test  = np.load('./data/total_data/x_test.npy')
    y_test  = np.load('./data/total_data/y_test.npy')
    
    Experiment_model = MTSFBet(kernel_size, filter_size, num_classes).to(device)
    Experiment_model.load_state_dict(torch.load(save_path), strict=False) # 将与训练权重载入模型
    Experiment_model.eval()
    
    acc_flag_1 = 0
    acc_flag_2 = 0
    # 不设置梯度
    with torch.no_grad():
        for i in tqdm(range(len(x_test))):
            test_data_1 = data_process(x_test[i][0:6000])
            test_data_2 = data_process(x_test[i][6000:12000])
            output = Experiment_model(test_data_1, test_data_2)
            if output[0].cpu().data.numpy() > 0.9 :
                acc_flag_1 += 1
            if (output[1].cpu().data.numpy() > 0.5 and y_test[i][1]==1) or (output[1].cpu().data.numpy() <= 0.5 and y_test[i][1]==0):
                acc_flag_2 += 1
    print("Accuracy Task_1 = {:.2f}%".format(acc_flag_1/len(x_test)*100))
    print("Accuracy Task_2 = {:.2f}%".format(acc_flag_2/len(x_test)*100))