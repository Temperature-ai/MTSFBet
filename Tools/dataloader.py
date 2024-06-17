from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class Data_build(Dataset):
    def __init__(self, data_path, label_path):
        self.data  = np.load(data_path)
        self.label = np.load(label_path)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data  = np.float32(self.data[idx])
        inputs_data_1  = self.data_process(data[0:6000])
        inputs_data_2 = self.data_process(data[6000:12000])
        label = self.label[idx]
        return inputs_data_1, inputs_data_2, label

    def data_process(self, inputs):
        data_1 = inputs[0:1000]
        data_2 = inputs[1000:2000]
        data_3 = inputs[2000:3000]
        data_4 = inputs[3000:4000]
        data_5 = inputs[4000:5000]
        data_6 = inputs[5000:6000]
        temp = np.stack([data_1, data_2, data_3, data_4, data_5, data_6])
        data = np.transpose(temp, (1, 0))
        return data

def Data(x_train_path,
         y_train_path,
         x_test_path,
         y_test_path,
         Batch_size):

    train_data = Data_build(x_train_path, y_train_path)
    test_data  = Data_build(x_test_path,  y_test_path )

    train_data = DataLoader(
        dataset=train_data,
        batch_size=Batch_size,
        shuffle=True
    )

    test_data = DataLoader(
        dataset=test_data,
        batch_size=Batch_size*2,
        shuffle=True
    )

    return train_data, test_data