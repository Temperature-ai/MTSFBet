import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    select_class = 20
    
    x_train = np.load('./data/total_data/x_train.npy')
    y_train = np.load('./data/total_data/y_train.npy')
    x_test  = np.load('./data/total_data/x_test.npy')
    y_test  = np.load('./data/total_data/y_test.npy')
    
    # built part data
    x_train_part = []
    y_train_part = []
    
    for i in tqdm(range(len(x_train))):
        if y_train[i][0] < select_class:
            x_train_part.append(x_train[i])
            y_train_part.append(y_train[i])
    x_train_part = np.array(x_train_part)
    y_train_part = np.array(y_train_part)
    
    print(x_train_part.shape)
    print(y_train_part.shape)
    
    np.save('./data/part_data/x_train.npy', x_train_part)
    np.save('./data/part_data/y_train.npy', y_train_part)
    
    print('train_done')
    
    x_test_part = []
    y_test_part = []
    for i in tqdm(range(len(x_test))):
        if y_test[i][0] < select_class:
            x_test_part.append(x_test[i])
            y_test_part.append(y_test[i])
    x_test_part = np.array(x_test_part)
    y_test_part = np.array(y_test_part)
    
    print(x_test_part.shape)
    print(y_test_part.shape)
    
    np.save('./data/part_data/x_test.npy', x_test_part)
    np.save('./data/part_data/y_test.npy', y_test_part)
    
    print('test_done')
            