import json
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import sys
import warnings
warnings.filterwarnings('ignore')

# Path to local library
sys.path.append('./models')

# Local library
from util import load_data, normalise, train_validate, test

# Random seed
params = json.load(open("hyperparams.json", "r"))
NJOBS = params['njobs']
RANDOM_STATE = params['random_state']
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE) 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    
    rssi_train, accl_train, location_train, _ = load_data(params['data_path'], control=True)
    rssi_test, accl_test, location_test, _ = load_data(params['data_path'], control=False)

    train_idx, validation_idx = np.array([]).astype(np.int64), np.array([]).astype(np.int64)
    for room in ['dining_room', 'living_room', 'kitchen', 'hall', 'stairs', 'porch/outside_front_door']:
        temp = np.unique(np.argwhere(location_train == room)[:, 0])
        if len(temp):
            # Limit train data to 300 samples per room (= 1 minute worth of samples per room with 5Hz sampling rate)
            limit = np.min([len(temp), 300])
            if limit < 300:
                limit = int(limit / 2)
            train_idx = np.concatenate([train_idx, temp[:limit]])
            validation_idx = np.concatenate([validation_idx, temp[limit:]])
        else:
            print(f'Data for ({room}) is not available!')

    rssi_scaler, rssi_train, rssi_test, rssi_val = normalise(rssi_train[train_idx], rssi_test, rssi_train[validation_idx])
    accl_scaler, accl_train, accl_test, accl_val = normalise(accl_train[train_idx], accl_test, accl_train[validation_idx])

    encoder = OneHotEncoder(sparse=False)
    encoder.fit(np.unique(np.concatenate(
        [location_train[train_idx], location_train[validation_idx], location_test]
    )).reshape(-1, 1))
    location_val = encoder.transform(
        location_train[validation_idx].reshape(-1, 1)
    ).reshape(len(validation_idx), location_train.shape[1], -1)
    location_train = encoder.transform(
        location_train[train_idx].reshape(-1, 1)
    ).reshape(len(train_idx), location_train.shape[1], -1)
    location_test = encoder.transform(
        location_test.reshape(-1, 1)
    ).reshape(rssi_test.shape[0], rssi_test.shape[1], -1)

    print('Train data: {}, Validation data: {}, Test data: {}'.format(
        rssi_train.shape, rssi_val.shape, rssi_test.shape
    ))
    
    model = train_validate(
        torch.from_numpy(rssi_train).type(torch.float).to(device), 
        torch.from_numpy(accl_train).type(torch.float).to(device), 
        torch.from_numpy(location_train).type(torch.float).to(device),
        torch.from_numpy(rssi_val).type(torch.float).to(device),
        torch.from_numpy(accl_val).type(torch.float).to(device),
        torch.from_numpy(location_val).type(torch.float).to(device), 
        params
    )
    
    test(
        model, 
        torch.from_numpy(rssi_test).type(torch.float).to(device), 
        torch.from_numpy(accl_test).type(torch.float).to(device), 
        torch.from_numpy(location_test).type(torch.float).to(device), 
        params
    )