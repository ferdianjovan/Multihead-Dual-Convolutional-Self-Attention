import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch import optim
from sklearn.preprocessing import MinMaxScaler

# Local library
from mdcsacrf import MDCSACRF
from optimisers import Lookahead, EarlyStopping


def load_pkl(file: str):
    """
    Load a pickle file
    """
    with open(file, 'rb') as f:
        return pickle.load(f)

    
def load_data(data_path: str, control: bool=True, daily_limit: list=[(8, 0), (20, 0)]):
    """
    data_path: Path to data
    control: Load HC or PD
    daily_limit: Limit the loaded data within time window
    """
    if control:
        file_path = os.path.join(data_path, 'C')
    else:
        file_path = os.path.join(data_path, 'PD')
    rssi = load_pkl(os.path.join(file_path, 'rssi.pkl'))
    accl = load_pkl(os.path.join(file_path, 'accl.pkl'))
    location = load_pkl(os.path.join(file_path, 'location.pkl'))
    ts = load_pkl(os.path.join(file_path, 'timestamp.pkl'))
    min_start = pd.Timestamp(str(ts[0, 0].date()) + f' 0{daily_limit[0][0]}:0{daily_limit[0][1]}:00')
    max_end = pd.Timestamp(str(ts[0, 0].date()) + f' 0{daily_limit[1][0]}:0{daily_limit[1][1]}:00')
    idx = np.argwhere((min_start <= ts[:, 0]) & (ts[:, 0] < max_end)).reshape(-1)
    return rssi[idx], accl[idx], location[idx], ts[idx]


def normalise(train: np.array, test: np.array=None, val: np.array=None):
    _, t, d = train.shape
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train.reshape(-1, d))
    if test is not None and val is not None:
        scaled_test = scaler.transform(test.reshape(-1, d))
        scaled_val = scaler.transform(val.reshape(-1, d))
        return (
            scaler, scaled_train.reshape(-1, t, d), 
            scaled_test.reshape(-1, t, d), scaled_val.reshape(-1, t, d)
        )
    elif val is None and test is not None:
        scaled_test = scaler.transform(test.reshape(-1, d))
        return scaler, scaled_train.reshape(-1, t, d), scaled_test.reshape(-1, t, d)
    else:
        return scaler, scaled_train.reshape(-1, t, d)


def train_validate(
    rssi: torch.Tensor, accl: torch.Tensor, locations: torch.Tensor, 
    rssi_val: torch.Tensor=None, accl_val: torch.Tensor=None, loc_val: torch.Tensor=None,
    params: dict=dict()
):
    """
    rssi: Rssi [Batch, T, Features]
    accl: Accelerometer [Batch, T, Features]
    locations: Room-level labels [Batch, T, Loc_Size]
    rssi_val: Rssi [Batch, T, Features]
    accl_val: Accelerometer [Batch, T, Features]
    loc_val: Room-level labels [Batch, T, Loc_Size]
    params: Hyperparameters {'hidden_size', 'learning_rate', 'epoch', 'batch_size'}
    """
    if accl is None:
        model = MDCSACRF(
            rssi_size=rssi.shape[-1], hidden_size=params['hidden_size'], 
            T=rssi.shape[1], loc_size=locations.shape[-1]
        )
    else:
        model = MDCSACRF(
            rssi_size=rssi.shape[-1], accl_size=accl.shape[-1], hidden_size=params['hidden_size'],
            T=rssi.shape[1], loc_size=locations.shape[-1]
        )
    model = model.to(rssi.device)
    # Optimizer with Lookahead
    base_optimizer = optim.RAdam(model.parameters(), lr=params['learning_rate'])
    optimizer = Lookahead(base_optimizer, k=5, alpha=.5)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=30, minimum_epoch=5)
    for epoch in range(params['epoch']):
        ###################
        # train the model #
        ###################
        model.train()
        total_loss = 0.0
        epoch_num_correct = 0
        epoch_num_correct_val = 0
        for batch_idx in range(int(rssi.shape[0] / params['batch_size'])):
            X0 = rssi[(batch_idx * params['batch_size']):((batch_idx + 1) * params['batch_size'])]
            # y: [B, T]
            y = locations[(batch_idx * params['batch_size']):((batch_idx + 1) * params['batch_size'])]
            X1 = None if accl is None else accl[
                (batch_idx * params['batch_size']):((batch_idx + 1) * params['batch_size'])
            ]
            loc_hat = model(rssi=X0, accl=X1)
            optimizer.zero_grad()
            loss = model.calculate_loss(
                rssi=X0, accl=X1, locations=y.argmax(-1).type(torch.long), with_recon=True
            ) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            epoch_num_correct += (loc_hat[:, -1] == y[:, -1].argmax(-1)).type(torch.float).sum().item()
            if batch_idx % 5 == 0:
                print('Batch: {}, Train Loss: {:.5f}, Acc: {:.5f}'.format(
                    batch_idx * params['batch_size'], total_loss / (batch_idx + 1), epoch_num_correct / (batch_idx + 1)
                ))

        ###################
        # validate model  #
        ###################
        if rssi_val is not None:
            model.eval()
            for batch_idx in range(int(rssi_val.shape[0] / params['batch_size'])):
                X0 = rssi_val[(batch_idx * params['batch_size']):((batch_idx + 1) * params['batch_size'])]
                y = loc_val[(batch_idx * params['batch_size']):((batch_idx + 1) * params['batch_size'])]
                X1 = None if accl_val is None else accl_val[
                    (batch_idx * params['batch_size']):((batch_idx + 1) * params['batch_size'])
                ]
                loc_hat = model(rssi=X0, accl=X1)
                epoch_num_correct_val += (loc_hat[:, -1] == y[:, -1].argmax(-1)).type(torch.float).sum().item()
            # early_stopping needs the validation accuracy to check if it has increased, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(epoch_num_correct_val / len(rssi_val), model)
            print('Val Acc: {:.5f}'.format(epoch_num_correct_val / len(rssi_val)))
            if early_stopping.early_stop:
                model.load_state_dict(torch.load('checkpoint.pt'))
                break
    return model
  
    
def test(model: torch.nn, rssi: torch.Tensor, accl: torch.Tensor, locations: torch.Tensor, params: dict=dict()):
    """
    model: NN model
    rssi: Rssi [Batch, T, Features]
    accl: Accelerometer [Batch, T, Features]
    locations: Room-level labels [Batch, T, Loc_Size]
    params: Hyperparameters {'hidden_size', 'learning_rate', 'epoch', 'batch_size'}
    """
    if accl is None:
        model = MDCSACRF(
            rssi_size=rssi.shape[-1], hidden_size=params['hidden_size'], 
            T=rssi.shape[1], loc_size=locations.shape[-1]
        )
    else:
        model = MDCSACRF(
            rssi_size=rssi.shape[-1], accl_size=accl.shape[-1], hidden_size=params['hidden_size'],
            T=rssi.shape[1], loc_size=locations.shape[-1]
        )
    model = model.to(rssi.device)
    model.eval()
    
    correct = 0
    num_batches = np.ceil(rssi.shape[0] / params['batch_size'])
    with torch.no_grad():
        for batch_idx in range(int(num_batches)):
            X0 = rssi[(batch_idx * params['batch_size']):((batch_idx + 1) * params['batch_size'])]
            y = locations[(batch_idx * params['batch_size']):((batch_idx + 1) * params['batch_size'])]
            X1 = None if accl is None else accl[
                (batch_idx * params['batch_size']):((batch_idx + 1) * params['batch_size'])
            ]
            loc_hat = model(rssi=X0, accl=X1)
            correct += (loc_hat[:, -1] == y[:, -1].argmax(-1)).type(torch.float).sum().item()
    correct /= rssi.shape[0]
    print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%")
    return (100*correct)