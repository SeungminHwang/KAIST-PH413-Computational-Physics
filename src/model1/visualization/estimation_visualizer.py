import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.ndimage
import time
import os
from dateutil.parser import parse
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint

from util import *
import Dataset
import Network


import random
random.seed(1111)
torch.manual_seed(1111)
np.random.seed(1111)
np.random.default_rng(1111)



# Settings
EPOCH = 1
BATCH_SIZE = 100
#target_pt = "000014_model_all.pt"
target_pt = "000451_model.pt"
#target_pt = "out_all/000013_model.pt"


# Queries
target_date = 20210101
target_countries = ['South Korea']
#target_countries = ['Austria']


# global variables
dates = np.zeros(0)
params = np.zeros((11, 0))
states = np.zeros((9, 0))
ys = np.zeors((2, 0))



# device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
check_device()


# load data
DatasetInput, DatasetLabel, DatasetState, DatasetAux = load_dataset(target_countries=target_countries)

# unpack dataset
train_input, train_label, train_state, train_aux, valid_input, valid_label, valid_state, valid_aux = unpack_dataset(DatasetInput, DatasetLabel, DatasetState, DatasetAux)
print("Train data shape: ", train_input.shape, train_label.shape, train_state.shape)
print("Valid data shape: ", valid_input.shape, valid_label.shape, valid_state.shape)


# transform to pytorch Tensor
train_dataset = Dataset.COVID_SIR_Dataset(
                    train_input,
                    train_label,
                    train_state,
                    train_aux, device=device)
valid_dataset = Dataset.COVID_SIR_Dataset(
                    valid_input,
                    valid_label,
                    valid_state,
                    valid_aux, device=device)


# dataloader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=False)
valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE, shuffle=False)



# load `pre-trained` model
print("Loading model...", end="")
n_input = DatasetInput[0].shape[1]
n_window = DatasetInput[0].shape[0]
n_label = DatasetLabel[0].shape[0]
n_state = DatasetState[0].shape[0]

model = Network.ParameterNet(n_input, 11, n_window).to(device)
model.load_state_dict(torch.load(target_pt))



n_sample_train = train_dataset.n_sample
lr_step_size = int(n_sample_train / BATCH_SIZE)

print("Completed!")

# loss function, optimizer, learing rate
loss_fn = nn.L1Loss().to(device)

#model.eval()

# evaluation
start_time = time.time()
train_loss = 0
valid_loss = 0
with torch.no_grad():
    for idx_batch, (x, y, s0, aux) in enumerate(tqdm (train_loader)):
        
        x = x.to(device) # batch x window x input
        y = y.to(device) # batch x window*2 x (acc. pos, acc. death)
        s0 = s0.to(device) # [S, E, ...]
        #aux_info = aux
        
        # ParameterNet
        ParameterNet_output = model(x) # batch x num_param
        curr_batch, _ = ParameterNet_output.shape
        #print(ParameterNet_output.shape)
        
        # SEIRD odeint
        t = torch.linspace(0., 2 * n_window - 1, 2 * n_window * 24).to(device)
        #print(t)
        
        #print(s0.shape)
        Network.set_param(ParameterNet_output, device)
        SEIRD_output = odeint(Network.SEIRD(), s0, t)
        SEIRD_output = torch.swapaxes(SEIRD_output, 0, 1)
        SEIRD_output = torch.index_select(SEIRD_output, 1, torch.Tensor(range(0, 2 * n_window * 24, 24)).int().to(device))
        
        
        #odeint_plot(SEIRD_output, vis=False, save=True)
        
        pred = torch.index_select(SEIRD_output, 2, torch.Tensor([SEIRD_State.P.value, SEIRD_State.D.value]).int().to(device))
        
        
        
        
        #odeint_plot(SEIRD_output, vis=False, save=True)
        
        # loss
        loss1 = loss_fn(pred, y)
        loss2 = loss_fn(pred, torch.abs(pred))
        loss3 = loss_fn(ParameterNet_output, torch.abs(ParameterNet_output))
        loss = loss1 + loss2 + loss3
        train_loss += loss
        
        
        
        
        
        # SAVE RESULTS
        ## save dates
        date = aux[:, 0, 1]
        dates = np.concatenate((dates, date))
        
        ## save params
        #ParameterNet_output -> batch x num_param
        param_numpy = ParameterNet_output.cpu().detach().numpy()
        param_write = np.swapaxes(param_numpy, 0, 1)
        params = np.concatenate((params, param_write), axis=1)
        
        ## save states
        states_numpy = SEIRD_output.cpu().detach().numpy()
        states_write = np.swapaxes(states_numpy, 0, 1)
        states = np.concatenate((states, states_write), axis=1)
        
        ## save labels
        labels_numpy = y.cpu().detach().numpy()
        y_write = np.swapaxes(labels_numpy, 0, 1)
        ys = np.concatenate((ys, y_write), axis=1)
        
        
        
        
    elapsed_time = time.time() - start_time
    print("\r %05d | Train Loss: %.7f | time: %.3f" %
            (1, train_loss, elapsed_time))


# save and visualize
print(params.shape)
print(dates.shape)

## prepare datetime
dates = dates.astype(np.int32)
date_arr = [str(x) for x in dates]
dt_arr = []
for date in date_arr:
    dt = parse(date)
    #dt = dt - timedelta(days=30)
    dt_arr.append(dt)


# get enum values
P_idx = SEIRD_State.S.value
D_idx = SEIRD_







