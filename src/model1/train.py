import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint

from tqdm import tqdm
import matplotlib.pyplot as plt


from util import *
import Dataset
import Network

EPOCH = 1000
BATCH_SIZE = 100
INIT_LR = 0.001
WEIGHT = 0.0001


# device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
check_device()


# load dataset
DatasetInput, DatasetLabel, DatasetState, DatasetAux = load_dataset()

# unpack dataset
train_input, train_label, train_state, train_aux, valid_input, valid_label, valid_state, valid_aux = unpack_dataset(DatasetInput, DatasetLabel, DatasetState, DatasetAux)
print("Train data shape: ", train_input.shape, train_label.shape, train_state.shape)
print("Valid data shape: ", valid_input.shape, valid_label.shape, valid_state.shape)


# transform to the pytorch Tensor
print("Putting data to loader...", end="")
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

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE, shuffle=False)
print("Completed!")



print("Loading model...", end="")
n_input = DatasetInput[0].shape[1]
n_window = DatasetInput[0].shape[0]
n_label = DatasetLabel[0].shape[0]
n_state = DatasetState[0].shape[0]

#print(n_window, n_input, n_label, n_state)

model = Network.ParameterNet(n_input, 11, n_window).to(device)

n_sample_train = train_dataset.n_sample
lr_step_size = int(n_sample_train / BATCH_SIZE)

# loss function, optimizer, learing rate
loss_fn = nn.L1Loss().to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=INIT_LR, weight_decay=WEIGHT)
lr_sch = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer, step_size=lr_step_size, gamma=0.99)
print("Completed!")






print("########## START TRAIN ##########")

for idx_epoch in range(EPOCH + 1):
    start_time = time.time()
    
    train_loss = 0.
    
    for idx_batch, (x, y, s0, aux) in enumerate(tqdm (train_loader)):
        model.zero_grad()
        
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
        # guiding loss
        param_guide = [1/3, 1/3.2, 0.84, 0.26, 0.035, 0.025, 1/3.5, 1/16, 1/16, 1/180, 0.001] * curr_batch
        param_guide = np.array(param_guide).reshape(-1, 11).transpose()
        param_guide = torch.Tensor(param_guide)
        param_guide = torch.swapaxes(param_guide, 0, 1)
        
        
        #loss_guide = 
        if idx_epoch < 0:
            loss = loss_fn(ParameterNet_output,param_guide)
            loss.backward()
            train_loss += loss
        else:
            # predict loss
            loss1 = loss_fn(pred, y)
            loss2 = loss_fn(pred, torch.abs(pred))
            loss3 = loss_fn(ParameterNet_output, torch.abs(ParameterNet_output))
            loss = loss1 + loss2 + loss3
            loss.backward()
            train_loss += loss
        
        
        # update
        optimizer.step()
        lr_sch.step()
        
        
        #print(loss)
        
    elapsed_time = time.time() - start_time
    print("\r %05d | Train Loss: %.7f | lr: %.7f | time: %.3f" %
          (idx_epoch + 1, train_loss, optimizer.param_groups[0]['lr'], elapsed_time))
    
    if idx_epoch % 50 == 0:
        save_model(model, idx_epoch+1, 'out/')
    
    if idx_epoch % 10 == 0:
        pass
        odeint_plot(SEIRD_output, vis=False, save=True, epoch=idx_epoch+1, aux=aux,
                    out_folder='out')
        export_results(ParameterNet_output, SEIRD_output,
                       pred, y, out_folder='out7', aux=aux, epoch=idx_epoch+1)
    
    if idx_epoch % 50 == 0:
        pass










######## DATA PREPROCESSING ########
# TODO
####################################


