import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F

from util import *

# global variables
initial_params = torch.Tensor()
#[1/3, 1/3.2, 0.84, 0.26, 0.035, 0.025, 1/3.5, 1/16, 1/16, 1/45, 0.001]



class ParameterNet(nn.Module):
    
    def __init__(self, n_input, n_output, n_window):
        super(ParameterNet, self).__init__()
        
        self.w = n_window
        h = 16
        
        self.lstm = nn.LSTM(n_input, h, num_layers=5,
                            batch_first=True, dropout=0.5)
        self.out = nn.Linear(n_window * h, n_output)
                
    def forward(self, x):
        b, _, _ = x.size()
        #print(size)
        
        r, _ = self.lstm(x)
        #x = self.lstm(x)
        r = F.elu(r)
        r = torch.flatten(r, start_dim=1)
        
        lo = self.out(r)
        out = lo.reshape(b, -1)
        return out

class ParameterNet_(nn.Module):
    
    def __init__(self, input_dim=65, output_dim=11):
        super(ParameterNet_, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0, std=0.1)
                nn.init.constant_(m.bias, val=0)
                
    def forward(self, x):
        x = self.net(x)
        return x


class SEIRD(nn.Module):
    def forward(self, t, y):
        # update coefficient (fix)
        '''
        initial_params = {
            'alpha_E': 1/3,
            'alpha_I': 1/3.2,
            'rho': 0.84,
            'kappa': 0.26,
            'beta_I': 0.035,
            'beta_N': 0.025,
            'gamma_N': 1/3.5,
            'gamma_M': 1/16,
            'gamma_S': 1/16,
            'delta': 1/45,
            'mu': 0.001,
        }
        '''
        
        alpha_E = initial_params[:, SEIRD_Param.alpha_E.value]
        alpha_I = initial_params[:, SEIRD_Param.alpha_I.value]
        rho = initial_params[:, SEIRD_Param.rho.value]
        kappa = initial_params[:, SEIRD_Param.kappa.value]
        beta_I = initial_params[:, SEIRD_Param.beta_I.value]
        beta_N = initial_params[:, SEIRD_Param.beta_N.value]
        gamma_N = initial_params[:, SEIRD_Param.gamma_N.value]
        gamma_M = initial_params[:, SEIRD_Param.gamma_M.value]
        gamma_S = initial_params[:, SEIRD_Param.gamma_S.value]
        delta = initial_params[:, SEIRD_Param.delta.value]
        mu = initial_params[:, SEIRD_Param.mu.value]
        
        # unpack states
        #print(y.shape)
        S = y[:,0]
        E = y[:,1]
        I = y[:,2]
        N = y[:,3]
        P = y[:,4]
        Hm = y[:,5]
        Hs = y[:,6]
        R = y[:,7]
        D = y[:,8]
        
        #S, E, I, N, P, Hm, Hs, R, D = y # (fix)add param???
        
        # System of Differential Equations
        dS_dt = - beta_I * I - beta_N * N + delta * R
        dE_dt = beta_I * I + beta_N * N - alpha_E * E
        dI_dt = alpha_E * E - rho * alpha_I * I - (1 - rho) * alpha_I * I
        dN_dt = (1 - rho) * alpha_I * I - gamma_N * N
        dP_dt = rho * alpha_I * I - kappa * P - (1 - kappa) * P
        dHm_dt = (1 - kappa) * P - gamma_M * Hm
        dHs_dt = kappa * P - gamma_S * Hs - mu * Hs
        dR_dt = gamma_N * N + gamma_M * Hm + gamma_S * Hs - delta * R
        dD_dt = mu * Hs
        
        out = torch.stack([dS_dt, dE_dt, dI_dt, dN_dt, dP_dt, dHm_dt, dHs_dt, dR_dt, dD_dt], axis=-1)
        
        self.output = out
        
        return out

def set_param(params, device):
    global initial_params
    initial_params = params
    initial_params.to(device)