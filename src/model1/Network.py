import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchdiffeq import odeint


class ParameterNet(nn.Module):
    
    def __init__(self, input_dim=65, output_dim=11):
        super(ParameterNet, self).__init__()
        
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
        alpha_E = initial_params['alpha_E']
        alpha_I = initial_params['alpha_I']
        rho = initial_params['rho']
        kappa = initial_params['kappa']
        beta_I = initial_params['beta_I']
        beta_N = initial_params['beta_N']
        gamma_N = initial_params['gamma_N']
        gamma_M = initial_params['gamma_M']
        gamma_S = initial_params['gamma_S']
        delta = initial_params['delta']
        mu = initial_params['mu']
        
        # unpack states
        S, E, I, N, P, Hm, Hs, R, D = y # (fix)add param???
        
        # System of Differential Equations
        dS_dt = - beta_I * S - beta_N * S + delta * R
        dE_dt = beta_I * S + beta_N * S - alpha_E * E
        dI_dt = alpha_E * E - rho * alpha_I * I - (1 - rho) * alpha_I * I
        dN_dt = (1 - rho) * alpha_I * I - gamma_N * N
        dP_dt = rho * alpha_I * I - kappa * P - (1 - kappa) * P
        dHm_dt = (1 - kappa) * P - gamma_M * Hm
        dHs_dt = kappa * P - gamma_S * Hs - mu * Hs
        dR_dt = gamma_N * N + gamma_M * Hm + gamma_S * Hs - delta * R
        dD_dt = mu * Hs
        
        return torch.stack([dS_dt, dE_dt, dI_dt, dN_dt, dP_dt, dHm_dt, dHs_dt, dR_dt, dD_dt])