import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchdiffeq import odeint


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


class Lambda(nn.Module):
    def forward(self, t, y):
        
        # unpack variables
        S, I, R = y
        N = S + I + R
        
        beta = 30# * (1 - torch.linspace(0., 1., 1000))
        gamma = 10
        
        dS_dt = -beta * I * S / N
        dI_dt = beta * I * S / N - gamma * I
        dR_dt = gamma * I
        
        return torch.stack([dS_dt, dI_dt, dR_dt])
    
    
NUM_SAMPLES = 1000
initial_states = torch.Tensor([9999., 100., 0.])
t = torch.linspace(0., 1., NUM_SAMPLES)
result = odeint(Lambda(), initial_states, t)

X1 = result[:, 0]
X2 = result[:, 1]
X3 = result[:, 2]

plt.plot(X1)
plt.plot(X2)
plt.plot(X3)
plt.plot(X1 + X2 + X3)


plt.show()

