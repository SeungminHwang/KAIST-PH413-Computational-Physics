import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

class SEIRD(nn.Module):    
    def forward(self, t, y):
        # update coefficient
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
        
        # set_params(params[t])
        
        # unpack states
        S, E, I, N, P, Hm, Hs, R, D = y
        
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
        # fix (add more DEs)
        
        # validity check
        #print(np.sum(np.array([dS_dt, dE_dt, dI_dt, dN_dt, dP_dt, dHm_dt, dHs_dt, dR_dt, dD_dt])))
        
        
        return torch.stack([dS_dt, dE_dt, dI_dt, dN_dt, dP_dt, dHm_dt, dHs_dt, dR_dt, dD_dt])

# test ODEINT

NUM_SAMPLES = 10000
initial_states = torch.Tensor([50000000., 0., 100., 0., 0., 0., 0., 0., 0.])
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
t = torch.linspace(0., 1000., NUM_SAMPLES)
result = odeint(SEIRD(), initial_states, t)

# S, E, I, N, P, Hm, Hs, R, D = y
S = result[:, 0]
E = result[:, 1]
I = result[:, 2]
N = result[:, 3]
P = result[:, 4]
Hm = result[:, 5]
Hs = result[:, 6]
R = result[:, 7]
D = result[:, 8]


ax = plt.subplot()
ax.plot(S)
ax.plot(E)
ax.plot(I)
ax.plot(N)
ax.plot(P)
ax.plot(Hm)
ax.plot(Hs)
ax.plot(R)
ax.plot(D)
ax.plot(S + E + I + N + P + Hm + Hs + R + D)

ax.legend(['S', 'E', 'I', 'N', 'P', 'Hm', 'Hs', 'R', 'D', 'SUM'])



plt.show()





'''
# Usage
NUM_SAMPLES = 1000
initial_states = torch.Tensor([9999., 100., 0.])
t = torch.linspace(0., 1., NUM_SAMPLES)
result = odeint(Lambda(), initial_states, t)
'''