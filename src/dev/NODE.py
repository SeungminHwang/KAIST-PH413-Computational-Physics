import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint

import matplotlib.pyplot as plt

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., 1000 ).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class Lambda(nn.Module):
    
    def forward(self, t, y):
        return torch.mm(y**3, true_A)

true_y = odeint(Lambda(), true_y0, t, method='dopri5')


class ODEFunc(nn.Module):
    
    def __init__(self):
        super(ODEFunc, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        x = self.net(y**3)
        return x


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
        

x = torch.Tensor(np.ones(65)).to(device)
y = torch.Tensor(np.ones(2)).to(device)

model = ParameterNet().to(device)

optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

loss_fn = nn.MSELoss().to(device)

for iter in range(1, 1000 + 1):
    optimizer.zero_grad()
    
    pred_y = model(x)
    
    
    new_x = pred_y[:2]
    
    new_x = new_x.view(1, 2)
    
    print(new_x.shape)
    
    
    
    pred_y = odeint(Lambda(), new_x, t).to(device)
    
    
    loss = loss_fn(new_x, y)
    
    print(loss)
    
    loss.backward()
    
    optimizer.step()
  



'''
func = ODEFunc().to(device)
optimizer = optim.RMSprop(func.parameters(), lr=1e-3)


for iter in range(1, 100 + 1):
    optimizer.zero_grad()
    
    pred_y = odeint(func, true_y0, t).to(device)
    
    print(pred_y.shape)
    x = pred_y[-1,:]
    
    print(x.shape)
    
    print("before ode", x)
    
    x = odeint(Lambda(), x, t).to(device)
    
    print("after ode", x)
    
    loss = torch.mean(torch.abs(x - true_y))
    
    loss.backward()
    
    optimizer.step()
    
    print(loss)
    break

print(pred_y.shape)


#print(true_y)
Y = true_y.numpy()
print(Y.shape)
plt.plot(true_y[:, 0, 0])
plt.plot(true_y[:, 0, 1])
#plt.plot(pred_y.detach().numpy()[:, 0, 0])
#plt.plot(pred_y.detach().numpy()[:, 0, 1])
#plt.show()



'''