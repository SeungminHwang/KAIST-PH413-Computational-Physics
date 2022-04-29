import torch
import torch.nn as nn
import torch.nn.functional as F




'''
class SolveRNN(nn.Module):
    def __init__(self, n_input, n_output, n_window):
        super(SolveRNN, self).__init__()

        self.w = n_window
        h = 38 * 3

        self.lstm1 = nn.LSTM(n_input, h, num_layers=2,
                             batch_first=True, dropout=0.5)
        self.lin1 = nn.Linear(n_window * h, n_window * 96)
        self.out = nn.Linear(n_window * 96, n_window * n_output)
        # quat to matrix?

    def forward(self, x):
        b, _, _ = x.size()

        r, _ = self.lstm1(x)
        r = F.elu(r)
        r = torch.flatten(r, start_dim=1)
        l1 = self.lin1(r)
        l1 = F.elu(l1)
        lo = self.out(l1)

        out = lo.reshape(b, self.w, -1)
        
        # quat to matrix

        return out
'''

class SEIRA_Net(nn.Module):
    def __init__(self, n_input, n_output, n_window):
        super(SEIRA_Net, self).__init__()
        
        self.w = n_window
        hidden = 5
        
        self.lstm1 = nn.LSTM(n_input, h, num_layers=2,
                             batch_first=True, dropout=0.5)
        self.lin1 = nn.Linear(n_window * h, n_window * 96)
        self.out = nn.Linear(n_window * 96, n_window * n_output)
        
    def forward(self, x):
        pass