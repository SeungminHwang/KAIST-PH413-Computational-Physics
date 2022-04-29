import torch
import torch.nn as nn

rnn = nn.LSTM(10, 20, 2) # input_size, hidden_size, num_layers
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20) # ?, ?, hidden_size
c0 = torch.randn(2, 3, 20) # ?, ?, hidden_size

output, (hn, cn) = rnn(input, (h0, c0))

print(input)
print(output) # 5 * 3 * 20