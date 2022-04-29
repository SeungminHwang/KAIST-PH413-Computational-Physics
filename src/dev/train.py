import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms

from torchdiffeq import odeint

import Network


'''
parser = argparse.ArgumentParser()
parser.add_argument("--save", type=str, required=True, help="folder to save model and generated images")
parser.add_argument("--gpu", type=int, default=0, help="gpu to use")
parser.add_argument("--lr1", type=float, default=2e-3, help="learning rate for Generator")
parser.add_argument("--lr2", type=float, default=2e-3, help="learning rate for Discriminator")
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument("--epoch", type=int, default=5, help="epochs to run")
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
args = parser.parse_args()
'''


def main():
    
    NUM_EPOCHS = 100
    
    
    is_cuda = torch.cuda.is_available()
    # device = torch.device('cuda:{}'.format(args.gpu) if is_cuda else 'cpu')
    device = torch.device("cpu")
    
    train_data = None
    train_loader = None
    
    test_data = None
    
    
    # optimizer
    optim = optim.SGD()
    criterion = nn.L1Loss()
    
    
    pass


if __name__ == "__main__":
    main()