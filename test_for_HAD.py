import argparse
import copy
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from tensorboardX import SummaryWriter

import datasets
import flows as fnn
import utils
import multiprocessing
import os
import scipy.io
if sys.version_info < (3, 6):
    print('Sorry, this code might need Python 3.6 or higher')
# if __name__ == '__main__':
#     multiprocessing.freeze_support()
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Flows')
parser.add_argument(
    '--batch-size',
    type=int,
    default=1000,
    help='input batch size for training (default: 100)')
parser.add_argument(
    '--epochs',
    type=int,
    default=101,
    help='number of epochs to train (default: 1000)')
parser.add_argument(
    '--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument(
    '--dataset',
    default='HYPERION',
    help='HU | PAVIA | HYPERION')
parser.add_argument(
    '--flow', default='glow', help='flow to use: maf | realnvp | glow')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--cond',
    action='store_true',
    default=False,
    help='train class conditional flow (only for MNIST)')
parser.add_argument(
    '--num-blocks',
    type=int,
    default=3,
    help='number of invertible blocks (default: 5)')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=1000,
    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

assert args.dataset in [
    'PAVIA', 'HU', 'HYPERION'
]
dataset = getattr(datasets, args.dataset)()


train_tensor = torch.from_numpy(dataset.HIM.x)
train_dataset = torch.utils.data.TensorDataset(train_tensor)

num_cond_inputs = None
    
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=dataset.HIM.N,
    shuffle=False,
    drop_last=False,
    **kwargs)

num_inputs = dataset.n_dims
num_hidden = {
    'PAVIA': 100,
    'HU': 100,
    'HYPERION': 100
}[args.dataset]

act = 'relu'

modules = []

assert args.flow in ['maf', 'maf-split', 'maf-split-glow', 'realnvp', 'glow']
if args.flow == 'glow':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    print("Warning: Results for GLOW are not as good as for MAF yet.")
    for _ in range(args.num_blocks):
        modules += [
            fnn.BatchNormFlow(num_inputs),
            fnn.LUInvertibleMM(num_inputs),
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
                s_act='tanh', t_act='relu')
        ]
        mask = 1 - mask
elif args.flow == 'realnvp':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    for _ in range(args.num_blocks):
        modules += [
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
                s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs)
        ]
        mask = 1 - mask
elif args.flow == 'maf':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
elif args.flow == 'maf-split':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                         s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
elif args.flow == 'maf-split-glow':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                         s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs),
            fnn.InvertibleMM(num_inputs)
        ]

model = fnn.FlowSequential(*modules)

for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(0)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

writer = SummaryWriter(comment=args.flow + "_" + args.dataset)
global_step = 0

def train(epoch):
    global global_step, writer
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        log_probs_batch = model.log_probs(data, cond_data)
        # a = log_probs_batch.view(-1)
        topk_indices = torch.argsort(log_probs_batch.view(-1))[5:]
        topk_samples = log_probs_batch[topk_indices]
        loss = -topk_samples.mean()
        # loss = -model.log_probs(data, cond_data).mean()

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))
        
        writer.add_scalar('training/loss', loss.item(), global_step)
        global_step += 1
        
    pbar.close()
        
    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    if args.cond:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device),
                train_loader.dataset.tensors[1].to(data.device).float())
    else:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device))


    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1


def validate(model, loader, prefix='Validation'):
    global global_step, writer

    model.eval()
    val_loss = 0
    loss_tensor = torch.Tensor([]).to(device)

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            # loss_tensor = torch.cat((loss_tensor, model.log_probs(data, cond_data).unsqueeze(0)), dim=0)
            loss_tensor = model.log_probs(data, cond_data)
            val_loss += -model.log_probs(data, cond_data).sum().item()  # sum up batch loss
        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    writer.add_scalar('validation/LL', val_loss / len(loader.dataset))

    pbar.close()

    # loss_tensor = loss_tensor.unsqueeze(0)
    return loss_tensor
    


for epoch in range(args.epochs):
    print('\nEpoch: {}'.format(epoch))
    
    train(epoch)

    if epoch % 10 == 0:
        utils.save_plot(epoch, model, dataset, args.dataset)
        logpz = validate(model, valid_loader).cpu().numpy()
        logpz = (max(logpz)-logpz)/(max(logpz)-min(logpz))
        result = logpz.reshape((dataset.H, dataset.W))
        try:
            os.makedirs('results')
        except OSError:
            pass
        result_path = "./results/"

        ouput_path = result_path + args.dataset + '_flow_result_{:04d}_1_9'.format(epoch) + ".mat"
        scipy.io.savemat(ouput_path, {'flow_result': result})
