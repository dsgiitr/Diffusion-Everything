import os
import numpy as np
import functools

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from model import ScoreNet1C, ScoreNet3C
from sampler import ode_sampler, Euler_Maruyama_sampler, pc_sampler
import tqdm
import argparse


import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class ScoreModel():
    def __init__(self, batch_size, lr, sigma, device, is_RGB):

        self.device = device

        self.batch_size =  batch_size
        self.lr = lr
        self.sigma = sigma

        self.is_RGB = is_RGB

        self.marginal_prob_std_fn = functools.partial(self.marginal_prob_std, sigma=self.sigma, device=self.device)
        self.diffusion_coeff_fn = functools.partial(self.diffusion_coeff, sigma=self.sigma, device=self.device)

        if(is_RGB):
            self.score_model = torch.nn.DataParallel(ScoreNet3C(marginal_prob_std=self.marginal_prob_std_fn))
        else:
            self.score_model = torch.nn.DataParallel(ScoreNet1C(marginal_prob_std=self.marginal_prob_std_fn))

        self.score_model = self.score_model.to(device)

    def marginal_prob_std(self, t, sigma, device):

        t = t.clone().detach()
        return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))
    
    def diffusion_coeff(self, t, sigma, device):

        return torch.tensor(sigma**t, device=device)
    
    def loss_fn(self, model, x, marginal_prob_std, eps=1e-5):
        
        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
        z = torch.randn_like(x, device=x.device)
        std = marginal_prob_std(random_t)
        perturbed_x = x + z * std[:, None, None, None]
        score = model(perturbed_x, random_t)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
        return loss
    
    def train(self, data_loader, n_epochs, checkpoint_path):

        if os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.score_model.load_state_dict(ckpt)
        else:
            print(f"No checkpoint file found at {checkpoint_path}. Model will be initialized randomly.")

        optimizer = Adam(self.score_model.parameters(), lr=self.lr)
        tqdm_epoch = tqdm.trange(n_epochs)
        for epoch in tqdm_epoch:
            avg_loss = 0.
            num_items = 0
            for x, y in data_loader:
                x = x.to(self.device)    
                loss = self.loss_fn(self.score_model, x, self.marginal_prob_std_fn)
                optimizer.zero_grad()
                loss.backward()    
                optimizer.step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
                
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            
            torch.save(self.score_model.state_dict(), checkpoint_path)

    def generate(self, checkpoint_path, dataset, sampler, device = 'cpu'):

        ckpt = torch.load(checkpoint_path, map_location=device)
        self.score_model.load_state_dict(ckpt)

        if(dataset=='mnist'):
            self.z = torch.randn(self.batch_size, 1, 28, 28, device=device)
        elif(dataset=='cifar10'):
            self.z = torch.randn(self.batch_size, 3, 64, 64, device=device)

        # 'Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'
        if(sampler=='ode_sampler'):
            self.sampler = ode_sampler
        elif(sampler=='pc_sampler'):
            self.sampler = pc_sampler
        elif(sampler=='Euler_Maruyama_sampler'):
            self.sampler = Euler_Maruyama_sampler

        samples = self.sampler(
            score_model=self.score_model, 
            marginal_prob_std=self.marginal_prob_std_fn,
            diffusion_coeff=self.diffusion_coeff_fn,
            batch_size=self.batch_size,
            z=self.z,
            device=device)
        
        print(f'Samples generated of dimension {samples.shape}')

        samples = samples.clamp(0.0, 1.0)

        sample_grid = make_grid(samples, nrow=int(np.sqrt(self.batch_size)))

        plt.figure(figsize=(6,6))
        plt.axis('off')
        plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
        plt.savefig(f'{sampler}_samples.png')
            
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["mnist", "cifar10"])

    parser.add_argument("--epochs", type = int, default = 20)
    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--sigma", type = int, default = 25)
    parser.add_argument("--device", type = str, default = "cpu")

    parser.add_argument("mode", type=str, choices=["train", "sample"])

    parser.add_argument("--sampler", type=str, choices=['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'], default='Euler_Maruyama_sampler')
    
    args = parser.parse_args()

    if(args.dataset=='mnist'):
        checkpoint_path = 'mnist/ckpt.pth'
        if not os.path.exists('mnist'):
            os.makedirs('mnist')
        score = ScoreModel(args.batch_size, args.lr, args.sigma, args.device, is_RGB=False)
    elif(args.dataset=='cifar10'):
        checkpoint_path = 'cifar10/ckpt.pth'
        if not os.path.exists('cifar10'):
            os.makedirs('cifar10')
        score = ScoreModel(args.batch_size, args.lr, args.sigma, args.device, is_RGB=True)

    if(args.mode=='train'):

        if(args.dataset=='mnist'):
            dataset = MNIST('./data/', train=True, transform=transforms.ToTensor(), download=True)
        elif(args.dataset=='cifar10'):
            dataset = CIFAR10('./data/', train=True, transform=transforms.ToTensor(), download=True)

        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        score.train(data_loader, args.epochs, checkpoint_path)

    if(args.mode=='sample'):
        score.generate(checkpoint_path, args.dataset, args.sampler)
# python main.py 'cifar10' 'train' --device=='cuda/mps'
# python main.py 'cifar10' 'sampler' 