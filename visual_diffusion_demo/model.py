import torch
from torch import nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from utils import noise_scheduler

class DiffusionConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, steps):
        super(DiffusionConvBlock, self).__init__()
        self.diffconv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding='same')
        self.activation = nn.ReLU()
        self.embedding = nn.Embedding(steps, out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, y):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        out = self.diffconv(x)
        out = self.activation(out)
        embed = self.embedding(y).unsqueeze(2)
        out = embed*out
        return out

class DiffusionConvNet(nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim, steps):
        super(DiffusionConvNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_channels

        layers = []
        in_channels = 1
        for hidden_channel in hidden_channels:
            layers.append(DiffusionConvBlock(in_channels, hidden_channel, steps))
            in_channels = hidden_channel
        layers.append(DiffusionConvBlock(in_channels, 1, steps))
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, output_dim)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam

    def forward(self, x, y):
        out = x
        for layer in self.layers:
            out = layer(out, y)
        out = self.output_layer(out)
        out = out.squeeze()
        return out

class DiffusionMLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, steps):
        super(DiffusionMLPLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.steps = steps
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.embedding = nn.Embedding(steps, output_dim)
        self.embedding.weight.data.uniform_()
    
    def forward(self, x, y):
        out = self.linear(x)
        out = self.activation(out)
        embed = self.embedding(y)
        out = embed * out
        return out

class DiffusionMLPNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, steps):
        super(DiffusionMLPNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.steps = steps

        layers = []
        in_channels = input_dim
        for hidden_dim in hidden_dims:
            layers.append(DiffusionMLPLayer(in_channels, hidden_dim, steps))
            in_channels = hidden_dim

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_channels, output_dim)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam
    
    def forward (self, x, y):
        out = x
        for layer in self.layers:
            out = layer(out, y)
        out = self.output_layer(out)
        return out 
    
    
def trainer(model, data, num_epochs, num_batches, lr, device, log_interval, logfile, progress_bar_callback = None):
    model.train()
    ns = model.ns
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    T = ns.timesteps
    for _ in range(num_epochs):
        net_epoch_loss = 0
        for s in range(0,num_batches):
            batch = data[s:data.shape[0]:num_batches].to(device)
            input_ = torch.zeros((0,batch.shape[1])).to(device)
            t_ = torch.tensor(np.repeat(np.arange(T), batch.shape[0])).to(device)
            epsilon_ = torch.zeros((0,batch.shape[1])).to(device)
            for t in range(1,T+1):
                input, epsilon = ns.forward_process(batch, (torch.ones(len(batch), device = device) * (t-1)).long())
                input_ = torch.vstack((input_, input))
                epsilon_ = torch.vstack((epsilon_, epsilon))
            optimizer.zero_grad()
            loss = F.mse_loss(model(input_, t_), epsilon_)
            loss.backward()
            optimizer.step()
            net_epoch_loss += loss.item()*batch.shape[0]/data.shape[0]
        if log_interval > 0 and (_+1)%log_interval == 0:
            f = open(logfile, 'a')
            print(f'Epoch : {_+1}, Loss : {net_epoch_loss}')
            f.write(f'Epoch : {_+1}, Loss : {net_epoch_loss}\n')
            f.close()
            if (progress_bar_callback is not None):
                progress_bar_callback((_+1)/num_epochs)
        

def model_loader(model_type, hidden_dims, dimensionality, timesteps):
    if model_type == 'mlp_diffusion':
        model = DiffusionMLPNet(dimensionality, hidden_dims, dimensionality, timesteps)
    if model_type == 'conv_diffusion':
        model = DiffusionConvNet(dimensionality, hidden_dims, dimensionality, timesteps)
    return model   

@torch.inference_mode()
def inferrer(model, n, n_dim, T, eta, repeated, model_device, streamlit_callback = None):
    model.eval()
    ns = model.ns
    x_t = torch.randn(n,n_dim, device = model_device)
    timesteps_data = torch.zeros((T+1, *(x_t.shape))).to(model_device)
    timesteps_drift = torch.zeros((T, *(x_t.shape))).to(model_device)
    dataset = torch.zeros(x_t.shape).to(model_device)
    for i in range(T,0,-1):
        timesteps_data[i] = x_t
        for j in range(repeated):
            z_t = torch.randn_like(x_t)
            beta_t = ns.beta[i-1]
            alpha_t = ns.alpha[i-1]
            alpha_bar_t = ns.alpha_cumprod[i-1]
            alpha_bar_t_1 = ns.alpha_cumprod[i-2] if i > 1 else 1


            sigma_t = ((1-alpha_bar_t_1) * beta_t/(1-alpha_bar_t)).sqrt()
            ts = (torch.ones(x_t.shape[0], device = model_device) * (i-1)).long()
            epsilon_t = model(x_t, ts).squeeze()
            mu_t = (x_t - ((1-alpha_t)/(1-alpha_bar_t).sqrt())*epsilon_t)/(alpha_t).sqrt()
            drift_t = mu_t - x_t
            timesteps_drift[i-1] = drift_t
            x_t = mu_t + eta*sigma_t*z_t   
            if streamlit_callback is not None :
                streamlit_callback(x_t, i)

    timesteps_data[0] = x_t
    dataset = x_t
    timesteps_drift = timesteps_drift.detach().cpu().numpy()
    timesteps_data = timesteps_data.detach().cpu().numpy()
    dataset = dataset.detach().cpu().numpy()
    return dataset, timesteps_data, timesteps_drift