import argparse
import data, model
import torch
import numpy as np
import pandas as pd
import plotly.express as px
from torch import nn

parser = argparse.ArgumentParser(prog='DDPM/DDIM Training Program',
                                description="""The program trains a DDPM/DDIM-based Diffusion Model.
                                Using incorrect values for any of the arguments may lead to errors or erroneous results.
                                A default set of arguments is provided for convenience.""")

# args relevant to the dataset

parser.add_argument('--data', type=str, default='swissroll',
                    help="""Choose the dataset to train the model on.
                    Currently supported choices are :
                    1) swissroll
                    2) donut
                    3) custom (in this case, you must pass the datafile path using the --datafile argument)""")

parser.add_argument('--visualise', type=bool, default=True,
                    help="""Visualise the dataset using plotly (default : True).
                    Supported for the custom datasets if they are 2D or 3D""")

parser.add_argument('--n', type=int, default=10000,
                    help="""Number of samples to generate for the dataset, if custom is not used (default : 10000)""")

parser.add_argument('--data_args', type=str, default=None,
                    help="""Arguments to pass to the data generation function if custom is not used.
                    Pass as a string of hyphen separated integers if multiple, the syntax is as follows:
                    For swissroll, pass the number of spirals only (default : 2)
                    For donut, pass the inner radius of the donut and the outer radius of the donut's middle (default : 1-4)
                    Ignored if custom dataset is used""")

parser.add_argument('--datafile', type=str, default=None,
                    help="""Path to the custom datafile. Required if the data argument is set to custom, ignored otherwise""")

# args relevant to the model

parser.add_argument('--model', type=str, default='mlp_diffusion',
                    help = """Choose the model to train on the data.
                    Currently supported choices are :
                    1) mlp_diffusion
                    2) conv_diffusion""")

parser.add_argument('--hidden_dims', type=str, default='32-32',
                    help = """Hidden dimensions for the MLP model and channels for the CNN model.
                    Pass as a string of hyhen separated integers (default : 32-32)""")

parser.add_argument('--num_epochs', type=int, default=1000,
                    help = """Number of epochs to train the model for (default : 1000)""")

parser.add_argument('--lr', type=float, default=0.01,
                    help = """Learning rate for the optimizer (default : 0.01)""")

parser.add_argument('--model_device', type=str, default='cpu',
                    help = """Device to train the model on (default : cpu)
                    Use cuda if available for faster training
                    Avoid using mps due to issues with the PyTorch MPS backend""")

parser.add_argument('--model_path', type=str, default='ddpm_ddim_diffusion_model.pt',
                    help = """Path to save the trained model (default : ddpm_ddim_diffusion_model.pt)""")

# args relevant to the diffusion process

parser.add_argument('--timesteps', type=int, default=100,
                    help = """Number of timesteps to run the diffusion process for (default : 100)""")

parser.add_argument('--beta_scheduler', type=str, default='linear',
                    help = """Scheduler to use for diffusion. Currently supported choices are :
                    1) linear
                    2) quadratic""")

parser.add_argument('--beta_min', type=float, default=0.0001,
                    help = """Initial value of beta (default : 0.0001)""")

parser.add_argument('--beta_max', type=float, default=0.1,
                    help = """Final value of beta (default : 0.1)""")

args = parser.parse_args()

# Loading the data

if args.data == 'swissroll':
    from data import swissroll
    if args.data_args is None:
        data = swissroll()
    else:
        spirals = list(map(int, args.data_args.split('-')))
        data = swissroll(spirals = spirals, n = args.n)

if args.data == 'donut':
    from data import donut
    if args.data_args is None:
        data = donut()
    else:
        a, b = list(map(int, args.data_args.split('-')))
        data = donut(a = a, b = b, n = args.n)

if args.data == 'custom':
    data = torch.load(args.datafile)

# Visualising the data

if args.visualise:
    if data.shape[1] == 2:
        fig = px.scatter(x = data[:,0], y = data[:,1], color = (np.sqrt(data[:,0]**2 + data[:,1]**2)), color_continuous_scale='viridis')
        fig.show()
    if data.shape[1] == 3:
        fig = px.scatter_3d(x = data[:,0], y = data[:,1], z = data[:,2], color = (np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)), color_continuous_scale='viridis')
        fig.show()

# Loading the model for training

if args.model == 'mlp_diffusion':
    from model import DiffusionMLPNet
    hidden_dims = list(map(int, args.hidden_dims.split('-')))
    model = DiffusionMLPNet(data.shape[1], hidden_dims, data.shape[1], args.timesteps)

if args.model == 'conv_diffusion':
    from model import DiffusionConvNet
    hidden_channels = list(map(int, args.hidden_dims.split('-')))
    model = DiffusionConvNet(data.shape[1], hidden_channels, data.shape[1], args.timesteps)


# Initialising the beta scheduler

T = args.timesteps
beta_1 = args.beta_min
beta_T = args.beta_max
kind = args.beta_scheduler

def beta_schedule(t, T = 100, beta_1 = beta_1, beta_T = beta_T, kind = kind):
    if (kind == 'linear'):
        beta = beta_1 + (beta_T - beta_1)*(t-1)/(T-1)
    if (kind == 'quadratic'):
        beta = beta_1 + (beta_T - beta_1)*((t-1)/(T-1))**2
    return beta

def alpha_schedule(t, T = 100, beta_1 = beta_1, beta_T = beta_T, kind = kind):
    alpha = 1 - beta_schedule(t, T, beta_1, beta_T, kind)
    return alpha

def alpha_bar_schedule(t, T = 100, beta_1 = beta_1, beta_T = beta_T, kind = kind):
    alpha_bar = 1
    for i in range(1, t+1):
        alpha_bar *= alpha_schedule(i, T, beta_1, beta_T, kind)
    return alpha_bar

beta_ = [beta_schedule(t, T) for t in range(1, T+1)]
alpha_ = [alpha_schedule(t, T) for t in range(1, T+1)]
alpha_bar_ = [alpha_bar_schedule(t, T) for t in range(1, T+1)]

# Training the model 
if args.model_device == 'cuda':
    model = model.to(args.model_device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch = data.to(torch.float32).to(args.model_device)
    for _ in range(args.num_epochs):
        input_ = torch.zeros((0,data.shape[1]))
        t_ = torch.tensor(np.repeat(np.arange(T), data.shape[0])).to(args.model_device)
        epsilon_ = torch.zeros((0,data.shape[1]))
        for t in range(1,T+1):
            epsilon = torch.randn_like(batch)
            alpha_bar_t = alpha_bar_[t-1]
            input = (np.sqrt(alpha_bar_t)*batch + np.sqrt(1-alpha_bar_t)*epsilon)
            input_ = torch.vstack((input_, input))
            epsilon_ = torch.vstack((epsilon_, epsilon))
        optimizer.zero_grad()
        loss = criterion(model(input_.to(args.model_device), t_), epsilon_)
        loss.backward()
        optimizer.step()
        print(f"Loss at epoch {_+1} is : ", loss.item())
if args.model_device == 'cpu':
    model = model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch = data.to(torch.float32)
    for _ in range(args.num_epochs):
        input_ = torch.zeros((0,data.shape[1]))
        t_ = torch.tensor(np.repeat(np.arange(T), data.shape[0]))
        epsilon_ = torch.zeros((0,data.shape[1]))
        for t in range(1,T+1):
            epsilon = torch.randn_like(batch)
            alpha_bar_t = alpha_bar_[t-1]
            input = (np.sqrt(alpha_bar_t)*batch + np.sqrt(1-alpha_bar_t)*epsilon)
            input_ = torch.vstack((input_, input))
            epsilon_ = torch.vstack((epsilon_, epsilon))
        optimizer.zero_grad()
        loss = criterion(model(input_, t_), epsilon_)
        loss.backward()
        optimizer.step()
        print(f"Loss at epoch {_+1} is : ", loss.item())

# Saving the model

torch.save(model.state_dict(), args.model_path)