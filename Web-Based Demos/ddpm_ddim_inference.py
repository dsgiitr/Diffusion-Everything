import argparse
import data, model
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly as plt
from plotly import graph_objects as go
from plotly import figure_factory as ff
from plotly.subplots import make_subplots
from torch import nn

parser = argparse.ArgumentParser(prog='DDPM/DDIM Inference Program',
                                description="""The program infers from a trained DDIM-based Diffusion Model.
                                Using incorrect values for any of the arguments may lead to errors or erroneous results.
                                Ensure your arguments are consistent with the training arguments
                                A default set of arguments is provided for convenience.""")

# args relevant to the dataset

parser.add_argument('--n_dim', type=int, default=2,
                    help="""Choose the dimensions of the data, for instance if it is 2D or 3D (default : 2)""")

parser.add_argument('--n', type=int, default=10000,
                    help="""Number of samples to generate for the dataset, if custom is not used (default : 10000)""")

# args relevant to the model

parser.add_argument('--model', type=str, default='mlp_diffusion',
                    help = """Choose the model which was trained on the data.
                    Currently supported choices are :
                    1) mlp_diffusion
                    2) conv_diffusion""")

parser.add_argument('--hidden_dims', type=str, default='32-32',
                    help = """Hidden dimensions for the MLP model and channels for the CNN model.
                    Pass as a string of hyhen separated integers (default : 32-32)""")

parser.add_argument('--model_device', type=str, default='cpu',
                    help = """Device for inferring the model on (default : cpu)
                    Use cuda if available for faster training
                    Avoid using mps due to issues with the PyTorch MPS backend""")

parser.add_argument('--model_path', type=str, default='ddpm_ddim_diffusion_model.pt',
                    help = """Path to load the trained model from (default : ddim_diffusion_model.pt)""")

# args relevant to the diffusion process

parser.add_argument('--timesteps', type=int, default=100,
                    help = """Number of timesteps to run the diffusion process for (default : 100)""")

parser.add_argument('--eta', type=float, default=1.0,
                    help = """Choice for the range between DDIM (eta = 0.0) and DDPM (eta = 1.0) (default : 1.0)""")

parser.add_argument('--beta_scheduler', type=str, default='linear',
                    help = """Scheduler to use for diffusion. Currently supported choices are :
                    1) linear
                    2) quadratic""")

parser.add_argument('--beta_min', type=float, default=0.0001,
                    help = """Initial value of beta (default : 0.0001)""")

parser.add_argument('--beta_max', type=float, default=0.1,
                    help = """Final value of beta (default : 0.1)""")

parser.add_argument('--repeated', type=int, default=5,
                    help = """Number of times to repeat the denoising process at each timestep (default : 5)""")

# args relevant to the drift

parser.add_argument('--visualise_drift', type=bool, default=True,
                    help = """Visualise the drift at each timestep (default : True),
                    all subsequent drift visualisation-related arguments are ignored if set to False""")

parser.add_argument('--dim_steps', type=int, default=31,
                    help = """Number of points for each axis of the drift plot (default : 31)""")

parser.add_argument('--dim_max', type=float, default=15,
                    help = """Maximum value for each axis of the drift plot (default : 15)""")

parser.add_argument('--dim_min', type=float, default=-15,
                    help = """Minimum value for each axis of the drift plot (default : -15)""")

args = parser.parse_args()

# Loading the model for inference

if args.model == 'mlp_diffusion':
    from model import DiffusionMLPNet
    hidden_dims = list(map(int, args.hidden_dims.split('-')))
    model = DiffusionMLPNet(args.n_dim, hidden_dims, args.n_dim, args.timesteps)
    model.load_state_dict(torch.load(args.model_path))

if args.model == 'conv_diffusion':
    from model import DiffusionConvNet
    hidden_channels = list(map(int, args.hidden_dims.split('-')))
    model = DiffusionConvNet(args.n_dim, hidden_channels, args.n_dim, args.timesteps)
    model.load_state_dict(torch.load(args.model_path))


# Initialising the beta scheduler

T = args.timesteps
beta_1 = args.beta_min
beta_T = args.beta_max
kind = args.beta_scheduler

def beta_schedule(t, T = T, beta_1 = beta_1, beta_T = beta_T, kind = kind):
    if (kind == 'linear'):
        beta = beta_1 + (beta_T - beta_1)*(t-1)/(T-1)
    if (kind == 'quadratic'):
        beta = beta_1 + (beta_T - beta_1)*((t-1)/(T-1))**2
    return beta

def alpha_schedule(t, T = T, beta_1 = beta_1, beta_T = beta_T, kind = kind):
    alpha = 1 - beta_schedule(t, T, beta_1, beta_T, kind)
    return alpha

def alpha_bar_schedule(t, T = T, beta_1 = beta_1, beta_T = beta_T, kind = kind):
    alpha_bar = 1
    for i in range(1, t+1):
        alpha_bar *= alpha_schedule(i, T, beta_1, beta_T, kind)
    return alpha_bar

beta_ = [beta_schedule(t, T) for t in range(1, T+1)]
alpha_ = [alpha_schedule(t, T) for t in range(1, T+1)]
alpha_bar_ = [alpha_bar_schedule(t, T) for t in range(1, T+1)]

# Inferring from the model

if args.model_device != 'cpu':
    model.cpu()

model.eval()
x_t = torch.randn(args.n,args.n_dim)
timesteps_data = torch.zeros((T+1, *(x_t.shape)))
#timesteps_drift = torch.zeros((T, *(x_t.shape)))
dataset = torch.zeros(x_t.shape).to(args.model_device)
for i in range(T,0,-1):
    timesteps_data[i] = x_t
    for j in range(args.repeated):
        z_t = torch.randn_like(x_t)
        beta_t = beta_[i-1]
        alpha_t = alpha_[i-1]
        alpha_bar_t = alpha_bar_[i-1]
        alpha_bar_t_1 = alpha_bar_[i-2] if i > 1 else 1
        sigma_t = np.sqrt((1-alpha_bar_t_1) * beta_t/(1-alpha_bar_t))
        t_ = torch.tensor([i-1]*x_t.shape[0])
        epsilon_t = model(x_t, t_).squeeze()
        mu_t = (x_t - ((1-alpha_t)/np.sqrt(1-alpha_bar_t))*epsilon_t)/np.sqrt(alpha_t)
        drift_t = mu_t - x_t
        #timesteps_drift[i-1] = drift_t
        x_t = mu_t + args.eta*sigma_t*z_t         
timesteps_data[0] = x_t
dataset = x_t
timesteps_data = timesteps_data.detach().numpy()
dataset = dataset.detach().numpy()

# Visualising the reverse diffusion process

if (args.n_dim == 2):
    fig = make_subplots(rows = T//4+1, cols = 4, subplot_titles = ['Timestep '+str(i) for i in range(T,-1,-1)])
    for i in range(T,-1,-1):
        figtemp = go.Scatter(x=timesteps_data[i][:,0], y=timesteps_data[i][:,1],
                             mode = 'markers', name = 'Timestep '+str(i),
                             marker = dict(colorscale='viridis',
                                           color = np.sqrt(timesteps_data[i][:,0]**2 + timesteps_data[i][:,1]**2)))
        fig.add_trace(figtemp, row = (T-i)//4 + 1, col = (T-i)%4 + 1)

if (args.n_dim == 3):
    fig = make_subplots(rows = T//4+1, cols = 4, subplot_titles = ['Timestep '+str(i) for i in range(T,-1,-1)],
                        specs = [[{"type": "scatter3d"} for _ in range(0, 4)] for _ in range(0, T//4+1)])
    for i in range(T,-1,-1):
        figtemp = go.Scatter3d(x=timesteps_data[i][:,0], y=timesteps_data[i][:,1], z=timesteps_data[i][:,2],
                               mode = 'markers', name = 'Timestep '+str(i),
                               marker = dict(colorscale='viridis',
                                             color = np.sqrt(timesteps_data[i][:,0]**2 + timesteps_data[i][:,1]**2 + timesteps_data[i][:,2]**2)))
        fig.add_trace(figtemp, row = (T-i)//4 + 1, col = (T-i)%4 + 1)

fig.update_layout(title_text = 'Reverse Diffusion Process',
                  height = (T//4+1)*400, width = 1600)

fig.show()

# Visualising the drift at each time_step

if (args.visualise_drift == False):
    exit(0)



if (args.n_dim == 2):
    fig = make_subplots(rows = (T+3)//4, cols = 4, subplot_titles = ['Drift at Timestep '+str(i) for i in range(T,0,-1)])
    dim_grid = np.linspace(args.dim_min,args.dim_max,args.dim_steps)
    X, Y = np.meshgrid(dim_grid, dim_grid)
    grid_coords = np.array((X.ravel(),Y.ravel())).T
    x_t = torch.Tensor(grid_coords)
    for i in range(T,0,-1):
        alpha_t = alpha_[i-1]
        alpha_bar_t = alpha_bar_[i-1]
        t_ = torch.tensor([i-1]*x_t.shape[0])
        epsilon_t = model(x_t, t_).squeeze()
        mu_t = (x_t - ((1-alpha_t)/np.sqrt(1-alpha_bar_t))*epsilon_t)/np.sqrt(alpha_t)
        drift_t = mu_t - x_t
        drift_t = drift_t.detach().numpy()
        ffquiver = ff.create_quiver(x = grid_coords[:,0], y = grid_coords[:,1],
                                    u = drift_t[:,0], v = drift_t[:,1], scale = 1)
        for d in ffquiver.data:
            fig.add_trace(go.Scatter(x=d['x'], y=d['y'],
                                     name = 'Drift at Timestep '+str(i)),
                                     row=(T-i)//4 + 1, col = (T-i)%4 + 1)
if (args.n_dim == 3):
    fig = make_subplots(rows = (T+3)//4, cols = 4, subplot_titles = ['Drift at Timestep '+str(i) for i in range(T,0,-1)],
                        specs = [[{"type": "scatter3d"} for _ in range(0, 4)] for _ in range(0, (T+3)//4)])
    dim_grid = np.linspace(args.dim_min,args.dim_max,args.dim_steps)
    X, Y, Z = np.meshgrid(dim_grid, dim_grid, dim_grid)
    grid_coords = np.array((X.ravel(),Y.ravel(),Z.ravel())).T
    x_t = torch.Tensor(grid_coords)
    for i in range(T,0,-1):
        alpha_t = alpha_[i-1]
        alpha_bar_t = alpha_bar_[i-1]
        t_ = torch.tensor([i-1]*x_t.shape[0])
        epsilon_t = model(x_t, t_).squeeze()
        mu_t = (x_t - ((1-alpha_t)/np.sqrt(1-alpha_bar_t))*epsilon_t)/np.sqrt(alpha_t)
        drift_t = mu_t - x_t
        drift_t = drift_t.detach().numpy()
        fig.add_trace(go.Cone(x = grid_coords[:,0], y = grid_coords[:,1], z = grid_coords[:,2],
                              u = drift_t[:,0], v = drift_t[:,1], w = drift_t[:,2]),
                              row=(T-i)//4 + 1, col = (T-i)%4 + 1)

fig.update_layout(title_text = 'Reverse Diffusion Process Drifts',
                    height = ((T+3)//4)*400, width = 1600,
                    showlegend = False)
fig,show()