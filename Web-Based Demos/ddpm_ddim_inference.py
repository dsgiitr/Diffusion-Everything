import argparse
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly as plt
from plotly import graph_objects as go
from plotly import figure_factory as ff
from plotly.subplots import make_subplots
from torch import nn
from visualise import *
from model import model_loader
from beta_scheduler import beta_scheduler

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
                    Use cuda if available for faster inference
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
                    help = """Number of times to repeat the denoising process at each timestep (default : 5)
                    This is done to reduce the variance in the final dataset, albeit it is not strictly necessary for decent results""")

# args relevant to the dataset and visualisation

parser.add_argument('--dataset_path', type=str, default='ddpm_ddim_inference_dataset.csv',
                    help = """Path to save the final denoised dataset to (default : ddpm_ddim_inference_dataset.csv)""")

parser.add_argument('--num_steps', type=int, default=3,
                    help = """Starting from t = (timesteps) to t = 0 (both inclusive), a total of (num_steps) timesteps are visualised (default : 3)
                    Also try to ensure that num_steps does not exceed 5 to avoid issues where some plots may not be rendered correctly
                    and that num_steps-1 is a factor of timesteps, else the last step may not be visualised""")

# args relevant to the drift

parser.add_argument('--drift_steps', type=int, default=3,
                    help = """Visualise the drift at (drift_steps) timesteps starting from t = (timesteps) to t = 1 (both inclusive) (default : 3),
                    Try to ensure than drift_steps-1 is a factor of timesteps-1, else some timestep other than t = 1 may be visualised for the final render
                    all subsequent drift visualisation-related arguments are ignored if set to 0""")

parser.add_argument('--dim_steps', type=int, default=31,
                    help = """Number of points for each axis of the drift plot (default : 31)""")

parser.add_argument('--dim_max', type=float, default=15,
                    help = """Maximum value for each axis of the drift plot (default : 15)""")

parser.add_argument('--dim_min', type=float, default=-15,
                    help = """Minimum value for each axis of the drift plot (default : -15)""")

args = parser.parse_args()

# Loading the model for inference

hidden_dims = list(map(int, args.hidden_dims.split('-')))
model = model_loader(args.model, hidden_dims, args.n_dim, args.timesteps)
model.load_state_dict(torch.load(args.model_path, map_location = args.model_device))
model.eval()

# Initialising the beta scheduler

T = args.timesteps
beta_scheduler_ = beta_scheduler(args.beta_min, args.beta_max, args.beta_scheduler)
beta_ = beta_scheduler_.beta_schedule(T)
alpha_ = beta_scheduler_.alpha_schedule(T)
alpha_bar_ = beta_scheduler_.alpha_bar_schedule(T)

# Inferring from the model

dataset, timesteps_data, timesteps_drift = model.inferrer(args.n, args.n_dim, T, args.eta, alpha_, alpha_bar_, beta_, args.repeated, args.model_device)
dataset_ = pd.DataFrame(dataset, columns = [f"dim_{i}" for i in range(1,args.n_dim+1)])
dataset_.to_csv(args.dataset_path, index = False)

# Visualising the final dataset

visualise_data(dataset, "Diffusion-Generated Dataset")

# Visualising the reverse diffusion process

if (args.num_steps > 0):
    visualise_reverse_diffusion(timesteps_data, args.num_steps)

# Visualising the drift at each time_step

if (args.drift_steps > 0):
    visualise_reverse_drift(args, model, alpha_, alpha_bar_)