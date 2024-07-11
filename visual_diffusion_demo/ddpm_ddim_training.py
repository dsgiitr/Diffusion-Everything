import argparse
import torch
import numpy as np
import pandas as pd
import plotly.express as px
from torch import nn
from beta_scheduler import beta_scheduler
from data import data_loader
from model import model_loader
from visualise import visualise_data

parser = argparse.ArgumentParser(prog='DDPM/DDIM Training Program',
                                description="""The program trains a DDPM/DDIM-based Diffusion Model.
                                Using incorrect values for any of the arguments may lead to errors or erroneous results.
                                A default set of arguments is provided for convenience.""")

# args relevant to the dataset

parser.add_argument('--data', type=str, default='swissroll',
                    help="""Choose the dataset to train the model on.
                    Currently supported choices are :
                    1) swissroll (2D)
                    2) circle (2D)
                    3) polygon (2D)
                    4) donut (3D)
                    5) spring (3D)
                    6) mobius (3D)
                    7) custom (in this case, you must pass the datafile path using the --datafile argument)""")

parser.add_argument('--visualise', type=bool, default=True,
                    help="""Visualise the dataset using plotly (default : True).
                    Supported for the custom datasets if they are 2D or 3D
                    For it to evaluate to False, pass an empty string as the argument (eg. '')""")

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

parser.add_argument('--num_batches', type=int, default=1000,
                    help = """Number of batches for training the model (default : 1)""")

parser.add_argument('--device', type=str, default='cpu',
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

# args relevant to the script

parser.add_argument('--log_interval', type=int, default=1,
                    help = """Log the model loss after every log_interval epochs (default : 1)""")

parser.add_argument('--logfile', type = str, default = 'model_training_log.txt',
                    help = """Path to the logfile to log the model training progress (default : model_training_log.txt)""")

args = parser.parse_args()

# Loading the data

data = data_loader(args.data, args.data_args, args.n, args.datafile)

# Visualising the data

if args.visualise:
    visualise_data(data, "Original Dataset")

# Loading the model for training

hidden_dims = list(map(int, args.hidden_dims.split('-')))
model = model_loader(args.model, hidden_dims, data.shape[1], args.timesteps)

# Initialising the beta scheduler

beta_scheduler_ = beta_scheduler(args.beta_min, args.beta_max, args.beta_scheduler)
alpha_bar_ = beta_scheduler_.alpha_bar_schedule(args.timesteps)

# Training the model

model.trainer(data, args.num_epochs, args.num_batches, alpha_bar_, args.lr, args.device, args.log_interval, args.logfile)

# Saving the model

torch.save(model.state_dict(), args.model_path)