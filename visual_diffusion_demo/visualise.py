import torch
import plotly as plt
import plotly.express as px
import numpy as np
from plotly import graph_objects as go
from plotly import figure_factory as ff
from plotly.subplots import make_subplots

def visualise_data(data, title, show = True):
    if data.shape[1] == 2:
        fig = px.scatter(x = data[:,0], y = data[:,1], color = (np.sqrt(data[:,0]**2 + data[:,1]**2)), color_continuous_scale='viridis')
    if data.shape[1] == 3:
        fig = px.scatter_3d(x = data[:,0], y = data[:,1], z = data[:,2], color = (np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)), color_continuous_scale='viridis')
    if (show):
        fig.show()
    return fig

def visualise_reverse_diffusion(timesteps_data, num_steps, show = True, streamlit_callback = None):
    T = timesteps_data.shape[0]-1
    k = T//(num_steps-1)

    if (timesteps_data.shape[-1] == 2):
        fig = make_subplots(rows = (num_steps+1)//2, cols = 2, subplot_titles = ['Timestep '+str(i) for i in range(T,-1,-k)])
        for j, i in enumerate(range(T,-1,-k)):
            figtemp = go.Scatter(x=timesteps_data[i][:,0], y=timesteps_data[i][:,1],
                                mode = 'markers', name = 'Timestep '+str(i),
                                marker = dict(colorscale='viridis',
                                            color = np.sqrt(timesteps_data[i][:,0]**2 + timesteps_data[i][:,1]**2)))
            fig.add_trace(figtemp, row = j//2+1, col = j%2+1)

    if (timesteps_data.shape[-1] == 3):
        fig = make_subplots(rows = (num_steps+1)//2, cols = 2, subplot_titles = ['Timestep '+str(i) for i in range(T,-1,-k)],
                            specs = [[{"type": "scatter3d"} for _ in range(0, 2)] for _ in range(0, T//(2*k)+1)])
        for j, i in enumerate(range(T,-1,-k)):
            figtemp = go.Scatter3d(x=timesteps_data[i][:,0], y=timesteps_data[i][:,1], z=timesteps_data[i][:,2],
                                mode = 'markers', name = 'Timestep '+str(i),
                                marker = dict(colorscale='viridis',
                                                color = np.sqrt(timesteps_data[i][:,0]**2 + timesteps_data[i][:,1]**2 + timesteps_data[i][:,2]**2)))
            fig.add_trace(figtemp, row = j//2+1, col = j%2+1)
    
    fig.update_layout(title_text = 'Reverse Diffusion Process',
                  height = ((num_steps+1)//2)*400, width = 1600)

    if (show):
        plt.offline.plot(fig, filename = 'reverse_diffusion_path.html')
    
    return fig


def visualise_reverse_drift(args, model, alpha_, alpha_bar_, device, show = True, streamlit_callback = None):
    T = args.timesteps
    k = (T-1)//(args.drift_steps-1)
    if (args.n_dim == 2):
        fig = make_subplots(rows = (args.drift_steps+1)//2, cols = 2, subplot_titles = ['Drift at Timestep '+str(i) for i in range(T,0,-k)],
                            shared_xaxes = True, shared_yaxes = True)
        dim_grid = np.linspace(args.dim_min,args.dim_max,args.dim_steps)
        X, Y = np.meshgrid(dim_grid, dim_grid)
        grid_coords = np.array((X.ravel(),Y.ravel())).T
        x_t = torch.Tensor(grid_coords).to()
        for j,i in enumerate(range(T,0,-k)):
            alpha_t = alpha_[i-1]
            alpha_bar_t = alpha_bar_[i-1]
            t_ = torch.tensor([i-1]*x_t.shape[0])
            epsilon_t = model(x_t.to(device), t_.to(device)).squeeze().cpu()
            mu_t = (x_t - ((1-alpha_t)/np.sqrt(1-alpha_bar_t))*epsilon_t)/np.sqrt(alpha_t)
            drift_t = mu_t - x_t
            drift_t = drift_t.detach().numpy()
            ffquiver = ff.create_quiver(x = grid_coords[:,0], y = grid_coords[:,1],
                                        u = drift_t[:,0], v = drift_t[:,1], scale = 1)
            if streamlit_callback is not None :
                streamlit_callback(ffquiver, i)
    if (args.n_dim == 3):
        fig = make_subplots(rows = (args.drift_steps+1)//2, cols = 2, subplot_titles = ['Drift at Timestep '+str(i) for i in range(T,0,-k)],
                            specs = [[{"type": "scatter3d"} for _ in range(0, 2)] for _ in range(0, (args.drift_steps+1)//2)])
        dim_grid = np.linspace(args.dim_min,args.dim_max,args.dim_steps)
        X, Y, Z = np.meshgrid(dim_grid, dim_grid, dim_grid)
        grid_coords = np.array((X.ravel(),Y.ravel(),Z.ravel())).T
        x_t = torch.Tensor(grid_coords)
        for j,i in enumerate(range(T,0,-k)):
            alpha_t = alpha_[i-1]
            alpha_bar_t = alpha_bar_[i-1]
            t_ = torch.tensor([i-1]*x_t.shape[0])
            epsilon_t = model(x_t, t_).squeeze()
            mu_t = (x_t - ((1-alpha_t)/np.sqrt(1-alpha_bar_t))*epsilon_t)/np.sqrt(alpha_t)
            drift_t = mu_t - x_t
            drift_t = drift_t.detach().numpy()
            fig.add_trace(go.Cone(x = grid_coords[:,0], y = grid_coords[:,1], z = grid_coords[:,2],
                                u = drift_t[:,0], v = drift_t[:,1], w = drift_t[:,2], name = 'Drift at Timestep '+str(i)),
                                row = j//2+1, col = j%2+1)
    
    fig.update_layout(title_text = 'Reverse Diffusion Process Drifts',
                    height = ((args.drift_steps+1)//2)*400, width = 1600)

    if (show):
        plt.offline.plot(fig, filename = 'reverse_diffusion_drifts.html')
    
    return fig  