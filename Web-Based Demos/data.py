import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.express as px

def swissroll(spirals = 2, n = 10000):
    theta = np.linspace(0,2*spirals*np.pi,(int)(1e5))
    r = 1 + theta
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    data = torch.tensor(np.array([x,y])).T
    return data

def donut(a = 1, b = 4, n = 10000):
    theta = np.linspace(0,2*np.pi,np.sqrt(n).astype(int))
    phi = np.linspace(0,2*np.pi, np.sqrt(n).astype(int))
    theta, phi = np.meshgrid(theta,phi)
    grid_coords = np.array((theta.ravel(),phi.ravel())).T
    x = (a*np.cos(grid_coords[:,0]) + b)*np.cos(grid_coords[:,1])
    y = (a*np.cos(grid_coords[:,0]) + b)*np.sin(grid_coords[:,1])
    z = a*np.sin(grid_coords[:,0])
    data = torch.tensor(np.array([x,y,z])).T
    return data