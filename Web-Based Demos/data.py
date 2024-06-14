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

def polygon(sides = 6, length = 5, n = 10000):
    t = np.linspace(0,sides,n)
    s = length*np.cot(np.pi/sides)
    k = s*np.tan(np.pi/sides)*s*(2*np.floor(t) - 1)
    x = s*np.cos(((2*np.pi)/sides)*np.floor(t)) - k*np.sin(((2*np.pi)/sides)*np.floor(t))
    y = s*np.sin(((2*np.pi)/sides)*np.floor(t)) + k*np.cos(((2*np.pi)/sides)*np.floor(t))
    data = torch.tensor(np.array([x,y])).T
    return data

def circle(radius = 4, n = 10000):
    theta = np.linspace(0,2*np.pi,n)
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
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

def spring(length = 10, radius = 4, n = 10000):
    theta = np.linspace(0,length*np.pi,n)
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    z = theta/np.pi
    data = torch.tensor(np.array([x,y,z])).T
    return data

def mobius(n = 10000):
    theta = np.linspace(0,2*np.pi, n//100)
    s = np.linspace(-1,1,100)
    theta, s = np.meshgrid(theta,s)
    theta = theta.ravel()
    s = s.ravel()
    x = np.cos(theta)+s*np.sin(theta/2)*np.cos(theta)
    y = np.sin(theta)+s*np.sin(theta/2)*np.sin(theta)
    z = s*np.cos(theta/2)
    data = torch.tensor(np.array([x,y,z])).T
    return data

def data_loader(data, data_args, n, datafile):
    if data == 'swissroll':
        if data_args is None:
            data = swissroll(n = n)
        else:
            spirals = list(map(int, data_args.split('-')))
            data = swissroll(spirals = spirals, n = n)
    
    elif data == 'donut':
        if data_args is None:
            data = donut(n = n)
        else:
            a, b = list(map(int, data_args.split('-')))
            data = donut(a = a, b = b, n = n)
    
    elif data == 'polygon':
        if data_args is None:
            data = polygon(n = n)
        else:
            sides, length = list(map(int, data_args.split('-')))
            data = polygon(sides = sides, length = length, n = n)

    elif data == 'circle':
        if data_args is None:
            data = circle(n = n)
        else:
            radius = list(map(int, data_args.split('-')))
            data = circle(radius = radius, n = n)

    elif data == 'spring':
        if data_args is None:
            data = spring(n = n)
        else:
            length, radius = list(map(int, data_args.split('-')))
            data = spring(length = length, radius = radius, n = n)

    elif data == 'mobius':
        data = mobius(n = n)

    elif data == 'custom':
        data = torch.load(datafile)
    
    else:
        raise ValueError("Invalid data type")
    
    return data