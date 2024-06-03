import torch
from torch import nn

class DiffusionConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, steps):
        super(DiffusionConvBlock, self).__init__()
        self.diffconv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding='same')
        self.activation = nn.ReLU()
        self.embedding = nn.Embedding(steps, out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, y):
        out = self.diffconv(x)
        out = self.activation(out)
        embed = self.embedding(y)[...,None]
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

    def forward(self, x, y):
        out = x
        for layer in self.layers:
            out = layer(out, y)
        out = self.output_layer(out)
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
    
    def forward (self, x, y):
        out = x
        for layer in self.layers:
            out = layer(out, y)
        out = self.output_layer(out)
        return out