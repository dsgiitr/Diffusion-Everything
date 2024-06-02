import torch
import torch.nn as nn
import numpy as np

class GaussianFourierProjection(nn.Module):
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
      
  def forward(self, x):
    x_expanded = x[:, None]
    W_expanded = self.W[None, :]

    x_proj = x_expanded * W_expanded * 2 * np.pi

    sin_features = torch.sin(x_proj)
    cos_features = torch.cos(x_proj)

    return torch.cat([sin_features, cos_features], dim=-1)
  
class Dense(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    return self.dense(x)[..., None, None]
  
class ScoreNet1C(nn.Module):
  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    super(ScoreNet1C, self).__init__()
    
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(
        GaussianFourierProjection(embed_dim=embed_dim),
        nn.Linear(embed_dim, embed_dim)
    )
    
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

    self.tconv3 = nn.ConvTranspose2d(channels[2] * 2, channels[1], 3, stride=2, bias=False, output_padding=1)
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

    self.tconv2 = nn.ConvTranspose2d(channels[1] * 2, channels[0], 3, stride=2, bias=False, output_padding=1)
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

    self.tconv1 = nn.ConvTranspose2d(channels[0] * 2, 1, 3, stride=1)

    # Swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std

  def forward(self, x, t):
    # Input dimensions: (batch_size, 1, H, W)
    embed = self.act(self.embed(t))
    # embed: (batch_size, embed_dim)

    # Encoding path
    h1 = self.conv1(x)
    # h1: (batch_size, channels[0], H, W)
    h1 += self.dense1(embed)
    # h1: (batch_size, channels[0], H, W)
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    
    h2 = self.conv2(h1)
    # h2: (batch_size, channels[1], H/2, W/2)
    h2 += self.dense2(embed)
    # h2: (batch_size, channels[1], H/2, W/2)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)

    h3 = self.conv3(h2)
    # h3: (batch_size, channels[2], H/4, W/4)
    h3 += self.dense3(embed)
    # h3: (batch_size, channels[2], H/4, W/4)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)

    h4 = self.conv4(h3)
    # h4: (batch_size, channels[3], H/8, W/8)
    h4 += self.dense4(embed)
    # h4: (batch_size, channels[3], H/8, W/8)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)

    # Decoding path
    h = self.tconv4(h4)
    # h: (batch_size, channels[2], H/4, W/4)
    h += self.dense5(embed)
    # h: (batch_size, channels[2], H/4, W/4)
    h = self.tgnorm4(h)
    h = self.act(h)

    h = self.tconv3(torch.cat([h, h3], dim=1))
    # h: (batch_size, channels[1], H/2, W/2)
    h += self.dense6(embed)
    # h: (batch_size, channels[1], H/2, W/2)
    h = self.tgnorm3(h)
    h = self.act(h)

    h = self.tconv2(torch.cat([h, h2], dim=1))
    # h: (batch_size, channels[0], H, W)
    h += self.dense7(embed)
    # h: (batch_size, channels[0], H, W)
    h = self.tgnorm2(h)
    h = self.act(h)

    h = self.tconv1(torch.cat([h, h1], dim=1))
    # h: (batch_size, 1, H, W)

    h = h / self.marginal_prob_std(t)[:, None, None, None]
    return h
  

class ScoreNet3C(nn.Module):
  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    super(ScoreNet3C, self).__init__()
    self.embed = nn.Sequential(
        GaussianFourierProjection(embed_dim=embed_dim),
        nn.Linear(embed_dim, embed_dim)
    )
    self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, padding=1, bias=False)
    
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, padding=1, output_padding=1, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

    self.tconv3 = nn.ConvTranspose2d(channels[2] * 2, channels[1], 3, stride=2, padding=1, output_padding=1, bias=False)
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

    self.tconv2 = nn.ConvTranspose2d(channels[1] * 2, channels[0], 3, stride=2, padding=1, output_padding=1, bias=False)
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

    self.tconv1 = nn.ConvTranspose2d(channels[0] * 2, 3, 3, stride=1, padding=1)  # Adjusted for RGB output

    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std

  def forward(self, x, t):
    # Input dimensions: (batch_size, 3, H, W)
    embed = self.act(self.embed(t))
    # embed: (batch_size, embed_dim)

    # Encoding path
    h1 = self.conv1(x)
    # h1: (batch_size, channels[0], H, W)
    h1 += self.dense1(embed)
    # h1: (batch_size, channels[0], H, W)
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    
    h2 = self.conv2(h1)
    # h2: (batch_size, channels[1], H/2, W/2)
    h2 += self.dense2(embed)
    # h2: (batch_size, channels[1], H/2, W/2)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)

    h3 = self.conv3(h2)
    # h3: (batch_size, channels[2], H/4, W/4)
    h3 += self.dense3(embed)
    # h3: (batch_size, channels[2], H/4, W/4)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)

    h4 = self.conv4(h3)
    # h4: (batch_size, channels[3], H/8, W/8)
    h4 += self.dense4(embed)
    # h4: (batch_size, channels[3], H/8, W/8)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)

    # Decoding path
    h = self.tconv4(h4)
    # h: (batch_size, channels[2], H/4, W/4)
    h += self.dense5(embed)
    # h: (batch_size, channels[2], H/4, W/4)
    h = self.tgnorm4(h)
    h = self.act(h)

    h = self.tconv3(torch.cat([h, h3], dim=1))
    # h: (batch_size, channels[1], H/2, W/2)
    h += self.dense6(embed)
    # h: (batch_size, channels[1], H/2, W/2)
    h = self.tgnorm3(h)
    h = self.act(h)

    h = self.tconv2(torch.cat([h, h2], dim=1))
    # h: (batch_size, channels[0], H, W)
    h += self.dense7(embed)
    # h: (batch_size, channels[0], H, W)
    h = self.tgnorm2(h)
    h = self.act(h)

    h = self.tconv1(torch.cat([h, h1], dim=1))
    # h: (batch_size, 3, H, W)

    h = h / self.marginal_prob_std(t)[:, None, None, None]
    return h