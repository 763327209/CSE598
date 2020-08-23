import torch.nn as nn
import torch.nn.functional as F
import torch


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
  """Custom deconvolutional layer for simplicity."""
  layers = []
  layers.append(nn.ConvTranspose2d(
      c_in, c_out, k_size, stride, pad, bias=False))
  if bn:
    layers.append(nn.BatchNorm2d(c_out))
  return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
  """Custom convolutional layer for simplicity."""
  layers = []
  layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
  if bn:
    layers.append(nn.BatchNorm2d(c_out))
  return nn.Sequential(*layers)


class GAN_A(nn.Module):
  """Generator for transfering from mnist to svhn"""

  def __init__(self, conv_dim=64):
    super(GAN_A, self).__init__()
    self.model = nn.Sequential(
        conv(1, conv_dim, 4), nn.LeakyReLU(0.05),
        conv(conv_dim, conv_dim * 2, 4), nn.LeakyReLU(0.05),
        conv(conv_dim * 2, conv_dim * 2, 3, 1, 1), nn.LeakyReLU(0.05),
        conv(conv_dim * 2, conv_dim * 2, 3, 1, 1), nn.LeakyReLU(0.05),
        deconv(conv_dim * 2, conv_dim, 4), nn.LeakyReLU(0.05),
        deconv(conv_dim, 3, 4, bn=False), nn.Tanh()
    )

  def forward(self, x):
    return self.model(x)


class GAN_B(nn.Module):
  """Generator for transfering from svhn to mnist"""

  def __init__(self, conv_dim=64):
    super(GAN_B, self).__init__()
    self.model = nn.Sequential(
        conv(3, conv_dim, 4), nn.LeakyReLU(0.05),
        conv(conv_dim, conv_dim * 2, 4),
        conv(conv_dim * 2, conv_dim * 2, 3, 1, 1), nn.LeakyReLU(0.05),
        conv(conv_dim * 2, conv_dim * 2, 3, 1, 1), nn.LeakyReLU(0.05),
        deconv(conv_dim * 2, conv_dim, 4), nn.LeakyReLU(0.05),
        deconv(conv_dim, 1, 4, bn=False), nn.Tanh()
    )

  def forward(self, x):
    return self.model(x)


class D_B(nn.Module):
  """Discriminator for mnist."""

  def __init__(self, conv_dim=64, use_labels=False):
    super(D_B, self).__init__()
    n_out = 11 if use_labels else 1
    self.model = nn.Sequential(
        conv(1, conv_dim, 4, bn=False), nn.LeakyReLU(0.05),
        conv(conv_dim, conv_dim * 2, 4), nn.LeakyReLU(0.05),
        conv(conv_dim * 2, conv_dim * 4, 4), nn.LeakyReLU(0.05),
        conv(conv_dim * 4, n_out, 4, 1, 0, False)
    )

  def forward(self, x):
    return self.model(x).squeeze()


class D_A(nn.Module):
  """Discriminator for svhn."""

  def __init__(self, conv_dim=64, use_labels=False):
    super(D_A, self).__init__()
    n_out = 11 if use_labels else 1
    self.model = nn.Sequential(
        conv(3, conv_dim, 4, bn=False), nn.LeakyReLU(0.05),
        conv(conv_dim, conv_dim * 2, 4), nn.LeakyReLU(0.05),
        conv(conv_dim * 2, conv_dim * 4, 4), nn.LeakyReLU(0.05),
        conv(conv_dim * 4, n_out, 4, 1, 0, False)
    )

  def forward(self, x):
    return self.model(x).squeeze()
