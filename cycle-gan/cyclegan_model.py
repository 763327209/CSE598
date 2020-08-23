"""
Author: Hong Guan

Todo:
1. learning rate scheduler
2. detach when store in buffer?
"""
import torch

import itertools
import random
import os
import argparse
from pathlib import Path

from network import GAN_A, GAN_B, D_A, D_B
from dataset import get_dataloader
from tqdm import tqdm

from torchvision.utils import save_image

random.seed(598)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode",
                      type=str,
                      help="")
  parser.add_argument("--buffer_size",
                      type=int,
                      default=240,
                      help="Size of buffer")
  parser.add_argument("--dataset_path",
                      type=str,
                      default="datasets",
                      help="Size of buffer")
  parser.add_argument("--batch_size",
                      type=int,
                      default=64,
                      help="Batch size")
  parser.add_argument("--num_workers",
                      type=int,
                      default=40,
                      help="Number of workers to load dataset")
  parser.add_argument("--lr",
                      type=float,
                      default=0.0002,
                      help="learning rate for adam")
  parser.add_argument("--beta1",
                      type=float,
                      default=0.5,
                      help="momentum for adam.")
  parser.add_argument("--load_size",
                      type=int,
                      default=200,
                      help="Size of images")
  parser.add_argument("--train_epochs",
                      type=int,
                      default=20,
                      help="Number of training epochs.")
  parser.add_argument("--lambda_A",
                      type=float,
                      default=10.0)
  parser.add_argument("--lambda_B",
                      type=float,
                      default=10.0)
  parser.add_argument("--saved_net_path",
                      type=str,
                      default="saved_models/")
  args = parser.parse_args()
  return args


class ImageBuffer():
  def __init__(self, args):
    self.buffer_size = args.buffer_size

    self.actual_size = 0
    self.buffer = []

  def get_image(self, images):
    if self.buffer_size == 0:
      return images

    return_images = []
    for image in images:
      if self.actual_size < self.buffer_size:
        self.buffer.append(image)
        self.actual_size += 1
        return_images.append(image)
      else:
        if random.random() < 0.5:
          index = random.randrange(self.buffer_size)
          return_images.append(self.buffer[index])
          self.buffer[index] = image
        else:
          return_images.append(image)

    return torch.stack(return_images)


class CycleGANModel():
  def __init__(self, args):
    self.args = args
    if torch.cuda.is_available():
      self.device = torch.device('cuda')
      self.n_gpu = 1
    else:
      self.device = torch.device('cpu')
      self.n_gpu = 0

    # model save path
    self.saved_net_path = Path(args.saved_net_path)
    self.saved_net_path.mkdir(exist_ok=True)

    # load generate nets
    if args.mode == 'continue_train' or args.mode == 'test':
      self.gnet_A = torch.load(self.saved_net_path / 'gnet_A.bin')
      self.gnet_B = torch.load(self.saved_net_path / 'gnet_B.bin')
    else:  # for 'train' mode
      self.gnet_A = GAN_A().to(self.device)
      self.gnet_B = GAN_B().to(self.device)

    # Path of output images
    self.output_A_dir = Path('output/A/')
    self.output_B_dir = Path('output/B/')
    self.output_A_dir.mkdir(parents=True, exist_ok=True)
    self.output_B_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'train' or args.mode == 'continue_train':
      self.image_buffer_A = ImageBuffer(args)
      self.image_buffer_B = ImageBuffer(args)

      if args.mode == 'continue_train':
        self.dnet_A = torch.load(self.saved_net_path / 'dnet_A.bin')
        self.dnet_B = torch.load(self.saved_net_path / 'dnet_B.bin')
      else:
        self.dnet_A = D_A().to(self.device)
        self.dnet_B = D_B().to(self.device)

      self.gan_loss_func = torch.nn.L1Loss()
      self.cycle_loss_func = torch.nn.MSELoss()

      self.g_optim = torch.optim.Adam(
          itertools.chain(self.gnet_A.parameters(), self.gnet_B.parameters()),
          lr=args.lr, betas=(args.beta1, 0.999))
      self.d_optim = torch.optim.Adam(
          itertools.chain(self.dnet_A.parameters(), self.dnet_B.parameters()),
          lr=args.lr, betas=(args.beta1, 0.999))

  def _set_requires_grad(self, parameters, requires_grad):
    for parameter in parameters:
      parameter.requires_grad = requires_grad

  def forward(self):
    self.b_fake = self.gnet_A(self.a_real)
    self.a_rec = self.gnet_B(self.b_fake)

    self.a_fake = self.gnet_B(self.b_real)
    self.b_rec = self.gnet_A(self.a_fake)

  def train(self):
    a_loader, b_loader = get_dataloader(self.args)

    for epoch in range(self.args.train_epochs):
      print(f'Training epoch {epoch}')
      for i, (a_real, b_real) in tqdm(enumerate(zip(a_loader, b_loader))):
        # forward pass
        self.a_real = a_real[0].to(self.device)
        self.b_real = b_real[0].to(self.device)
        self.forward()

        # update generator parameters
        self.g_optim.zero_grad()
        with torch.no_grad():
          dnet_A_pred = self.dnet_A(self.b_fake)
          dnet_B_pred = self.dnet_B(self.a_fake)

        gan_loss_A = self.gan_loss_func(
            dnet_A_pred, torch.ones_like(dnet_A_pred, device=self.device))
        gan_loss_B = self.gan_loss_func(
            dnet_B_pred, torch.ones_like(dnet_B_pred, device=self.device))
        cycle_loss_A = self.cycle_loss_func(
            self.a_real, self.a_rec) * self.args.lambda_A
        cycle_loss_B = self.cycle_loss_func(
            self.b_real, self.b_rec) * self.args.lambda_B
        total_loss = gan_loss_A + gan_loss_B + cycle_loss_A + cycle_loss_B
        total_loss.backward()
        self.g_optim.step()

        # update discriminator parameters
        self.d_optim.zero_grad()
        iter_list = [(self.a_real, self.a_fake, self.dnet_B, self.image_buffer_A),
                     (self.b_real, self.b_fake, self.dnet_A, self.image_buffer_B)]
        for real, fake, dnet, image_buffer in iter_list:
          fake = fake.detach()
          fake = image_buffer.get_image(fake)
          real_pred = dnet(real)
          fake_pred = dnet(fake)

          real_loss = self.cycle_loss_func(
              real_pred, torch.ones_like(real_pred, device=self.device))
          fake_loss = self.cycle_loss_func(
              fake_pred, torch.zeros_like(fake_pred, device=self.device))
          combined_d_loss = (real_loss + fake_loss) / 2
          combined_d_loss.backward()
        self.d_optim.step()
      self.save_networks()

  def save_networks(self):
    torch.save(self.gnet_A, self.saved_net_path / 'gnet_A.bin')
    torch.save(self.gnet_B, self.saved_net_path / 'gnet_B.bin')
    torch.save(self.dnet_A, self.saved_net_path / 'dnet_A.bin')
    torch.save(self.dnet_B, self.saved_net_path / 'dnet_B.bin')

  def test(self):
    # load trained model
    gnet_A = torch.load(self.saved_net_path / 'gnet_A.bin')
    gnet_B = torch.load(self.saved_net_path / 'gnet_B.bin')

    a_loader, b_loader = get_dataloader(self.args)
    for i, (a_real, b_real) in tqdm(enumerate(zip(a_loader, b_loader))):
      with torch.no_grad():
        fake_B = self.gnet_A(a_real[0].to(self.device)).cpu().data
        fake_A = self.gnet_B(b_real[0].to(self.device)).cpu().data

      # Save image files
      save_image(fake_A, self.output_A_dir / ('%04d.png' % (i + 1)))
      save_image(fake_B, self.output_B_dir / ('%04d.png' % (i + 1)))


if __name__ == '__main__':
  args = parse_args()
  model = CycleGANModel(args)
  if args.mode == 'train' or args.mode == 'continue_train':
    model.train()
    model.save_networks()
  elif args.mode == 'test':
    model.test()
  else:
    raise NotImplementedError(f"Not such mode name {args.mode}")
