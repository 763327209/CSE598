from torchvision import datasets
from torchvision import transforms
import torch


def get_dataloader(args):
  svhn_transform = transforms.Compose([
      transforms.Resize((args.load_size, args.load_size)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  mnist_transform = transforms.Compose([
      transforms.Resize((args.load_size, args.load_size)),
      transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

  if args.mode == 'train' or args.mode == 'continue_train':
    train = True
    shuffle = True
    split = 'train'
  else:
    train = False
    shuffle = False
    split = 'test'

  svhn = datasets.SVHN(root=args.dataset_path, split=split,
                       download=True, transform=svhn_transform)
  mnist = datasets.MNIST(root=args.dataset_path, train=train,
                         download=True, transform=mnist_transform)

  svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                            batch_size=args.batch_size,
                                            shuffle=shuffle,
                                            num_workers=args.num_workers)

  mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                             batch_size=args.batch_size,
                                             shuffle=shuffle,
                                             num_workers=args.num_workers)

  return mnist_loader, svhn_loader
