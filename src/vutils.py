import torch
import torchvision
import numpy as np
from torch import nn
from src.config import *
import matplotlib.pyplot as plt

def plot_single(img_batch, nrows=8):
    img_grid = torchvision.utils.make_grid(img_batch, nrows)
    img_grid = np.transpose(img_grid, (1, 2, 0))
    plt.axis("off")
    plt.imshow(img_grid)
    plt.show()

def plot_side_by_side(img_batch, netG: nn.Module, device: torch.device):
    netG.eval() ## set to evaluation mode

    plt.rcParams["figure.figsize"] = (20, 20)
    plt.subplots_adjust(wspace=1.0, hspace=1.0)

    fig, ax = plt.subplots(1, 2)

    curr_batch_size = img_batch.shape[0]
    noise = torch.randn(curr_batch_size, nz, device=device)
    conditional_noise = torch.randint(0, num_classes, (curr_batch_size, ), device=device)
    with torch.no_grad():
        fake_batch = netG(noise, conditional_noise).cpu()

    real_grid = torchvision.utils.make_grid(img_batch.cpu(), 8, normalize=True) # image grid
    real_grid = np.transpose(real_grid, (1, 2, 0)) # reshape images from [num_channels, H, W] to [H, W, num_channels]
    ax[0].set_title("Batch of real images")
    ax[0].imshow(real_grid, cmap="gray")
    ax[0].axis("off")

    fake_grid = torchvision.utils.make_grid(fake_batch, 8, normalize=True) # image grid
    fake_grid = np.transpose(fake_grid, (1, 2, 0)) # reshape images from [num_channels, H, W] to [H, W, num_channels]
    ax[1].set_title("Batch of generated images")
    ax[1].imshow(fake_grid, cmap="gray")
    ax[1].axis("off")
    
    ## set back to training mode 
    netG.train()

    plt.show()

def plot_stats(stats: list, names: list, total_epochs, nrows=2, ncols=2):
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.subplots_adjust(wspace=1.0, hspace=1.0)

    fig, ax = plt.subplots(nrows, ncols)
    fig.suptitle("Evolution of statistics over training")
    for i, stat, name in zip(range(len(stats)),
                            stats,
                            names):
        print(f"{name}:{stat}")
        ax[i // ncols, i % ncols].set_title(name)
        ax[i // ncols, i % ncols].set_xlabel("Epochs")
        ax[i // ncols, i % ncols].set_ylabel(name)
        ax[i // ncols, i % ncols].plot(range(total_epochs), stat)

    plt.show()
