import torch
import torchvision
import numpy as np
from torch import nn
import config
import matplotlib.pyplot as plt

def plot_single(img_batch, nrows=8):
    img_grid = torchvision.utils.make_grid(img_batch, nrows)
    img_grid = np.transpose(img_grid, (1, 2, 0))
    plt.axis("off")
    plt.imshow(img_grid)
    plt.show()

def plot_side_by_side(img_batches: list, titles: list, nrows=8):
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.subplots_adjust(wspace=1.0, hspace=1.0)

    fig, ax = plt.subplots(1, len(img_batches))
    for i in range(len(img_batches)):
        grid_i = torchvision.utils.make_grid(img_batches[i].cpu(), nrows, normalize=True) # image grid
        grid_i = np.transpose(grid_i, (1, 2, 0)) # reshape images from [num_channels, H, W] to [H, W, num_channels]
        ax[i].set_title(titles[i])
        ax[i].imshow(grid_i, cmap="gray")
        ax[i].axis("off")
    plt.show()


def viewGeneratedBatch(dl: torch.utils.data.dataloader.DataLoader, 
                  netG: nn.Module, 
                  device: torch.device):
    netG.eval() ## set to evaluation mode
    real_batch, _ = next(iter(dl))
    noise = torch.randn([real_batch.shape[0], config.nz], device=device)
    conditional_noise = torch.randint(0, config.num_classes, [real_batch.shape[0]], device=device)
    with torch.no_grad():
        fake_batch = netG(noise, conditional_noise)

    plot_side_by_side([real_batch, fake_batch], ["Batch of real images", "Batch of fake images"])    
    ## set back to training mode 
    netG.train()


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
