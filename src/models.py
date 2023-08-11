import torch
from torch import nn
from torchvision import transforms
from config import *

def data_augmentation_layers(curr_p : float):
    """Using only pixel_blitting layers for now.

    Randomization, according to the paper, should be done for each transformation"""
    x_flip = transforms.RandomHorizontalFlip(curr_p)
    rotation_90 = transforms.RandomApply([transforms.RandomRotation(90)], curr_p)
    translation = transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.3))], curr_p)

    return transforms.Compose([
        x_flip,
        rotation_90,
        translation
    ])


def _init_weights(m: torch.nn.ModuleList):
    """
        Weight initialization from 
        https://discuss.pytorch.org/t/how-to-do-weights-initialization-in-nn-modulelist/34263
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if (m.bias is not None): # has bias
            m.bias.data.fill_(0.01)


# %% [markdown]
# ## 3. Creating network

# %% [markdown]
# ### 3.1 Generator

class cGenerator(nn.Module):
    def __init__(self):
        super(cGenerator, self).__init__()

        self.all_layers = [nz + n_embed] + g_hidden_layers
        self.main = nn.ModuleList()
        for i in range(len(self.all_layers) - 1):
            in_size = self.all_layers[i]
            out_size = self.all_layers[i + 1]
            self.main.append(nn.Dropout(g_dropouts[i]))
            self.main.append(nn.Linear(in_features=in_size, out_features=out_size))
            self.main.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)) 
            self.main.append(nn.BatchNorm1d(out_size))       
        self.main.append(nn.Dropout(g_dropouts[-1]))
        self.main.append(nn.Linear(self.all_layers[-1], nc * reduced_img_size[0] * reduced_img_size[1]))
        self.main.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)) 

        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=n_embed)
        
        self.upsample = nn.ModuleList()
        self.upsample.append(nn.ConvTranspose2d(in_channels=nc, 
                                                out_channels=ngf, 
                                                kernel_size=4, 
                                                stride=2, 
                                                padding=1,
                                                bias=False))
        self.upsample.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
        self.upsample.append(nn.BatchNorm2d(ngf))
        self.upsample.append(nn.ConvTranspose2d(in_channels=ngf, 
                                                out_channels=nc, 
                                                kernel_size=4, 
                                                stride=2, 
                                                padding=1,
                                                bias=False))
        self.upsample.append(nn.Sigmoid())

        self.main.apply(_init_weights)
        self.upsample.apply(_init_weights)

    def forward(self, z, y):
        embedded_y = self.embedding(y)
        tensor = torch.cat([z, embedded_y], dim=-1)
        for layer in self.main:
            tensor = layer(tensor)
        tensor = tensor.view(-1, nc, reduced_img_size[0], reduced_img_size[1])
        for layer in self.upsample:
            tensor = layer(tensor)
        return tensor

# %% [markdown]
# ### 3.2 Discriminator

# %%
class cDiscriminator(nn.Module):
    def __init__(self):
        super(cDiscriminator, self).__init__()
        self.downsample = nn.ModuleList()
        self.downsample.append(nn.Conv2d(in_channels=nc, 
                                         out_channels=ndf, 
                                         kernel_size=5, 
                                         stride=2, 
                                         padding=2,
                                         bias=False))
        self.downsample.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
        self.downsample.append(nn.Conv2d(in_channels=ndf, 
                                         out_channels=nc, 
                                         kernel_size=5, 
                                         stride=2, 
                                         padding=2, 
                                         bias=False))
        self.downsample.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
        
        self.all_layers = [nc * reduced_img_size[0] * reduced_img_size[1] + n_embed] + d_hidden_layers
        self.main = nn.ModuleList()
        for i in range(len(self.all_layers) - 1):
            in_size = self.all_layers[i]
            out_size = self.all_layers[i + 1]
            self.main.append(nn.Dropout(d_dropouts[i]))
            self.main.append(nn.Linear(in_features=in_size, out_features=out_size))
            self.main.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
        self.main.append(nn.Dropout(d_dropouts[-1]))
        self.main.append(nn.Linear(self.all_layers[-1], 1))
        self.main.append(nn.Sigmoid())

        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=n_embed)

        self.main.apply(_init_weights)
        self.downsample.apply(_init_weights)

    def forward(self, x, y):
        enlarged_x = x
        for layer in self.downsample:
            enlarged_x = layer(enlarged_x)
        flattened_x = torch.flatten(enlarged_x, 1)
        embedded_y = self.embedding(y)
        tensor = torch.cat([flattened_x, embedded_y], dim=-1)
        for layer in self.main:
            tensor = layer(tensor)
        return tensor
