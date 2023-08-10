import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
import os
# %%
# parameters
# root = "/home/david/datasets/pan-radiographs"
root = "mnist"
img_size = (224, 224)
batch_size = 25
workers = 0

# %%
reduced_img_size = [56, 56]
num_classes = 10
nc = 1 # num channels
nz = 100
n_embed = 10
leaky_slope = 0.2
lr = 0.0002
betas=(0.5, 0.999)
g_hidden_layers = [128, 512, 1024, 2048]
g_dropouts = [0.3, 0.3, 0.3, 0.2, 0.2]
d_hidden_layers = [512, 1024, 512, 1024, 256]
d_dropouts = [0.2, 0.3, 0.2, 0.3, 0.2, 0.2]
ndf = 32
ngf = 32
# criterion = nn.BCELoss() # y log x + (1 - y) log (1 - x) where x,y are of shape (batch_size,)
criterion = nn.MSELoss()

### ADA
p = 0.0
ada_N = 5  ## number of minibatches after which update probability of random data augmentation transforms
delta_p = 0.02
max_p = 0.65
min_p = 0.0
target_rt = 0.6  ## reference RT for measuring overfitting
### Model checkpoint setup

# %%
GEN_PATH = "weights/cGAN/generator-MSE.pth"
DISC_PATH = "weights/cGAN/discriminator-MSE.pth"

real_folder = "imgs-real"
fake_folder = "imgs-fake"
real_stats_file = "real-stats.npz"
fake_stats_file = "fake-stats.npz"

dataset = datasets.MNIST(annotations_file = os.path.join(root, "pan-radiographs.csv"),
                            img_dir = os.path.join(root, "1st-set/images"),
                           transform=transforms.Compose([
                               # transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Resize(img_size),
                               # transforms.Normalize([0.5], [0.5]), # normalize single channel
                               transforms.Grayscale()
                           ]))
