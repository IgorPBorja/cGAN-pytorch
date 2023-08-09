# %% [markdown]
# # Conditional GAN (cGAN)

# %% [markdown]
# ## 0. Imports e FID

# %% [markdown]
# ### 0.1 Imports

# %%
import torch
import torchvision
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import time

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print(f"Running on {device}")

# %% [markdown]
# ### 0.2 FID

# %% [markdown]
# * Código obtido de https://pytorch.org/hub/pytorch_vision_inception_v3/
# * O seguinte link menciona como converter de Grayscale para o RGB visualmente equivalente (https://discuss.pytorch.org/t/grayscale-to-rgb/94639/2)
# * Implementação do FID retirada de https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

# %%
## code taken from https://pytorch.org/hub/pytorch_vision_inception_v3/
model_inceptionV3 = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT).to(device)
model_inceptionV3.eval() # evaluation/inference mode

preprocess = transforms.Compose([
    transforms.Lambda(lambda img: img.expand(-1, 3, -1, -1)),  # Acts on batch. From grayscale to RGB with R=G=B = original channel intensity
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
# remove fc layers
model_inceptionV3._modules["dropout"] = nn.Identity()
model_inceptionV3._modules["fc"] = nn.Identity()

print(f"Total parameters: {sum(x.numel() for x in model_inceptionV3.parameters() if x.requires_grad)}")

# %%
from scipy.linalg import sqrtm # principal square root of positive (semi)definite matrix

def FID_score(real_batch: torch.Tensor, fake_batch: torch.Tensor):
    act1 = preprocess(real_batch).to(device)
    act2 = preprocess(fake_batch).to(device)

    ## inference only
    with torch.no_grad():
        act1 = model_inceptionV3(act1)
        act2 = model_inceptionV3(act2)

    ## pass to cpu and then convert to numpy array
    act1 = act1.cpu().detach().numpy()
    act2 = act2.cpu().detach().numpy()

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    mean_dist_square = np.sum((mu1 - mu2)**2)
    sqrt_cov = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(sqrt_cov):
        sqrt_cov = sqrt_cov.real

    return mean_dist_square + np.trace(sigma1 + sigma2 - 2.0 * sqrt_cov)

# %% [markdown]
# ## 1. Parameters

# %%
# parameters
# root = "/home/igor/IC-iVision/CIFAR10"
root = "/home/igor/IC-iVision/MNIST"
img_size = (28, 28)
batch_size = 25
workers = 0

# %%
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
# criterion = nn.BCELoss() # y log x + (1 - y) log (1 - x) where x,y are of shape (batch_size,)
criterion = nn.MSELoss()

### ADA
p = 0.0
ada_N = 5  ## number of minibatches after which update probability of random data augmentation transforms
delta_p = 0.02
max_p = 0.65
min_p = 0.0
target_rt = 0.6  ## reference RT for measuring overfitting

# %% [markdown]
# ## 2. Loading

# %% [markdown]
# ### 2.1 Standard dataloader

# %%
## loading

## use only training images to make loading easier
dataset = datasets.MNIST(root=root, train=True,
                           transform=transforms.Compose([
                               # transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Resize(img_size),
                               # transforms.Normalize([0.5], [0.5]), # normalize single channel
                               transforms.Grayscale()
                           ]),
                           download=True)

print(f"Loaded {len(dataset)} images of {len(dataset.classes)} classes")
print("Classes:", *dataset.classes, sep="\n\t")

dl = dataloader.DataLoader(dataset, batch_size=batch_size,
                           shuffle=True,
                           num_workers=workers)

# %% [markdown]
# ### 2.2 Plotting one batch

# %%
img_batch, labels = next(iter(dl))

# plt.rcParams["figure.figsize"] = (20, 20)

img_grid = vutils.make_grid(img_batch, 8)
img_grid = np.transpose(img_grid, (1, 2, 0))
plt.axis("off")

plt.imshow(img_grid)
plt.show()

print(f"FID score of {FID_score(img_batch, img_batch)}. Expected 0.0.") ## for FID testing purposes

# %% [markdown]
# ### 2.3 Data augmentation

# %% [markdown]
# * Ver o artigo [Training Generative Adversarial Networks with
# Limited Data](https://arxiv.org/abs/2006.06676) para uma descrição mais detalhada do método ADA (Adaptive Discriminator Augmentation), bem como um _pipeline_ de transformações de imagens mais completo que o seguinte (que só inclui transformaçoẽs envolvendo _pixel blitting_, movimentação de pixels).

# %%
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

augment_transforms = data_augmentation_layers(p) ## First/base instance of data augmentation layers

# %% [markdown]
# ## 3. Creating network

# %% [markdown]
# ### 3.1 Generator

# %%
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
        self.main.append(nn.Linear(self.all_layers[-1], nc * img_size[0] * img_size[1]))
        self.main.append(nn.Sigmoid())

        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=n_embed)

    def forward(self, z, y):
        embedded_y = self.embedding(y)
        tensor = torch.cat([z, embedded_y], dim=-1)
        for layer in self.main:
            tensor = layer(tensor)
        return tensor.view(-1, nc, img_size[0], img_size[1])

# %%
netG = cGenerator().to(device) # generator on GPU

optimG = optim.Adam(params=netG.parameters(), lr=lr, betas=betas)

# %%
print(netG)
print(f"Total parameters: {sum(x.numel() for x in netG.parameters() if x.requires_grad)}")

# %% [markdown]
# ### 3.2 Discriminator

# %%
class cDiscriminator(nn.Module):
    def __init__(self):
        super(cDiscriminator, self).__init__()
        self.all_layers = [nc * img_size[0] * img_size[1] + n_embed] + d_hidden_layers
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

    def forward(self, x, y):
        flattened_x = torch.flatten(x, 1)
        embedded_y = self.embedding(y)
        tensor = torch.cat([flattened_x, embedded_y], dim=-1)
        for layer in self.main:
            tensor = layer(tensor)
        return tensor

# %%
netD = cDiscriminator().to(device) # discriminator on GPU

optimD = optim.Adam(params=netD.parameters(), lr=lr, betas=betas)

# %%
print(netD)
print(f"Total parameters: {sum(x.numel() for x in netD.parameters() if x.requires_grad)}")

# %% [markdown]
# ## 4 Training

# %% [markdown]
# ### 4.1 Some definitions

# %% [markdown]
# #### 4.1.1 Plotting function (for visual comparison)

# %%
def plot_side_by_side(img_batch, labels_batch, netG: nn.Module):
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

# %% [markdown]
# #### 4.1.2 Model checkpoint setup

# %%
best_fid = float('inf')
GEN_PATH = "weights/cGAN/generator-MNIST-MSE.pth"
DISC_PATH = "weights/cGAN/discriminator-MNIST-MSE.pth"

# %% [markdown]
# #### 4.1.3 Training state variables

# %%
# training loop
D_loss_hist = []
D_real_hist = []
D_fake_hist = []
G_loss_hist = []
fid_hist = []
total_epochs_trained = 0

ada_sign_D_train_lastN = []

# %% [markdown]
# ### 4.2 Training loop

# %%
def train(curr_epochs: int, D_loss_hist: list, D_real_hist: list,
          D_fake_hist: list, G_loss_hist: list, fid_hist: list, * ,
          max_time_sec: float = float('inf'), max_epochs: int = 10_000):
    global best_fid, augment_transforms, ada_sign_D_train_lastN, p
    
    ## Set to training mode
    netD.train()
    netG.train()


    init_time = time.time()
    for i in range(max_epochs):
        D_loss = 0.0
        D_real = 0.0
        D_fake = 0.0
        G_loss = 0.0

        for j, (real_batch, y_batch) in enumerate(dl):
            ## move to GPU
            real_batch = real_batch.to(device) # load images in GPU
            y_batch = y_batch.to(device) # load labels to GPU

            ## do data augmentation (LIVE)
            real_batch = augment_transforms(real_batch)

            ## generate noise
            curr_batch_size = real_batch.shape[0]
            noise = torch.randn([curr_batch_size, nz], device=device)
            conditional_noise = torch.randint(0, num_classes, (curr_batch_size,), device=device)
            ## aux statments
            real_label = torch.full((curr_batch_size,), 1, dtype=torch.float, device=device)
            fake_label = torch.full((curr_batch_size,), 0, dtype=torch.float, device=device)

            ## generate fake batch
            fake_batch = netG(noise, conditional_noise)


            # ----- Train discriminator --------
            # zero all gradients
            netG.zero_grad()
            netD.zero_grad()

            # D_real_loss = 1/m * sum (D(xi) - 1)^2
            real_output = netD(real_batch, y_batch).view(-1)
            D_real_loss = criterion(real_output, real_label)
            D_real_loss.backward() # calcuate gradient
            # D_fake_loss = 1/m * sum (D(G(zi))^2
            # the detach makes it ignore the NetG parameter in gradient computation graph
            # (since fake_img_batch = NetG(noise))
            fake_output = netD(fake_batch.detach(), conditional_noise).view(-1)
            D_fake_loss = criterion(fake_output, fake_label)
            D_fake_loss.backward() # calculate (and accumulate) gradient
            # maximize 1/m * sum log D(xi) + 1/m * sum log (1 - D(G(zi)))
            optimD.step()
            # accumulate loss
            D_loss += D_real_loss.item() + D_fake_loss.item()
            # accumulate real and fake outputs
            D_real += real_output.sum().item()
            D_fake += fake_output.sum().item()

            # ----- Train generator -------
            # zero all gradients
            netG.zero_grad()
            netD.zero_grad()

            # G_loss = 1/m * sum (D(G(z)) - 1)^2
            # now fake_img_batch should be a part of the gradient computation graph
            fake_output = netD(fake_batch, conditional_noise).view(-1)
            G_loss_batch = criterion(fake_output, real_label)
            G_loss_batch.backward() # calculate gradient
            optimG.step()
            # accumulate loss
            G_loss += G_loss_batch.item()

            with torch.no_grad():
                ## ADA logic
                ## take D(x), transform from [0,1] to [-1,1] via linear transform, take the avg sign 
                avg_sign_D_batch = torch.mean(torch.sign(2.0 * real_output - 1.0)).item()
                ada_sign_D_train_lastN.append(avg_sign_D_batch) ## final shape (N,)
                if (j % ada_N == 0 and j != 0):  # do after every N minibatches, except the first
                    rt = np.abs(np.mean(ada_sign_D_train_lastN)) 
                    ada_sign_D_train_lastN = []  ## reset list of last N sign means
                    if (rt > target_rt):  # too much overfitting
                        p = np.clip(p + delta_p, min_p, max_p)
                    else:  # too little overfitting
                        p = np.clip(p - delta_p, min_p, max_p)
                    augment_transforms = data_augmentation_layers(p)


        D_real /= len(dataset) # avg D(x)
        D_fake /= len(dataset) # avg D(G(z))
        curr_epochs += 1 ## update: epoch ended

        print(f"Epoch {curr_epochs}. Generator loss: {G_loss}. " +
            f"Discriminator loss: {D_loss}. " +
            f"Average D(x): {D_real}. Average D(G(z)): {D_fake}. Probability of augmentation {p}.")

        # ## calculate FID
        with torch.no_grad():
            real_batch, _ = next(iter(dl)) # get new (random?) batch
            noise = torch.randn([real_batch.shape[0], nz], device=device)
            conditional_noise = torch.randint(0, num_classes, (real_batch.shape[0],), device=device)
            fake_batch = netG(noise, conditional_noise)
            fid = FID_score(real_batch, fake_batch)
            print(f"FID score for {real_batch.shape[0]} images: {fid}")

        # save for plotting latter
        D_loss_hist.append(D_loss)
        D_real_hist.append(D_real)
        D_fake_hist.append(D_fake)
        G_loss_hist.append(G_loss)
        fid_hist.append(fid)

        # plot every 1 epochs
        if ((i + 1) % 1 == 0):
            real_batch, y_batch = next(iter(dl)) # get new (random?) batch
            plot_side_by_side(real_batch, y_batch, netG)

        # check time
        if (time.time() - init_time > max_time_sec):
            return D_loss_hist, D_real_hist, D_fake_hist, G_loss_hist, fid_hist, curr_epochs

        # save best model
        if (fid < best_fid):
            best_fid = fid
            print(f"Saving generator to {GEN_PATH}")
            torch.save(netG, GEN_PATH)
            print(f"Saving discriminator to {DISC_PATH}")
            torch.save(netD, DISC_PATH)

    return D_loss_hist, D_real_hist, D_fake_hist, G_loss_hist, fid_hist, curr_epochs

# %%
# train
D_loss_hist, D_real_hist, D_fake_hist, \
G_loss_hist, fid_hist, total_epochs_trained = train(total_epochs_trained,
                                                         D_loss_hist,
                                                         D_real_hist,
                                                         D_fake_hist,
                                                         G_loss_hist,
                                                         fid_hist,
                                                         max_time_sec=5*60)

# %% [markdown]
# ### 3.3 Plot evolution of metrics

# %%
# plot loss evolution

plt.rcParams["figure.figsize"] = (10, 10)
plt.subplots_adjust(wspace=1.0, hspace=1.0)

fig, ax = plt.subplots(2, 2)
fig.suptitle("Evolution of statistics over training")
for i, stat, name in zip(range(4),
                         [D_loss_hist, G_loss_hist, D_real_hist, D_fake_hist],
                         ["Discriminator loss", "Generator loss", "Average prediction of real image", "Average prediction of fake image"]):
    print(f"{name}:{stat}")
    ax[i // 2, i % 2].set_title(name)
    ax[i // 2, i % 2].set_xlabel("Epochs")
    ax[i // 2, i % 2].set_ylabel(name)
    ax[i // 2, i % 2].plot(range(total_epochs_trained), stat)

plt.show()

# %%
# plot FID evolution

plt.rcParams["figure.figsize"] = (10, 10)
plt.subplots_adjust(wspace=1.0, hspace=1.0)

plt.title("FID evolution")
plt.xlabel("Epochs")
plt.ylabel("FID")
plt.plot(range(total_epochs_trained), fid_hist)

# %% [markdown]
# ## 4. Evaluation

# %% [markdown]
# ### 4.1 Visualizing real/generated images (side by side)

# %%
## Set to evaluation mode 
netD.eval()
netG.eval()

# %%
real_batch, labels_batch = next(iter(dl))
plot_side_by_side(real_batch, labels_batch, netG)

# %% [markdown]
# ## 4.2 FID calculation (over all 50k images)

# %% [markdown]
# ### 4.2.1 Saving real and fake images

# %%
real_folder = "mnist-real"
fake_folder = "mnist-fake"
real_stats_file = "/home/igor/IC-iVision/gans/cgan/mnist-real-stats.npz"
fake_stats_file = "/home/igor/IC-iVision/gans/cgan/mnist-fake-stats.npz"

# %%
from torchvision.utils import save_image
import os

def save_images(num_imgs, netG, dl, batch_size, num_classes, device, fake_folder, real_folder=None):
    """
        -- num_imgs: int
        -- dl: torch.utils.data.dataloader.Dataloader object
            We assume shuffle=True is set on the dataloader
        -- netG: generator model 
        -- batch_size: int 
        -- num_classes: int 
        -- device: torch.device
        -- fake_folder: str, path to fake dataset folder
        -- real_folder: str, path to real dataset folder
    
        Saves both k real images in a folder, and k fake images in another folder.
    """

    ## if no real_folder was passed, or the one passed was not empty, do not save real images in it
    if real_folder is None or (os.path.isdir(real_folder) and len(os.listdir(real_folder)) > 0):
        save_real_dataset = False
    else:
        save_real_dataset = True
    
    base_idx = 0

    for real_batch, _ in dl:        
        ## accounts for possible uneven batches (when batch_size does not divide len(dataset))
        curr_batch_size = real_batch.shape[0] 
        
        noise = torch.randn([curr_batch_size, nz], device=device)
        conditional_noise = torch.randint(0, num_classes, [curr_batch_size], device=device)

        fake_batch = netG(noise, conditional_noise)
        for j, (real_img, fake_img) in enumerate(zip(real_batch, fake_batch)):
            idx = base_idx + j 
            if (idx >= num_imgs):
                return
            if save_real_dataset:
                real_path = os.path.join(real_folder, f"img{idx:05}.jpeg") ## imgXXXXX.jpeg, where some leading X's might be zero
                with open(real_path, "w") as f:
                    save_image(real_img, f, format="jpeg")
            fake_path = os.path.join(fake_folder, f"img{idx:05}.jpeg") ## imgXXXXX.jpeg, where some leading X's might be zero
            with open(fake_path, "w") as f:
                save_image(fake_img, f, format="jpeg")

        base_idx += curr_batch_size

save_images(num_imgs=50_000, netG=netG, dl=dl, batch_size=batch_size, 
            num_classes=num_classes, device=device, fake_folder=fake_folder, real_folder=real_folder)
        

# %% [markdown]
# ### 4.2.2 FID calculation

# %% [markdown]
# The main reference used in this section is [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500), since this was the first paper to suggest the FID as a metric for evaluating GAN's performance.
# 
# Steps (summarized):
# 1. Clone the repository [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid), one of the most popular FID implementations in PyTorch, or install it via pip
#    ```bash
#     pip install pytorch-fid
#    ```
# 2. Save all real images (from the dataset) in a folder, `/path/to/real/dataset` (or, at least, the amount that will be used to calculate the FID), or use a `.npz` file with precalculated statistics about these images (mean and covariance).
#    1. Generate the `.npz` statistics file for the real dataset, via 
#    ```bash
#     pytorch-fid --save-stats path/to/real/dataset path/to/real/outputfile.npz
#    ```
# 3. Save the same amount of generated/fake images in a folder, `/path/to/fake/dataset`
# 4. Generate the `.npz` statistics file for the fake dataset, via
#    ```bash
#     pytorch-fid --save-stats path/to/fake/dataset path/to/fake/outputfile.npz
#    ```
# 5. Execute the following bash script (in the folder where all the files from the cloned repo are)
# ```bash
#    pytorch-fid path/to/real/outputfile.npz path/to/fake/outputfile.npz
# ```  

# %%
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048] # 2048 is the standard output dim of InceptionV3 activations 

model = InceptionV3([block_idx]).to(device)

if os.path.isfile(fake_stats_file):
    os.remove(fake_stats_file)

if not os.path.isfile(real_stats_file):
    mu_real, sigma_real = fid_score.calculate_activation_statistics(
        files=[os.path.join(real_folder, f) for f in os.listdir(real_folder)],
        model=model,
        batch_size=batch_size,
        device=device,
        num_workers=workers
        )
    np.savez_compressed(real_stats_file, mu=mu_real, sigma=sigma_real)

mu_fake, sigma_fake = fid_score.calculate_activation_statistics(
        files=[os.path.join(fake_folder, f) for f in os.listdir(fake_folder)],
        model=model,
        batch_size=batch_size,
        device=device,
        num_workers=workers
        )

np.savez_compressed(fake_stats_file, mu=mu_fake, sigma=sigma_fake)

print(f"FID: {fid_score.calculate_fid_given_paths([real_stats_file, fake_stats_file],batch_size=batch_size,device=device,num_workers=workers, dims=2048)}")


