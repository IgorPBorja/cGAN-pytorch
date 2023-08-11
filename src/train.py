import torch
import time
import config
import models
import vutils
import numpy as np
import os
from torchvision.utils import save_image


def train(netG: torch.nn.Module, 
          netD: torch.nn.Module,
          optimG: torch.optim.Adam,
          optimD: torch.optim.Adam,
          dl,
          augment_transforms: torch.nn.Sequential,
          device = "cuda:0",
          *,
          max_time_sec: float = float('inf'),
          max_epochs: int = 10_000,
          verbose=True,
          plotting=True,
          plotting_k = 1):
    # %%
    # training loop
    D_loss_hist = []
    D_real_hist = []
    D_fake_hist = []
    G_loss_hist = []
    curr_epochs = 0
    p = 0.0

    ada_sign_D_train_lastN = []
    
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
            noise = torch.randn([curr_batch_size, config.nz], device=device)
            conditional_noise = torch.randint(0, config.num_classes, (curr_batch_size,), device=device)
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
            D_real_loss = config.criterion(real_output, real_label)
            D_real_loss.backward() # calcuate gradient
            # D_fake_loss = 1/m * sum (D(G(zi))^2
            # the detach makes it ignore the NetG parameter in gradient computation graph
            # (since fake_img_batch = NetG(noise))
            fake_output = netD(fake_batch.detach(), conditional_noise).view(-1)
            D_fake_loss = config.criterion(fake_output, fake_label)
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
            G_loss_batch = config.criterion(fake_output, real_label)
            G_loss_batch.backward() # calculate gradient
            optimG.step()
            # accumulate loss
            G_loss += G_loss_batch.item()

            with torch.no_grad():
                ## ADA logic
                ## take D(x), transform from [0,1] to [-1,1] via linear transform, take the avg sign 
                avg_sign_D_batch = torch.mean(torch.sign(2.0 * real_output - 1.0)).item()
                ada_sign_D_train_lastN.append(avg_sign_D_batch) ## final shape (N,)
                if (j % config.ada_N == 0 and j != 0):  # do after every N minibatches, except the first
                    rt = np.abs(np.mean(ada_sign_D_train_lastN)) 
                    ada_sign_D_train_lastN = []  ## reset list of last N sign means
                    if (rt > config.target_rt):  # too much overfitting
                        p = np.clip(p + config.delta_p, config.min_p, config.max_p)
                    else:  # too little overfitting
                        p = np.clip(p - config.delta_p, config.min_p, config.max_p)
                    augment_transforms = models.data_augmentation_layers(p)


        D_real /= len(config.dataset) # avg D(x)
        D_fake /= len(config.dataset) # avg D(G(z))
        curr_epochs += 1 ## update: epoch ended

        if verbose:
            print(f"Epoch {curr_epochs}. Generator loss: {G_loss}. " +
                    f"Discriminator loss: {D_loss}. " +
                    f"Average D(x): {D_real}. Average D(G(z)): {D_fake}. Probability of augmentation {p}.")

        # save for plotting latter
        D_loss_hist.append(D_loss)
        D_real_hist.append(D_real)
        D_fake_hist.append(D_fake)
        G_loss_hist.append(G_loss)

        # plot every k epochs
        if (plotting and (i % plotting_k == 0)):
            vutils.viewGeneratedBatch(dl, netG, device=device)

        # check time
        if (time.time() - init_time > max_time_sec):
            return D_loss_hist, D_real_hist, D_fake_hist, G_loss_hist, curr_epochs

        # save best model
        if verbose:
            print(f"Saving generator to {config.GEN_PATH}")
        torch.save(netG, config.GEN_PATH)
        
        if verbose:
            print(f"Saving discriminator to {config.DISC_PATH}")
        torch.save(netD, config.DISC_PATH)

    return D_loss_hist, D_real_hist, D_fake_hist, G_loss_hist, curr_epochs

# %%

def save_images(num_imgs, netG, dl, device):
    """
        -- num_imgs: int
        -- netG: generator model 
        -- dl: torch.utils.data.dataloader.Dataloader object
            We assume shuffle=True is set on the dataloader
        -- device: torch.device
    
        Saves both k real images in a folder, and k fake images in another folder.
    """

    ## if no real_folder was passed, or the one passed was not empty, do not save real images in it
    if config.real_folder is None or (os.path.isdir(config.real_folder) and len(os.listdir(config.real_folder)) > 0):
        save_real_dataset = False
    else:
        save_real_dataset = True
    
    ## create folders, if not existent
    if (not os.path.isdir(config.fake_folder)):
        os.makedirs(config.fake_folder)

    if (save_real_dataset and (not os.path.isdir(config.real_folder))):
        os.makedirs(config.real_folder)

    base_idx = 0

    for real_batch, _ in dl:        
        ## accounts for possible uneven batches (when batch_size does not divide len(dataset))
        curr_batch_size = real_batch.shape[0] 
        
        noise = torch.randn([curr_batch_size, config.nz], device=device)
        conditional_noise = torch.randint(0, config.num_classes, [curr_batch_size], device=device)

        fake_batch = netG(noise, conditional_noise)
        for j, (real_img, fake_img) in enumerate(zip(real_batch, fake_batch)):
            idx = base_idx + j 
            if (idx >= num_imgs):
                return
            if save_real_dataset:
                real_path = os.path.join(config.real_folder, f"img{idx:05}.jpeg") ## imgXXXXX.jpeg, where some leading X's might be zero
                with open(real_path, "w") as f:
                    save_image(real_img, f, format="jpeg")
            fake_path = os.path.join(config.fake_folder, f"img{idx:05}.jpeg") ## imgXXXXX.jpeg, where some leading X's might be zero
            with open(fake_path, "w") as f:
                save_image(fake_img, f, format="jpeg")

        base_idx += curr_batch_size
