import torch
from torch.utils.data.dataloader import DataLoader 

import src.config as config
import src.models as models
from src.FID import FID
import vutils
import src.train as train

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print(f"Running on {device}")

FID_score = FID(device=device, verbose=True) # initialize FID calculator

print(f"Loaded {len(config.dataset)} images of 2 classes")

dl = DataLoader(config.dataset, batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.workers)

img_batch, _ = next(iter(dl))
vutils.plot_single(img_batch)
print(f"FID score (against itself) of {FID_score.calculate(img_batch, img_batch)}. Expected 0.0.")

augment_transforms = models.data_augmentation_layers(config.p)
netG = models.cGenerator().to(device)
netD = models.cDiscriminator().to(device)

D_loss_hist, D_real_hist, D_fake_hist,\
G_loss_hist, fid_hist, total_epochs_trained =  train.train(
    netG=netG, 
    netD=netD, 
    dl=dl,
    device=device,
    max_time_sec=1*60
    )

netG.eval()
netD.eval()

real_batch, labels_batch = next(iter(dl))
vutils.plot_side_by_side(real_batch, labels_batch, netG)

train.save_images(num_imgs=50_000, netG=netG, dl=dl, batch_size=config.batch_size, 
            num_classes=config.num_classes, device=device, fake_folder=config.fake_folder, real_folder=config.real_folder)

print(f"Final FID score of {FID_score.calculate()}")



