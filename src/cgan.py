import torch
from torch.utils.data.dataloader import DataLoader 
from torchinfo import summary

import config as config
import models as models
from FID import FID
import vutils
import train as train

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help="Amount of epochs to be run. If training by time limit instead, it becomes the maximum amount of epochs that can be run (if the time limit allows). Default: 200", type=int, default=200)
parser.add_argument("--time-limit", help="Maximum amount of time (t) that the network will run (approximately). More precisely, it only starts a new epoch if the total elapsed time if less than t. Default: infinity", type=float, default=float("inf"))  
parser.add_argument("-v", "--verbose", help="Activates some verbosity options regarding training (ex.: epoch information). Default: false", action="store_true")
parser.add_argument("-p", "--plot", help="Plot one batch of real images at the start (to test dataloader), and on batch of images (real and fake, side by side) after each epoch for visualization. Default: False", action="store_true")
parser.add_argument("-s", "--stats", help="Plot evolution of statistics at the end of the training. Default: False", action="store_true")
parser.add_argument("--final-plot", help="Plot one batch of generated images at the end of training. Default: False", action="store_true")
parser.add_argument("--fid", help="Calculate FID at the end of training. Default: False", action="store_true")
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

if args.verbose:
    print(f"Running on {device}")
    print(f"Loaded {len(config.dataset)} images of 2 classes")

dl = DataLoader(config.dataset, batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.workers)

if args.plot:
    img_batch, _ = next(iter(dl))
    vutils.plot_single(img_batch)
    # print(f"FID score (against itself) of {FID_score.calculate(img_batch, img_batch)}. Expected 0.0.")

augment_transforms = models.data_augmentation_layers(config.p)
netG = models.cGenerator().to(device)
netD = models.cDiscriminator().to(device)
optimG = torch.optim.Adam(netG.parameters(), lr = config.lr, betas = config.betas)
optimD = torch.optim.Adam(netD.parameters(), lr = config.lr, betas = config.betas)
if args.verbose:
    print(" Generator architecture ".center(80, "-"))
    summary(netG, input_size=[(1, config.nz), (1,)], 
            dtypes=[torch.float, torch.int],
            col_names=["kernel_size", "output_size", "num_params"],
            row_settings=["var_names"])
    print(" Discriminator architecture ".center(80, "-"))
    summary(netD, input_size=[(1, config.nc, config.img_size[0], config.img_size[1]), (1,)], 
            dtypes=[torch.float, torch.int],
            col_names=["kernel_size", "output_size", "num_params"],
            row_settings=["var_names"])

D_loss_hist, D_real_hist, D_fake_hist,\
G_loss_hist, total_epochs_trained =  train.train(
    netG=netG, 
    netD=netD,
    optimG=optimG,
    optimD=optimD,
    dl=dl,
    device=device,
    augment_transforms=augment_transforms,
    max_time_sec=args.time_limit,
    max_epochs=args.epochs,
    verbose=args.verbose,
    plotting=args.plot)

netG.eval()
netD.eval()

if args.final_plot:
    real_batch, labels_batch = next(iter(dl))
    vutils.viewGeneratedBatch(dl, netG, device=device)

if args.stats:
    vutils.plot_stats([D_loss_hist, G_loss_hist, D_real_hist, D_fake_hist],
                      ["Discriminator loss", "Generator loss", "Average D(x)", "Average D(G(z))"],
                      total_epochs=total_epochs_trained)

if args.verbose:
    print("Saving 50k generated images")
train.save_images(num_imgs=50_000, netG=netG, dl=dl, device=device)
if args.verbose:
    print("Saved")

if args.fid:
    FID_score = FID(device=device, verbose=True) # initialize FID calculator
    print(f"Final FID score of {FID_score.calculate()}")



