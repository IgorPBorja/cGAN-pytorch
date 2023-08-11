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

import torchvision
from torchvision import transforms
from scipy.linalg import sqrtm # principal square root of positive (semi)definite matrix
from torch import nn
import torch
import numpy as np
import os
import config
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3

class FID:
    # def init_inception(self, verbose=False):
    #     self.model_inceptionV3 = torchvision.models.inception_v3(pretrained=True).to(self.device)
    #     self.model_inceptionV3.eval() # evaluation/inference mode

    #     self.preprocess = transforms.Compose([
    #         transforms.Lambda(lambda img: img.expand(-1, 3, -1, -1)),  # Acts on batch. From grayscale to RGB with R=G=B = original channel intensity
    #         transforms.Resize(299),
    #         transforms.CenterCrop(299),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])
    #     # %%
    #     # remove fc layers
    #     self.model_inceptionV3._modules["dropout"] = nn.Identity()
    #     self.model_inceptionV3._modules["fc"] = nn.Identity()

    #     if verbose:
    #         print("Loaded InceptionV3")
    #         print(f"Total parameters: {sum(x.numel() for x in self.model_inceptionV3.parameters() if x.requires_grad)}")

    def __init__(self, device: torch.device, verbose = True):
        self.device = device
        # self.init_inception(verbose)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048] # 2048 is the standard output dim of InceptionV3 activations 
        self.model = InceptionV3([block_idx]).to(device)
        if verbose:
            print("Loaded InceptionV3")
            print(f"Total (trainable) parameters: {sum(x.numel() for x in self.model.parameters() if x.requires_grad)}")


# %%

    # def calculate_small(self, real_batch: torch.Tensor, fake_batch: torch.Tensor):
    #     act1 = self.preprocess(real_batch).to(self.device)
    #     act2 = self.preprocess(fake_batch).to(self.device)

    #     ## inference only
    #     with torch.no_grad():
    #         act1 = self.model_inceptionV3(act1)
    #         act2 = self.model_inceptionV3(act2)

    #     ## pass to cpu and then convert to numpy array
    #     act1 = act1.cpu().detach().numpy()
    #     act2 = act2.cpu().detach().numpy()

    #     mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    #     mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    #     mean_dist_square = np.sum((mu1 - mu2)**2)
    #     sqrt_cov = sqrtm(sigma1.dot(sigma2))
    #     if np.iscomplexobj(sqrt_cov):
    #         sqrt_cov = sqrt_cov.real

    #     return mean_dist_square + np.trace(sigma1 + sigma2 - 2.0 * sqrt_cov)

    def calculate(self) -> float:
        # %%
        if os.path.isfile(config.fake_stats_file):
            os.remove(config.fake_stats_file)

        if not os.path.isfile(config.real_stats_file):
            mu_real, sigma_real = fid_score.calculate_activation_statistics(
                files=[os.path.join(config.real_folder, f) for f in os.listdir(config.real_folder)],
                model=self.model,
                batch_size=config.batch_size,
                device=self.device,
                num_workers=config.workers
                )
            np.savez_compressed(config.real_stats_file, mu=mu_real, sigma=sigma_real)

        mu_fake, sigma_fake = fid_score.calculate_activation_statistics(
                files=[os.path.join(config.fake_folder, f) for f in os.listdir(config.fake_folder)],
                model=self.model,
                batch_size=config.batch_size,
                device=self.device,
                num_workers=config.workers
                )

        np.savez_compressed(config.fake_stats_file, mu=mu_fake, sigma=sigma_fake)

        final_fid = fid_score.calculate_fid_given_paths([config.real_stats_file, config.fake_stats_file],batch_size=config.batch_size,device=self.device,num_workers=config.workers, dims=2048)

        return final_fid
