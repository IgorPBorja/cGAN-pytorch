# "pytorch-fid" Tutorial

# Instalação

* Ver o repositório do GitHub, [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid), link em https://github.com/mseitzer/pytorch-fid, para mais orientações.

```bash
pip install pytorch-fid
```

# Help:

* Output de `pytorch-fid --help`
```
usage: pytorch-fid [-h] [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS] [--device DEVICE] [--dims {64,192,768,2048}]
                   [--save-stats]
                   path path

positional arguments:
  path                  Paths to the generated images or to .npz statistic files

options:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Batch size to use (default: 50)
  --num-workers NUM_WORKERS
                        Number of processes to use for data loading. Defaults to `min(8, num_cpus)` (default: None)
  --device DEVICE       Device to use. Like cuda, cuda:0 or cpu (default: None)
  --dims {64,192,768,2048}
                        Dimensionality of Inception features to use. By default, uses pool3 features (default: 2048)
  --save-stats          Generate an npz archive from a directory of samples. The first path is used as input and the second
                        as output. (default: False)
```

# Passos

0. Escolha a quantidade $k$ de imagens a serem utilizadas para cálculo de FID. O artigo original (que introduziu a métrica), [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500), sugere o uso de **50k** imagens reais (comparadas a, evidentemente, o mesmo número de imagens falsas)

1. Gere $k$ imagens falsas, e salve elas em um diretório sob algum formato (como `.png`, `.jpg` ou `.jpeg`).

    ```python
    from torchvision.utils import save_image
    import os

    def save_images(k, netG, dl, batch_size, num_classes, device, fake_folder, real_folder=None):
        """
            -- k: int
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
        if real_folder is None:
            save_real_dataset = False
        
        for i, (real_batch, label_batch) in enumerate(dl):
            if (i > k):
                break
            
            noise = torch.randn([batch_size, nz], device=device)
            conditional_noise = torch.randint(0, num_classes, [batch_size], device=device)

            fake_batch = netG(noise, conditional_noise)
            for j, (real_img, fake_img) in enumerate(zip(real_batch, fake_batch)):
                idx = batch_size * i + j 
                if save_real_dataset:
                    real_path = os.path.join(real_folder, f"img{idx:05}.jpeg") ## imgXXXXX.jpeg, where some leading X's might be zero
                    with open(real_path, "w") as f:
                        save_image(real_batch[j], f, format="jpeg")
                fake_path = os.path.join(fake_folder, f"img{idx:05}.jpeg") ## imgXXXXX.jpeg, where some leading X's might be zero
                with open(fake_path, "w") as f:
                    save_image(fake_img, f, format="jpeg")
    ```

2. Gene arquivo de estatísticas (`.npz`)
    * Isso deve levar em torno de 5 min (para 50k imagens)
    * Código (bash):
    ```bash
        pytorch-fid --device cuda:0 --batch-size 50 --save-stats /path/to/real/folder /path/to/real/outputfile.npz
        pytorch-fid --device cuda:0 --batch-size 50 --save-stats /path/to/fake/folder /path/to/fake/outputfile.npz
    ```

3. Compute o FID usando o pacote. 
   * Uma vez que estamos usando apenas os arquivos de estatísticas (`.npz`), deve ser bem mais rápido do que utilizar as imagens em si.
   * É bom manter o mesmo `batch-size` utilizado na geração das estatísticas.
   * Código (bash):
   ```bash
       pytorch-fid --device cuda:0 --batch-size 50 /path/to/real/outfile.npz /path/to/fake/outfile.npz
   ```
