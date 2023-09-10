# Image Inpainting for Anime Digital Painting (Underdevelopment)
This project is a personal endeavor undertaken, inspired by anime-related projects and DeepCreampy. It involves implementing a PyTorch model for [zoom-to-inpaint](https://github.com/google/zoom-to-inpaint).

## Disclaimer

We try to trained this model on cloud GPU rental (Vast.ai) with 4 Nvidia RTX A5000 GPUs. However, cost of training this model is very high due to 4.5 million parameters, require a lot of images (10k-100k images) and lot of training step (100k+ steps) to train. We decided to pause this project in training step and afterward. If anyone prefer to use this model, please use at your own risk. You can also train this model with regular image other than anime image.

## Requirements

### Cuda GPU and Cuda Software

This project is coded to train with CUDA, so CUDA GPU and [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) is required. [cuDNN](https://developer.nvidia.com/cudnn) is optional but can be installed to train the model faster.

### Python Environments

* Python >= 3.9
* Pytorch >= 1.13.1 with Cuda support
* torchvision >= 0.14.1 with Cuda support
* wandb >= 0.13.7

For Anaconda user, use these commands to install Python Environments

```
> conda create -n anime-inpainting
> conda activate anime-inpainting
> conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
> conda install -c conda-forge wandb
```

### WandB set up

WandB is used for monitoring loss during training. Sign up at [WandB](https://wandb.ai/site) and receive [API key](https://wandb.ai/authorize). In command line type

```
> wandb login
```

and paste received API key.

If you are using python notebook, you also can use this code

```
> import wandb
> wandb.login(key='YOU API KEY HERE')
```

## Training

This model consist of 3 networks namely Coarse Network, Super Resolution Network and Refinement Network. First, each network will be pre-trained individualy. After that, all networks will be combinded trained with small mask. Lastly, all networks will be combinded trained with larger mask.

### Pretrain

Pretrain commmand line usage example:

```
> python pretrain.py --model='coarse' --train_path='./trainset' --save_model='./save_model'
```

Pretrain command line full usage:

```
> python pretrain.py --batch_size=int --epochs=int --learning_rate=float [--load_model=str]
                     --model=str [--save_model=str] --train_path=str --world_size=int

Required arguments:
  --batch_size          Amount of images that pass simultaneously to model (default:8)
  --epochs              Amount of training steps (default:10)
  --learning_rate       Control weights change during optimization (default:1e-5)
  --model               Model to train specific one of these names: 'coarse', 'super_resolution' or 'refinement'
  --train_path          Folder contain images for training (image size must be 512x512 pixels or higher)
  --world_size          Number of GPUs to do multi-GPUs training. Ignore this argument if use single GPU (default:1)

Optional arguments:
  --load_model          Folder contain saved model (Model must match --model name and folder should contain only one model file)
  --save_model          Folder to save model
```

You can monitor pre-train loss in WandB's project.

### Combined train

Combined train command line usage example:

```
> python train.py --mask_type=1 --train_path='./trainset' --save_model='./save_model'
```

Combined train command line full usage:

```
> python train.py --batch_size=int --epochs=int --learning_rate=float [--load_discriminator=str] [--load_inpaint=str]
                  --mask_type=int [--save_model=str] --train_path=str [--val_path=str] --world_size=int

Required arguments:
  --batch_size          Amount of images that pass simultaneously to model (default:1)
  --epochs              Amount of training steps (default:10)
  --learning_rate       Control weights change during optimization (default:1e-5)
  --mask_type           Specific one of these numbers: 1 for small mask and 2 for larger mask (default:1)
  --train_path          Folder contain images for training (Image size must be 512x512 pixels or higher)
  --world_size          Number of GPUs to do multi-GPUs training. Ignore this argument if use single GPU (default:1)

Optional arguments:
  --load_discriminator  Folder contain discriminator model (Folder should contain only one discriminator model file)
  --load_inpaint        Folder contain coarse model, super_resulution model and refinement model (Folder should contain one file for each model)
  --save_model          Folder to save model
  --val_path=str        Folder contain images for validation (Image size must be 512x512 pixels or higher)
```

In jointly training, WandB will show 4 losses namely coarse_loss, super_resolution_loss, refinement_loss and discriminator_loss.

Discriminator loss should converge to 2.

## Evaluation

Evaluate command line usage example:

```
> python test.py --image_path='./testset' --load_model='./save_model' --output_path='./result' --model='inpaint'
```

Evaluate command line full usage:

```
> python test.py --image_path=str --load_model=str --output_path=str --model=str

Required arguments:
  --image_path          Folder contain image for testing (Image size must be 512x512 pixels or higher)
  --load_model          Folder contain saved model (Folder should contain only one model file or if test all model combined, one file for each model)
  --output_path         Folder to save testing result
  --model               Model to test specific one of these names: 'coarse', 'super_resolution', 'refinement' or 'inpaint'(all model combined)
```


