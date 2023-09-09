# Image Inpainting for Anime Digital Painting (Underdevelopment)

This project is free time project inspire by anime related project and DeepCreampy. Model is Pytorch implementation of [zoom-to-inpaint](https://github.com/google/zoom-to-inpaint).

## Disclaimer

Author try to trained this model on cloud GPU rental(Vast.ai) with 4 Nvidia RTX A5000 GPUs. However, cost of training this model is very high due to 4.5 million parameters, require a lot of images (10k-100k images) to train and require a lot of training step (100k+ steps). Author decided to abandon this project in training step and afterward. If anyone like to use this model, please use at your own risk. You can also train this model with regular image other than anime image.

## Requirements

### Cuda GPU and Cuda Software

This project is coded to train with Cuda, so Cuda GPU and [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) is required. [cuDNN](https://developer.nvidia.com/cudnn) is optional but can be installed to train the model faster.

### Python Environments

* Python >= 3.9
* Pytorch >= 1.13.1 with Cuda support
* torchvision >= 0.14.1
* wandb >= 0.13.7

If you use Anaconda, use these commands to install Python Environments

```
> conda create -n anime-inpainting
> conda activate anime-inpainting
> conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
> conda install -c conda-forge wandb
```

### WandB set up
WandB is used for monitoring loss during training. 
