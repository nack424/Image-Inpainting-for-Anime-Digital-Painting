import argparse
from datasets import *
import deepspeed
import glob
from network import *
import os
from torchvision.models import vgg19
from utils.loss import *
import wandb

parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--epochs', type=int, help='Number of epochs to train')
parser.add_argument('--load_model', type=str, help='(Optinal) Path to all saved models folder, make sure to leave only target model')
parser.add_argument('--mask_type', type=int, default=1, help='Mask type: 1 for normal mask, 2 for large mask')
parser.add_argument('--train_path', type=str,  help='Training image folder')
parser.add_argument('--save_model', type=str,  help='(Optinal) Folder to save all models')

if __name__ == '__main__':
    cmd_args = parser.parse_args()

    coarse_model = CoarseNet().to('cuda')
    super_resolution_model = SuperResolutionNet().to('cuda')
    refinement_model = RefinementNet(use_gpu=True).to('cuda')

    discriminator_model = Discriminator().to('cuda')

    vgg19_model = vgg19(weights='IMAGENET1K_V1').to('cuda')
    vgg19_model = vgg19_model.features[:21].to('cuda')

    trainset = JointDataset(cmd_args.train_path, cmd_args.mask_type)

    coarse_engine, coarse_optimizer, train_dataloader, _ = deepspeed.initialize(config = 'train_deepspeed_config.json', model=coarse_model,
                                                         model_parameters=coarse_model.parameters(), training_data = trainset)
    super_resolution_engine, super_resolution_optimizer, _, _ = deepspeed.initialize(config = 'train_deepspeed_config.json',
                                                                                     model=super_resolution_model,
                                                                                     model_parameters=super_resolution_model.parameters())
    refinement_engine, refinement_optimizer, _, _ = deepspeed.initialize(config = 'train_deepspeed_config.json',
                                                                         model=refinement_model,
                                                                         model_parameters=refinement_model.parameters())
    discriminator_engine, discriminator_optimizer, _, _ = deepspeed.initialize(config = 'train_deepspeed_config.json',
                                                                         model=discriminator_model,
                                                                         model_parameters=discriminator_model.parameters())

    if cmd_args.load_model is not None:
        if len(glob.glob(os.path.join(cmd_args.load_model, 'coarse_*'))) >0:
            coarse_folder = glob.glob(os.path.join(glob.glob(os.path.join(cmd_args.load_model, 'coarse_*'))[0], 'global_step*'))[0]
            coarse_engine.load_checkpoint(load_dir=os.path.join(coarse_folder, os.listdir(coarse_folder)[0]))

        if len(glob.glob(os.path.join(cmd_args.load_model, 'super_resolution_*'))) >0:
            super_resolution_folder = glob.glob(os.path.join(glob.glob(os.path.join(cmd_args.load_model, 'super_resolution_*'))[0], 'global_step*'))[0]
            super_resolution_engine.load_checkpoint(load_dir=os.path.join(super_resolution_folder, os.listdir(super_resolution_folder)[0]))

        if len(glob.glob(os.path.join(cmd_args.load_model, 'refinement_*'))) >0:
            refinement_folder = glob.glob(os.path.join(glob.glob(os.path.join(cmd_args.load_model, 'refinement_*'))[0], 'global_step*'))[0]
            refinement_engine.load_checkpoint(load_dir=os.path.join(refinement_folder, os.listdir(refinement_folder)[0]))

        if len(glob.glob(os.path.join(cmd_args.load_model, 'discriminator_*'))) >0:
            discriminator_folder = glob.glob(os.path.join(glob.glob(os.path.join(cmd_args.load_model, 'discriminator_*'))[0], 'global_step*'))[0]
            discriminator_engine.load_checkpoint(load_dir=os.path.join(discriminator_folder, os.listdir(discriminator_folder)[0]))

    coarse_loss_function = Coarse_loss(vgg19_model, vgg_loss_weight=0.01)
    super_resolution_loss_function = L1_loss()
    refinement_loss_function = Joint_refinement_loss(vgg19_model, vgg_loss_weight = 1e-5, gan_loss_weight = 0.5,
                                                     gradient_loss_weight = 1)
    discriminator_loss_function = Discriminator_loss()

    wandb.init(
        project="anime_inpaint",
        config={
            "learning_rate": 1e-5,
            "architecture": "CNN",
            "epochs": cmd_args.epochs,
        }
    )

    for epoch in range(cmd_args.epochs):
        for batch, data in enumerate(train_dataloader):
            masked_image, mask, lr_groundtruth, hr_groundtruth = data
            masked_image, mask, lr_groundtruth, hr_groundtruth = masked_image.to('cuda'), mask.to('cuda'), \
                                                                 lr_groundtruth.to('cuda'), hr_groundtruth.to('cuda')

            coarse_output = coarse_engine(masked_image, mask)
            coarse_output = mask * coarse_output + (1 - mask) * masked_image
            coarse_loss = coarse_loss_function(coarse_output, lr_groundtruth)

            super_resolution_output = super_resolution_engine(coarse_output)
            super_resolution_loss = super_resolution_loss_function(super_resolution_output, hr_groundtruth)

            mask = F.interpolate(mask, scale_factor=2, mode='nearest')
            refinement_output = refinement_engine(super_resolution_output, mask)
            refinement_output = mask * refinement_output + (1-mask) * hr_groundtruth

            real_prediction = discriminator_engine(hr_groundtruth, mask)
            fake_prediction = discriminator_engine(refinement_output, mask)

            discriminator_loss = discriminator_loss_function(real_prediction, fake_prediction)
            refinement_loss = refinement_loss_function(refinement_output, hr_groundtruth, fake_prediction)

            inpaint_loss = coarse_loss + super_resolution_loss + refinement_loss

            coarse_engine.backward(inpaint_loss, retain_graph=True)
            coarse_engine.step()

            super_resolution_engine.backward(inpaint_loss, retain_graph=True)
            super_resolution_engine.step()

            refinement_engine.backward(inpaint_loss, retain_graph=True)
            refinement_engine.step()

            discriminator_engine.backward(discriminator_loss)
            discriminator_engine.step()
