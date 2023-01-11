import argparse
from datasets import *
import glob
from network import *
import os
from torchvision.models import vgg19
from torch.utils.data import DataLoader
from utils.loss import *
import wandb

parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--batch_size', type=int, default = 1, help='Amount of data that pass simultaneously to model')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=int, default=1e-5, help='Control amount of weight change during optimization')
parser.add_argument('--load_discriminator', type=str, help='(Optinal) Folder contain discriminator model')
parser.add_argument('--load_inpaint', type=str, help='(Optinal) Folder contain 3 inpaints model, make sure to leave only target models')
parser.add_argument('--mask_type', type=int, default=1, help='Mask type: 1 for normal mask, 2 for large mask')
parser.add_argument('--num_workers', type=int, default=0, help='Amount of multiprocess in dataloader (0 mean use single process)')
parser.add_argument('--train_path', type=str,  help='Training image folder')
parser.add_argument('--val_path', type=str,  help='(Optinal) Validation image folder')
parser.add_argument('--save_model', type=str,  help='(Optinal) Folder to save all models')

if __name__ == '__main__':
    cmd_args = parser.parse_args()

    coarse_model = CoarseNet().to('cuda')
    super_resolution_model = SuperResolutionNet().to('cuda')
    refinement_model = RefinementNet(use_gpu=True).to('cuda')

    discriminator_model = Discriminator().to('cuda')

    vgg19_model = vgg19(weights='IMAGENET1K_V1').to('cuda')
    vgg19_model = vgg19_model.features[:21].to('cuda')

    train_dataset = JointDataset(cmd_args.train_path, cmd_args.mask_type)
    train_dataloader = DataLoader(train_dataset, batch_size=cmd_args.batch_size, num_workers=cmd_args.num_workers)

    if cmd_args.val_path is not None:
        val_dataset = JointDataset(cmd_args.val_path, cmd_args.mask_type)
        val_dataloader = DataLoader(val_dataset, batch_size=cmd_args.batch_size, num_workers=cmd_args.num_workers)

    if cmd_args.load_inpaint is not None:
        coarse_model.load_state_dict(torch.load(glob.glob(os.path.join(cmd_args.load_inpaint, 'coarse*'))[0]))
        super_resolution_model.load_state_dict(torch.load(glob.glob(os.path.join(cmd_args.load_inpaint, 'super_resolution*'))[0]))
        refinement_model.load_state_dict(torch.load(glob.glob(os.path.join(cmd_args.load_inpaint, 'refinement*'))[0]))

    if cmd_args.load_discriminator is not None:
        discriminator_model.load_state_dict(torch.load(glob.glob(os.path.join(cmd_args.load_inpaint, 'discriminator*'))[0]))

    coarse_loss_function = Coarse_loss(vgg19_model, vgg_loss_weight=0.01)
    super_resolution_loss_function = L1_loss()
    refinement_loss_function = Joint_refinement_loss(vgg19_model, vgg_loss_weight = 1e-5, gan_loss_weight = 0.5,
                                                     gradient_loss_weight = 1)
    discriminator_loss_function = Discriminator_loss()

    inpaint_parameters = list(coarse_model.parameters()) + list(super_resolution_model.parameters()) + \
                         list(refinement_model.parameters())

    discriminator_optimizer = torch.optim.AdamW(discriminator_model.parameters(), lr = cmd_args.learning_rate)
    inpaint_optimizer = torch.optim.AdamW(inpaint_parameters, lr = cmd_args.learning_rate)

    scaler = torch.cuda.amp.GradScaler(init_scale=16834.0, enabled=True)

    wandb.init(
        project="anime_inpaint_joint"
    )

    for epoch in range(cmd_args.epochs):
        total_train_inpaint_loss = 0
        total_train_discriminator_loss = 0

        total_val_inpaint_loss = 0
        total_val_discriminator_loss = 0

        num_batch = len(train_dataloader)

        for batch, data in enumerate(train_dataloader):
            masked_image, mask, lr_groundtruth, hr_groundtruth = data
            masked_image, mask, lr_groundtruth, hr_groundtruth = masked_image.to('cuda'), mask.to('cuda'), \
                                                                 lr_groundtruth.to('cuda'), hr_groundtruth.to('cuda')

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                output = coarse_model(masked_image, mask)
                output = mask*output + (1-mask)*masked_image #Coarse output
                coarse_loss = coarse_loss_function(output, lr_groundtruth)

                output = super_resolution_model(output) #Super resolution output
                super_resolution_loss = super_resolution_loss_function(output, hr_groundtruth)

                mask = F.interpolate(mask, size = (512, 512), mode = 'nearest')
                output = refinement_model(output, mask)

                output = mask*output + (1-mask)*hr_groundtruth #Refinement output

                real_prediction = discriminator_model(hr_groundtruth, mask)
                fake_prediction = discriminator_model(output, mask)

                discriminator_loss = discriminator_loss_function(real_prediction, fake_prediction)
                refinement_loss = refinement_loss_function(output, hr_groundtruth, fake_prediction)

                inpaint_loss = coarse_loss + super_resolution_loss + refinement_loss

                total_train_discriminator_loss += discriminator_loss.item()
                total_train_inpaint_loss += inpaint_loss.item()

            scaler.scale(discriminator_loss).backward(inputs = list(discriminator_model.parameters()), retain_graph=True)
            scaler.step(discriminator_optimizer)
            discriminator_optimizer.zero_grad(set_to_none=True)

            # scaler.scale(inpaint_loss).backward(inputs = inpaint_parameters)
            # scaler.step(inpaint_optimizer)
            # inpaint_optimizer.zero_grad(set_to_none=True)

            scaler.update()

        if cmd_args.val_path is not None:
            for batch, data in enumerate(val_dataloader):
                masked_image, mask, lr_groundtruth, hr_groundtruth = data
                masked_image, mask, lr_groundtruth, hr_groundtruth = masked_image.to('cuda'), mask.to('cuda'), \
                    lr_groundtruth.to('cuda'), hr_groundtruth.to('cuda')

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    output = coarse_model(masked_image, mask)
                    output = mask * output + (1 - mask) * masked_image  # Coarse output
                    coarse_loss = coarse_loss_function(output, lr_groundtruth)

                    output = super_resolution_model(output)  # Super resolution output
                    super_resolution_loss = super_resolution_loss_function(output, hr_groundtruth)

                    mask = F.interpolate(mask, size=(512, 512), mode='nearest')
                    output = refinement_model(output, mask)

                    output = mask * output + (1 - mask) * hr_groundtruth  # Refinement output

                    real_prediction = discriminator_model(hr_groundtruth, mask)
                    fake_prediction = discriminator_model(output, mask)

                    discriminator_loss = discriminator_loss_function(real_prediction, fake_prediction)
                    refinement_loss = refinement_loss_function(output, hr_groundtruth, fake_prediction)

                    inpaint_loss = coarse_loss + super_resolution_loss + refinement_loss

                    total_val_discriminator_loss += discriminator_loss.item()
                    total_val_inpaint_loss += inpaint_loss.item()

        average_train_inpaint_loss = total_train_inpaint_loss / num_batch
        average_train_discriminator_loss = total_train_discriminator_loss / num_batch

        if cmd_args.val_path is not None:
            average_val_inpaint_loss = total_val_inpaint_loss / num_batch
            average_val_discriminator_loss = total_val_discriminator_loss / num_batch
            wandb.log({"train_inpaint_loss": average_train_inpaint_loss,
                       "train_discriminator_loss": average_train_discriminator_loss,
                       "val_inpaint_loss": average_val_inpaint_loss,
                       "val_discriminator_loss": average_val_discriminator_loss
                       })
        else:
            wandb.log({"train_inpaint_loss": average_train_inpaint_loss,
                       "train_discriminator_loss": average_train_discriminator_loss})

        if cmd_args.save_model is not None and ((100 * (epoch + 1)) / cmd_args.epochs) % 10 == 0:
            torch.save(coarse_model.state_dict(), os.path.join(cmd_args.save_model, 'coarse_joint' + str(epoch + 1) + '.pt'))
            torch.save(super_resolution_model.state_dict(),
                       os.path.join(cmd_args.save_model, 'super_resolution_joint' + str(epoch + 1) + '.pt'))
            torch.save(refinement_model.state_dict(),
                       os.path.join(cmd_args.save_model, 'refinement_joint' + str(epoch + 1) + '.pt'))
            torch.save(discriminator_model.state_dict(),
                       os.path.join(cmd_args.save_model, 'discriminator_joint' + str(epoch + 1) + '.pt'))


