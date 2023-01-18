import argparse
from datasets import *
import glob
from network import *
import os
from torchvision.models import vgg19
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.distributed import *
from utils.loss import *
import wandb

parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--batch_size', type=int, default = 1, help='Amount of data that pass simultaneously to model')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=int, default=1e-5, help='Control amount of weight change during optimization')
parser.add_argument('--load_discriminator', type=str, help='(Optinal) Folder contain discriminator model')
parser.add_argument('--load_inpaint', type=str, help='(Optinal) Folder contain 3 inpaints model, make sure to leave only target models')
parser.add_argument('--mask_type', type=int, default=1, help='Mask type: 1 for normal mask, 2 for large mask')
parser.add_argument('--train_path', type=str,  help='Training image folder')
parser.add_argument('--val_path', type=str,  help='(Optinal) Validation image folder')
parser.add_argument('--world_size', type=int, default=1, help='Number of training process (Should be equal to number of GPUs)')
parser.add_argument('--save_model', type=str,  help='(Optinal) Folder to save all models')

def train(rank, world_size, batch_size, epochs, lr, load_discriminator, load_inpaint, mask_type,
          train_path, val_path, save_model):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    coarse_model = CoarseNet().to(rank)
    super_resolution_model = SuperResolutionNet().to(rank)
    refinement_model = RefinementNet().to(rank)

    discriminator_model = Discriminator().to(rank)

    vgg19_model = vgg19(weights='IMAGENET1K_V1')
    vgg19_model = vgg19_model.features[:21].to(rank)

    ddp_coarse = DDP(coarse_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    ddp_super_resolution = DDP(super_resolution_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    ddp_refinement = DDP(refinement_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    ddp_discriminator = DDP(discriminator_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    train_dataset = JointDataset(train_path, mask_type)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    if val_path is not None and is_main_process():
        val_dataset = JointDataset(val_path, mask_type)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    if load_inpaint is not None:
        ddp_coarse.load_state_dict(torch.load(glob.glob(os.path.join(load_inpaint, 'coarse*'))[0], map_location=map_location))
        ddp_super_resolution.load_state_dict(torch.load(glob.glob(os.path.join(load_inpaint, 'super_resolution*'))[0],
                                                        map_location=map_location))
        ddp_refinement.load_state_dict(torch.load(glob.glob(os.path.join(load_inpaint, 'refinement*'))[0],
                                                  map_location=map_location))

    if load_discriminator is not None:
        ddp_discriminator.load_state_dict(torch.load(glob.glob(os.path.join(load_discriminator, 'discriminator*'))[0],
                                                     map_location=map_location))

    coarse_loss_function = Coarse_loss(vgg19_model, vgg_loss_weight=0.01)
    super_resolution_loss_function = L1_loss()
    refinement_loss_function = Joint_refinement_loss(vgg19_model, vgg_loss_weight = 1e-5, gan_loss_weight = 0.5,
                                                     gradient_loss_weight = 1)
    discriminator_loss_function = Discriminator_loss()

    inpaint_parameters = list(ddp_coarse.parameters()) + list(ddp_super_resolution.parameters()) + \
                         list(ddp_refinement.parameters())

    discriminator_optimizer = torch.optim.AdamW(ddp_discriminator.parameters(), lr = lr)
    inpaint_optimizer = torch.optim.AdamW(inpaint_parameters, lr = lr)

    scaler = torch.cuda.amp.GradScaler(init_scale=16834.0, enabled=True)

    if is_main_process():
        wandb.init(
            project="anime_inpaint_joint"
        )

    for epoch in range(epochs):
        train_dataloader.sampler.set_epoch(epoch)

        total_train_coarse_loss = 0
        total_train_super_resolution_loss = 0
        total_train_refinement_loss = 0
        total_train_inpaint_loss = 0
        total_train_discriminator_loss = 0

        total_val_coarse_loss = 0
        total_val_super_resolution_loss = 0
        total_val_refinement_loss = 0
        total_val_inpaint_loss = 0
        total_val_discriminator_loss = 0

        num_batch = len(train_dataloader)

        for batch, data in enumerate(train_dataloader):
            masked_image, mask, lr_groundtruth, hr_groundtruth = data
            masked_image, mask, lr_groundtruth, hr_groundtruth = masked_image.to(rank), mask.to(rank), \
                                                                 lr_groundtruth.to(rank), hr_groundtruth.to(rank)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                output = ddp_coarse(masked_image, mask)
                output = mask*output + (1-mask)*masked_image #Coarse output
                coarse_loss = coarse_loss_function(output, lr_groundtruth)

                output = ddp_super_resolution(output) #Super resolution output
                super_resolution_loss = super_resolution_loss_function(output, hr_groundtruth)

                mask = F.interpolate(mask, size = (512, 512), mode = 'nearest')
                output = ddp_refinement(output, mask)

                output = mask*output + (1-mask)*hr_groundtruth #Refinement output

                real_prediction = ddp_discriminator(hr_groundtruth, mask)
                fake_prediction = ddp_discriminator(output, mask)

                # print(f'epoch : {epoch}')
                # print(f'real prediction : {real_prediction}')
                # print(f'fake prediction : {fake_prediction}')
                # print('\n')

                discriminator_loss = discriminator_loss_function(real_prediction, fake_prediction)
                refinement_loss = refinement_loss_function(output, hr_groundtruth, fake_prediction)

                inpaint_loss = coarse_loss + super_resolution_loss + refinement_loss

                total_train_coarse_loss += coarse_loss.item()
                total_train_super_resolution_loss += super_resolution_loss.item()
                total_train_refinement_loss += refinement_loss.item()
                total_train_discriminator_loss += discriminator_loss.item()
                total_train_inpaint_loss += inpaint_loss.item()

            scaler.scale(discriminator_loss).backward(inputs = list(ddp_discriminator.parameters()), retain_graph=True)
            scaler.step(discriminator_optimizer)
            discriminator_optimizer.zero_grad(set_to_none=True)

            scaler.scale(inpaint_loss).backward(inputs = inpaint_parameters)
            scaler.step(inpaint_optimizer)
            inpaint_optimizer.zero_grad(set_to_none=True)

            scaler.update()

        if val_path is not None and is_main_process():
            for batch, data in enumerate(val_dataloader):
                masked_image, mask, lr_groundtruth, hr_groundtruth = data
                masked_image, mask, lr_groundtruth, hr_groundtruth = masked_image.to(rank), mask.to(rank), \
                    lr_groundtruth.to(rank), hr_groundtruth.to(rank)

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    output = ddp_coarse(masked_image, mask)
                    output = mask * output + (1 - mask) * masked_image  # Coarse output
                    coarse_loss = coarse_loss_function(output, lr_groundtruth)

                    output = ddp_super_resolution(output)  # Super resolution output
                    super_resolution_loss = super_resolution_loss_function(output, hr_groundtruth)

                    mask = F.interpolate(mask, size=(512, 512), mode='nearest')
                    output = ddp_refinement(output, mask)

                    output = mask * output + (1 - mask) * hr_groundtruth  # Refinement output

                    real_prediction = ddp_discriminator(hr_groundtruth, mask)
                    fake_prediction = ddp_discriminator(output, mask)

                    discriminator_loss = discriminator_loss_function(real_prediction, fake_prediction)
                    refinement_loss = refinement_loss_function(output, hr_groundtruth, fake_prediction)

                    inpaint_loss = coarse_loss + super_resolution_loss + refinement_loss

                    total_val_coarse_loss += coarse_loss.item()
                    total_val_super_resolution_loss += super_resolution_loss.item()
                    total_val_refinement_loss += refinement_loss.item()
                    total_val_discriminator_loss += discriminator_loss.item()
                    total_val_inpaint_loss += inpaint_loss.item()

        average_train_coarse_loss = total_train_coarse_loss / num_batch
        average_train_super_resolution_loss = total_train_super_resolution_loss / num_batch
        average_train_refinement_loss = total_train_refinement_loss / num_batch
        average_train_inpaint_loss = total_train_inpaint_loss / num_batch
        average_train_discriminator_loss = total_train_discriminator_loss / num_batch

        if is_main_process():
            if val_path is not None:
                average_val_coarse_loss = total_val_coarse_loss / num_batch
                average_val_super_resolution_loss = total_val_super_resolution_loss / num_batch
                average_val_refinement_loss = total_val_refinement_loss / num_batch
                average_val_inpaint_loss = total_val_inpaint_loss / num_batch
                average_val_discriminator_loss = total_val_discriminator_loss / num_batch
                wandb.log({"train_coarse_loss": average_train_coarse_loss,
                          "train_SR_loss": average_train_super_resolution_loss,
                           "train_refinement_loss": average_train_refinement_loss,
                          "train_total_loss": average_train_inpaint_loss,
                           "train_discriminator_loss": average_train_discriminator_loss,
                           "val_coarse_loss": average_val_coarse_loss,
                           "val_SR_loss": average_val_super_resolution_loss,
                           "val_refinement_loss": average_val_refinement_loss,
                           "val_total_loss": average_val_inpaint_loss,
                           "val_discriminator_loss": average_val_discriminator_loss
                           })
            else:
                wandb.log({"train_coarse_loss": average_train_coarse_loss,
                          "train_SR_loss": average_train_super_resolution_loss,
                           "train_refinement_loss": average_train_refinement_loss,
                          "train_total_loss": average_train_inpaint_loss,
                           "train_discriminator_loss": average_train_discriminator_loss})

        if save_model is not None and ((100 * (epoch + 1)) / epochs) % 10 == 0:
            torch.save(ddp_coarse.state_dict(), os.path.join(save_model, 'coarse_joint' + str(epoch + 1) + '.pt'))
            torch.save(ddp_super_resolution.state_dict(),
                       os.path.join(save_model, 'super_resolution_joint' + str(epoch + 1) + '.pt'))
            torch.save(ddp_refinement.state_dict(),
                       os.path.join(save_model, 'refinement_joint' + str(epoch + 1) + '.pt'))
            torch.save(ddp_discriminator.state_dict(),
                       os.path.join(save_model, 'discriminator_joint' + str(epoch + 1) + '.pt'))

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    cmd_args = parser.parse_args()

    assert cmd_args.train_path is not None

    mp.spawn(train, args = (cmd_args.world_size, cmd_args.batch_size, cmd_args.epochs, cmd_args.learning_rate,
                               cmd_args.load_discriminator, cmd_args.load_inpaint, cmd_args.mask_type,
                               cmd_args.train_path, cmd_args.val_path, cmd_args.save_model),
             nprocs = cmd_args.world_size, join=True)

    dist.destroy_process_group() #Clean up



