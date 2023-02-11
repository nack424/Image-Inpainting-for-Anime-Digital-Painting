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
parser.add_argument('--batch_size', type=int, default=8, help='Amount of data that pass simultaneously to model')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Control amount of weight change during optimization')
parser.add_argument('--load_model', type=str, help='(Optinal) Folder contain saved model (Model must match --model)')
parser.add_argument('--model', type=str, help='Model to train: coarse, super_resolution or refinement')
parser.add_argument('--save_model', type=str,  help='(Optinal) Folder to save model')
parser.add_argument('--train_path', type=str,  help='Training image folder')
parser.add_argument('--world_size', type=int, default=1, help='Number of training process (Should be equal to number of GPUs)')

def pretrain(rank, world_size, batch_size, epochs, lr, load_model, model_name, train_path, save_model):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    vgg19_model = vgg19(weights='IMAGENET1K_V1')
    vgg19_model = vgg19_model.features[:21].to(rank)

    if (model_name == 'coarse') or (model_name == 'refinement'):
        if model_name == 'coarse':
            model = CoarseNet().to(rank)
        else:
            model = RefinementNet().to(rank)

        loss_function = Coarse_loss(vgg19_model, vgg_loss_weight = 0.01)

        train_dataset = MaskedDataset(train_path, 1)
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, sampler=sampler)

    else:
        model = SuperResolutionNet().to(rank)

        loss_function = L1_loss()

        train_dataset = SuperResolutionDataset(train_path)
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    if load_model is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model.load_state_dict(torch.load(glob.glob(os.path.join(load_model, model_name + '*'))[0],
                                             map_location=map_location))

    ddp_model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

    scaler = torch.cuda.amp.GradScaler(init_scale = 16834.0, enabled=True)

    if is_main_process():
        wandb.init(
            project="anime_inpaint_" + model_name
        )

    for epoch in range(epochs):
        train_dataloader.sampler.set_epoch(epoch)

        total_train_loss = 0
        num_batch = len(train_dataloader)

        ddp_model.train()

        for batch, data in enumerate(train_dataloader):
            if (model_name == 'coarse') or (model_name == 'refinement'):
                masked_image, mask, groundtruth = data
                masked_image, mask, groundtruth = masked_image.to(rank), mask.to(rank), groundtruth.to(rank)

                with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = True):
                    output = ddp_model(masked_image, mask)
                    output = mask * output + (1 - mask) * masked_image

                    loss = loss_function(output, groundtruth)
                    total_train_loss += loss.item()

                scaler.scale(loss).backward(inputs=list(ddp_model.parameters()))
                scaler.step(optimizer)
                scaler.update()
                ddp_model.zero_grad(set_to_none=True)

            else:
                resize_image, groundtruth = data
                resize_image, groundtruth = resize_image.to(rank), groundtruth.to(rank)

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    output = ddp_model(resize_image)

                    loss = loss_function(output, groundtruth)
                    total_train_loss += loss.item()

                scaler.scale(loss).backward(inputs=list(ddp_model.parameters()))
                scaler.step(optimizer)
                scaler.update()
                ddp_model.zero_grad(set_to_none=True)

        if is_main_process():
            main_total_train_loss = [torch.tensor(0, dtype=torch.float32) for _ in range(world_size)]
            gather(torch.tensor(total_train_loss), main_total_train_loss)

        else:
            gather(torch.tensor(total_train_loss))

        if is_main_process(): #Only main process to record log
            average_train_loss = torch.mean(torch.tensor(main_total_train_loss)) / num_batch

            wandb.log({"train_loss": average_train_loss})

            #Main process save per 10%
            if save_model is not None and ((100 * (epoch + 1)) / epochs) % 10 == 0:
                torch.save(ddp_model.module.state_dict(), os.path.join(save_model, model_name + str(epoch + 1) + '.pt'))

    cleanup()

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    cmd_args = parser.parse_args()
    assert cmd_args.model in ['coarse', 'super_resolution', 'refinement']
    assert cmd_args.train_path is not None

    mp.spawn(pretrain, args = (cmd_args.world_size, cmd_args.batch_size, cmd_args.epochs, cmd_args.learning_rate,
                               cmd_args.load_model, cmd_args.model, cmd_args.train_path, cmd_args.save_model),
             nprocs = cmd_args.world_size, join=True)
