import argparse
from datasets import *
from network import *
import os
from torchvision.models import vgg19
from torch.utils.data import DataLoader
from utils.loss import *
import wandb

parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--batch_size', type=int, default = 8, help='Amount of data that pass simultaneously to model')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=int, default=1e-5, help='Control amount of weight change during optimization')
parser.add_argument('--load_model', type=str, help='(Optinal) Path to saved model')
parser.add_argument('--model', type=str, help='Model to train: coarse, super_resolution or refinement')
parser.add_argument('--num_workers', type=int, default=0, help='Amount of multiprocess in dataloader (0 mean use single process)')
parser.add_argument('--train_path', type=str,  help='Training image folder')
parser.add_argument('--val_path', type=str,  help='(Optinal) Validation image folder')
parser.add_argument('--save_model', type=str,  help='(Optinal) Folder to save model')

if __name__ == '__main__':
    cmd_args = parser.parse_args()

    assert cmd_args.model in ['coarse', 'super_resolution', 'refinement']

    if (cmd_args.model == 'coarse') or (cmd_args.model == 'refinement'):
        if cmd_args.model == 'coarse':
            model = CoarseNet().to('cuda')
        else:
            model = RefinementNet(use_gpu=True).to('cuda')

        vgg19_model = vgg19(weights='IMAGENET1K_V1').to('cuda')
        vgg19_model = vgg19_model.features[:21].to('cuda')

        loss_function = Coarse_loss(vgg19_model, vgg_loss_weight = 0.01)

        train_dataset = MaskedDataset(cmd_args.train_path, 1)
        train_dataloader = DataLoader(train_dataset, batch_size = cmd_args.batch_size, num_workers = cmd_args.num_workers)

        if cmd_args.val_path is not None:
            val_dataset = MaskedDataset(cmd_args.val_path, 1)
            val_dataloader = DataLoader(val_dataset, batch_size = cmd_args.batch_size, num_workers = cmd_args.num_workers)

    else:
        model = SuperResolutionNet().to('cuda')

        loss_function = L1_loss()

        train_dataset = SuperResolutionDataset(cmd_args.train_path)
        train_dataloader = DataLoader(train_dataset, batch_size=cmd_args.batch_size, num_workers=cmd_args.num_workers)

        if cmd_args.val_path is not None:
            val_dataset = SuperResolutionDataset(cmd_args.val_path)
            val_dataloader = DataLoader(val_dataset, batch_size = cmd_args.batch_size, num_workers = cmd_args.num_workers)

    if cmd_args.load_model is not None:
        model.load_state_dict(torch.load(cmd_args.load_model))

    optimizer = torch.optim.AdamW(model.parameters(), lr = cmd_args.learning_rate)

    scaler = torch.cuda.amp.GradScaler(init_scale = 16834.0, enabled=True)

    wandb.init(
        project="anime_inpaint_" + cmd_args.model
    )
    wandb.define_metric("train_loss", goal="minimize")
    wandb.define_metric("val_loss", goal="minimize")

    for epoch in range(cmd_args.epochs):
        total_train_loss = 0
        total_val_loss = 0
        num_batch = len(train_dataloader)

        for batch, data in enumerate(train_dataloader):
            if (cmd_args.model == 'coarse') or (cmd_args.model == 'refinement'):
                masked_image, mask, groundtruth = data
                masked_image, mask, groundtruth = masked_image.to('cuda'), mask.to('cuda'), groundtruth.to('cuda')

                with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = True):
                    output = model(masked_image, mask)
                    output = mask * output + (1 - mask) * masked_image

                    loss = loss_function(output, groundtruth)
                    total_train_loss += loss.item()

                scaler.scale(loss).backward(inputs=list(model.parameters()))
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad(set_to_none=True)

            else:
                resize_image, groundtruth = data
                resize_image, groundtruth = resize_image.to('cuda'), groundtruth.to('cuda')

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    output = model(resize_image)

                    loss = loss_function(output, groundtruth)
                    total_train_loss += loss.item()

                scaler.scale(loss).backward(inputs=list(model.parameters()))
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad(set_to_none=True)

        if cmd_args.val_path is not None:
            for batch, data in enumerate(val_dataloader):
                if (cmd_args.model == 'coarse') or (cmd_args.model == 'refinement'):
                    masked_image, mask, groundtruth = data
                    masked_image, mask, groundtruth = masked_image.to('cuda'), mask.to('cuda'), groundtruth.to('cuda')

                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        output = model(masked_image, mask)
                        output = mask * output + (1 - mask) * masked_image

                        loss = loss_function(output, groundtruth)
                        total_val_loss += loss.item()

                else:
                    resize_image, groundtruth = data
                    resize_image, groundtruth = resize_image.to('cuda'), groundtruth.to('cuda')

                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        output = model(resize_image)

                        loss = loss_function(output, groundtruth)
                        total_val_loss += loss.item()

        #log
        average_train_loss = total_train_loss / num_batch
        if cmd_args.val_path is not None:
            average_val_loss = total_val_loss / num_batch
            wandb.log({"train_loss": average_train_loss, "val_loss": average_val_loss})
        else:
            wandb.log({"train_loss": average_train_loss})

        #save per 10%
        if cmd_args.save_model is not None and ((100 * (epoch + 1)) / cmd_args.epochs) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(cmd_args.save_model, cmd_args.model + str(epoch + 1) + '.pt'))