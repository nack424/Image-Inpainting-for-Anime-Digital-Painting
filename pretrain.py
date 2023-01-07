import argparse
from datasets import *
import deepspeed
from network import *
import os
from torchvision.models import vgg19
from utils.loss import *
import wandb

parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--epochs', type=int, help='Number of epochs to train')
parser.add_argument('--load_model', type=str, help='(Optinal) Path to saved model')
parser.add_argument('--model', type=str, help='Model to train: coarse, super_resolution or refinement')
parser.add_argument('--train_path', type=str,  help='Training image folder')
parser.add_argument('--save_model', type=str,  help='(Optinal) Folder to save model')
#Note: for change training parameters, edit deepspeed_config.

if __name__ == '__main__':
    cmd_args = parser.parse_args()

    assert cmd_args.model in ['coarse', 'super_resolution', 'refinement']

    if cmd_args.model == 'coarse' or 'refinement':
        if cmd_args.model == 'coarse':
            model = CoarseNet().to('cuda')
        else:
            model = RefinementNet().to('cuda')

        vgg19_model = vgg19(weights='IMAGENET1K_V1').to('cuda')
        vgg19_model = vgg19_model.features[:21].to('cuda')

        loss_function = Coarse_loss(vgg19_model, vgg_loss_weight = 0.01)

        trainset = MaskedDataset(cmd_args.train_path, 1)

    else:
        model = SuperResolutionNet().to('cuda')

        loss_function = L1_loss()

        trainset = SuperResolutionDataset(cmd_args.train_path)

    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(config = 'pretrain_deepspeed_config.json', model=model,
                                                         model_parameters=model.parameters(), training_data = trainset)

    if cmd_args.load_model is not None:
        model_engine.load_checkpoint(load_dir = cmd_args.load_model)

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
            if cmd_args.model == 'coarse' or cmd_args.model == 'refinement':
                masked_image, mask, groundtruth = data
                masked_image, mask, groundtruth = masked_image.to('cuda'), mask.to('cuda'), groundtruth.to('cuda')

                output = model_engine(masked_image, mask)
                output = mask * output + (1 - mask) * masked_image

                loss = loss_function(output, groundtruth)
                wandb.log({"train_loss": loss})

                model_engine.backward(loss)

                model_engine.step()

            else:
                resize_image, groundtruth = data
                resize_image, groundtruth = resize_image.to('cuda'), groundtruth.to('cuda')

                output = model_engine(resize_image)

                loss = loss_function(output, groundtruth)
                wandb.log({"train_loss": loss})

                model_engine.backward(loss)

                model_engine.step()

        if cmd_args.save_model is not None and ((100 * (epoch + 1)) / cmd_args.epochs) % 10 == 0:
            model_engine.save_checkpoint(save_dir = os.path.join(cmd_args.save_model, cmd_args.model + '_' + str(epoch)),
                                         save_latest = False)