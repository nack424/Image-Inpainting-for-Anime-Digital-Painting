import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

class L1_loss:
    def __call__(self, predict, groundtruth):
        output = torch.abs(predict - groundtruth)
        output = torch.mean(output)
        return output


class VGG_loss:
    def __init__(self, vgg_model):
        self.vgg_model = vgg_model
    def __call__(self, predict, groundtruth):
        image_transform = transforms.Compose([transforms.Resize(size = (224, 224)),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        predict = image_transform(predict)
        groundtruth = image_transform(groundtruth)

        predict_vgg = self.vgg_model(predict)
        groundtruth_vgg = self.vgg_model(groundtruth)

        output = torch.abs(predict_vgg - groundtruth_vgg)
        output = torch.mean(output)
        return output


class Coarse_loss:
    def __init__(self, vgg_model, vgg_loss_weight = 0.01):
        self.l1_loss = L1_loss()
        self.vgg_loss_weight = vgg_loss_weight
        self.vgg_loss = VGG_loss(vgg_model)
    def __call__(self, predict, groundtruth):
        return self.l1_loss(predict, groundtruth) + self.vgg_loss_weight*self.vgg_loss(predict, groundtruth)


class Gradient_loss:
    def __init__(self):
        self.horizontal_filter = torch.tensor([
            [-1, 1]], dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1).unsqueeze(
            0)

        self.vertical_filter = torch.tensor([
            [-1],
            [1]], dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1).unsqueeze(
            0)

        self.horizontal_padder = nn.ZeroPad2d((0, 1, 0, 0)) #For same padding
        self.vertical_padder = nn.ZeroPad2d((0, 0, 0, 1))

    def __call__(self, predict, groundtruth):
        if predict.is_cuda:
            device = predict.device
        else:
            device = 'cpu'

        self.horizontal_filter, self.vertical_filter = self.horizontal_filter.to(device), self.vertical_filter.to(device)
        self.horizontal_padder, self.vertical_padder = self.horizontal_padder.to(device), self.vertical_padder.to(device)

        different = predict - groundtruth

        horizontal_grad = F.conv2d(self.horizontal_padder(different), self.horizontal_filter)
        vertical_grad = F.conv2d(self.vertical_padder(different), self.vertical_filter)

        output = torch.mean(torch.square(horizontal_grad) + torch.square(vertical_grad))/2

        return output


class Discriminator_loss:
    def __call__(self, real_prediction, fake_prediction):
        loss = torch.mean(F.relu(1 - real_prediction)) + torch.mean(F.relu(1 + fake_prediction))
        return loss

class Generator_loss:
    def __call__(self, fake_prediction):
        generator_loss = -torch.mean(fake_prediction)
        return generator_loss

class Joint_refinement_loss:
    def __init__(self, vgg_model, vgg_loss_weight = 1e-5, gan_loss_weight = 0.5, gradient_loss_weight = 1):
        self.coarse_loss = Coarse_loss(vgg_model, vgg_loss_weight)
        self.gan_loss = Generator_loss()
        self.gan_loss_weight = gan_loss_weight
        self.gradient_loss = Gradient_loss()
        self.gradient_loss_weight = gradient_loss_weight

    def __call__(self, predict, groundtruth, fake_prediction):
        return self.coarse_loss(predict, groundtruth) + self.gan_loss_weight*self.gan_loss(fake_prediction) + \
               self.gradient_loss_weight*self.gradient_loss(predict, groundtruth)