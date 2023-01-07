import torch
import torch.nn.functional as F

class L1_loss:
    def __call__(self, predict, groundtruth):
        output = torch.abs(predict - groundtruth)
        output = torch.mean(output)
        return output


class VGG_loss:
    def __init__(self, vgg_model):
        self.vgg_model = vgg_model
    def __call__(self, predict, groundtruth):
        predict_vgg = self.vgg_model(predict)
        groundtruth_vgg = self.vgg_model(groundtruth)

        output = torch.abs(predict_vgg - groundtruth_vgg)
        output = torch.mean(output)
        return output


class Coarse_loss:
    def __init__(self, vgg_model, vgg_loss_weight = 0.01):
        self.L1_loss = L1_loss()
        self.vgg_loss_weight = vgg_loss_weight
        self.VGG_loss = VGG_loss(vgg_model)
    def __call__(self, predict, groundtruth):
        return self.L1_loss(predict, groundtruth) + self.vgg_loss_weight*self.VGG_loss(predict, groundtruth)


class Gradient_loss:
    def __init__(self, use_gpu = False):
        if use_gpu:
            device = 'cuda'
        else:
            device = 'cpu'

        self.horizontal_filter = torch.tensor([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]], dtype=torch.float32, device = device).unsqueeze(0).repeat(3, 1, 1).unsqueeze(
            0)  # Not sure if it same as paper

        self.vertical_filter = torch.tensor([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]], dtype=torch.float32, device = device).unsqueeze(0).repeat(3, 1, 1).unsqueeze(
            0)  # Not sure if it same as paper

    def __call__(self, predict, groundtruth):
        assert predict.shape == groundtruth.shape

        predict_horizontal_grad = F.conv2d(predict, self.horizontal_filter, padding = 'same')
        predict_vertical_grad = F.conv2d(predict, self.vertical_filter, padding = 'same')

        groundtruth_horizontal_grad = F.conv2d(groundtruth, self.horizontal_filter, padding = 'same')
        groundtruth_vertical_grad = F.conv2d(groundtruth, self.vertical_filter, padding = 'same')

        horizontal_grad_diff = predict_horizontal_grad - groundtruth_horizontal_grad #Shape batch x 1 x height x width
        vertical_grad_diff = predict_vertical_grad - groundtruth_vertical_grad

        num_pixels = horizontal_grad_diff.shape[0] * horizontal_grad_diff.shape[2] * horizontal_grad_diff.shape[3]

        output = (torch.sum(torch.square(horizontal_grad_diff)) + torch.sum(torch.square(vertical_grad_diff)))/(2 * num_pixels)
        #์Not sure about loss formula

        return output


class Discriminator_loss:
    def __call__(self, real_prediction, fake_prediction):
        #discriminator_pred shape batch x length
        real_batch_size = real_prediction.shape[0]
        fake_batch_size = fake_prediction.shape[0]

        real_loss = 0
        fake_loss = 0

        for prediction in real_prediction:
            real_loss += F.relu(1 - prediction)

        for prediction in fake_prediction:
            fake_loss += F.relu(1 + prediction)

        real_loss = real_loss/real_batch_size
        fake_loss = fake_loss/fake_batch_size

        discriminator_loss = real_loss + fake_loss

        return discriminator_loss

class Generator_loss:
    def __call__(self, fake_prediction):
        batch_size = fake_prediction.shape[0]

        generator_loss = 0

        for prediction in fake_prediction:
            generator_loss += -prediction

        generator_loss = generator_loss/batch_size

        return generator_loss