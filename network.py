import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from utils.tensor import *

class GateConv2d(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels, kernel_size, stride=1, padding='same', dilation=1):
        #Important! conv_out_channel is output channel of convolution layer not output of entire gate convolution
        super(GateConv2d, self).__init__()
        self.conv = nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size, stride, padding, dilation)
        self.BatchNorm2d = nn.BatchNorm2d(conv_out_channels // 2)
        self.BatchNorm2d_3channels = nn.BatchNorm2d(3)
        self.ELU = nn.ELU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)

        if x.shape[1] == 3:
            output = self.BatchNorm2d_3channels(x)
            return output

        feature, gating = x[:, :x.shape[1] // 2, :, :], x[:, x.shape[1] // 2:, :, :]  # split channels into 2

        feature = self.ELU(feature)
        gating = self.Sigmoid(gating)

        output = feature * gating
        output = self.BatchNorm2d(output)

        return output


class ResidualGateBlock(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels, gate_amount):
        super(ResidualGateBlock, self).__init__()

        layers = []

        layers.append(GateConv2d(conv_in_channels, conv_out_channels, 3))
        for i in range(1, gate_amount):
            layers.append(GateConv2d(conv_out_channels // 2, conv_out_channels, 3))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block[0](x)
        residual = x.clone()

        for i in range(1, len(self.block)):
            x = self.block[i](x)

        output = x + residual

        return output


class DilationResidualGateBlock(nn.Module):
    def __init__(self, conv_in_channels, conv_out_channels, gate_amount, dilation_list):
        super(DilationResidualGateBlock, self).__init__()

        layers = []

        layers.append(GateConv2d(conv_in_channels, conv_out_channels, 3, padding=dilation_list[0]
                                 , dilation=dilation_list[0]))
        for i in range(1, gate_amount):
            layers.append(GateConv2d(conv_out_channels // 2, conv_out_channels, 3, padding=dilation_list[i]
                                     , dilation=dilation_list[i]))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block[0](x)
        residual = x.clone()

        for i in range(1, len(self.block)):
            x = self.block[i](x)

        output = x + residual

        return output


####Coarse network####
class CoarseNet(nn.Module):
    def __init__(self):
        super(CoarseNet, self).__init__()
        self.encoder_block1 = ResidualGateBlock(4, 64, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_block2 = ResidualGateBlock(32, 128, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck_block = DilationResidualGateBlock(64, 256, 3, [2, 4, 8])
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder_block1 = ResidualGateBlock(128, 128, 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder_block2 = ResidualGateBlock(64, 64, 4)
        self.last_layer = GateConv2d(32, 3, 3)

    def forward(self, masked_image, mask):
        x = torch.cat((masked_image, mask), dim=1)  # Shape batch x 4 x W x H

        x = self.encoder_block1(x)
        x = self.maxpool1(x)
        x = self.encoder_block2(x)
        x = self.maxpool2(x)
        x = self.bottleneck_block(x)
        x = self.upsample1(x)
        x = self.decoder_block1(x)
        x = self.upsample2(x)
        x = self.decoder_block2(x)
        output = self.last_layer(x)  # Shape batch x 3 x W x H

        return output


####Super resolution network####
class SuperResolutionResidualBlock(nn.Module):
    def __init__(self):
        super(SuperResolutionResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.ReLU(), nn.BatchNorm2d(64))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.ReLU(), nn.BatchNorm2d(64))

    def forward(self, x):
        residual = x.clone()

        x = self.conv1(x)
        x = self.conv2(x)

        output = x + residual

        return output

class SuperResolutionNet(nn.Module):
    def __init__(self):
        super(SuperResolutionNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.ReLU(), nn.BatchNorm2d(64))
        self.block1 = SuperResolutionResidualBlock()
        self.block2 = SuperResolutionResidualBlock()
        self.block3 = SuperResolutionResidualBlock()
        self.block4 = SuperResolutionResidualBlock()
        self.conv2 = nn.Sequential(nn.Conv2d(64, 256, 3, 1, 1, bias=False), nn.ReLU(), nn.BatchNorm2d(256))
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 3, 3, 1, 1, bias=False), nn.ReLU(), nn.BatchNorm2d(3))
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')

    def forward(self, x):
        global_residual = x.clone()  # Shape batch x 3 x W x H
        global_residual = self.upsample(global_residual)  # Shape batch x 3 x 2W x 2H

        x = self.conv1(x)  # Shape batch x 64 x W x H
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv2(x)  # Shape batch x 256 x W x H
        x = self.pixel_shuffle(x)  # Shape batch x 64 x 2W x 2H
        x = self.conv3(x)  # Shape batch x 3 x 2W x 2H

        output = x + global_residual

        return output


####Refinement network####
class ContextualAttention(nn.Module):
    def __init__(self, block_size, stride, attention_rate, softmax_scale=10, use_gpu=False):
        super(ContextualAttention, self).__init__()
        self.block_size = block_size
        self.stride = stride
        self.attention_rate = attention_rate
        self.softmax_scale = softmax_scale

        if use_gpu:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def forward(self, x, mask):
        x_shape = x.shape

        if mask.shape[2] != x.shape[2] or mask.shape[3] != x.shape[3]:
            mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]))

        x_clone = x.clone()
        mask_clone = mask.clone()

        raw_x_block_size = 2 * self.attention_rate

        x_unfold = pad_and_unfold(x, raw_x_block_size, self.attention_rate * self.stride)  # Shape N x k*k*C x L
        x_unfold = x_unfold.view(x_shape[0], x_shape[1], raw_x_block_size, raw_x_block_size,
                                 -1)  # Shape N x C x k x k x L
        x_unfold = torch.permute(x_unfold, (0, 4, 1, 2, 3))  # Shape N x L x C x k x k

        x_unfold_minibatch = torch.split(x_unfold, 1, dim=0)  # Filter for devoncolve

        x_resize = F.interpolate(x, scale_factor=1. / self.attention_rate, mode='nearest')
        x_resize_shape = x_resize.shape  # Shape N x C x H/rate x W/rate

        x_resize_minibatch = torch.split(x_resize, 1, dim=0)  # Input for convolve

        x_resize_unfold = pad_and_unfold(x_resize, self.block_size, self.stride)
        x_resize_unfold = x_resize_unfold.view(x_resize_shape[0], x_resize_shape[1], self.block_size, self.block_size,
                                               -1)
        x_resize_unfold = torch.permute(x_resize_unfold, (0, 4, 1, 2, 3))
        x_resize_unfold_minibatch = torch.split(x_resize_unfold, 1, dim=0)  # Filter for convolve

        mask_resize = F.interpolate(mask, scale_factor=1. / self.attention_rate, mode='nearest')
        mask_resize_shape = mask_resize.shape

        mask_resize_unfold = pad_and_unfold(mask_resize, self.block_size, self.stride)
        mask_resize_unfold = mask_resize_unfold.view(mask_resize_shape[0], mask_resize_shape[1], self.block_size,
                                                     self.block_size, -1)
        mask_resize_unfold = torch.permute(mask_resize_unfold, (0, 4, 1, 2, 3))
        mask_resize_unfold_minibatch = torch.split(mask_resize_unfold, 1, dim=0)  # For filter mask region

        output = []

        for x_input, x_filter, x_defilter, mask_filter in zip(x_resize_minibatch, x_resize_unfold_minibatch,
                                                              x_unfold_minibatch, mask_resize_unfold_minibatch):
            x_filter = x_filter[0]  # Shape L x C x k x k
            escape_nan = torch.FloatTensor([1e-4]).to(self.device)
            x_filter_normed = x_filter / torch.sqrt(
                reduce_sum(torch.pow(x_filter, 2) + escape_nan, axis=[1, 2, 3], keepdim=True))

            x_input = same_padding(x_input, self.block_size, 1, 1)

            attention = F.conv2d(x_input, x_filter_normed, stride=1)  # Shape 1 x L x H/rate x W/rate
            attention = attention.view(1, x_resize_shape[2] * x_resize_shape[3], x_resize_shape[2],
                                       x_resize_shape[3])  # Shape 1 x H/rate*W/rate x H/rate x W/rate

            mask_filter = mask_filter[0]  # Shape L x C x k x k
            mask_filter = (reduce_mean(mask_filter, axis=[1, 2, 3], keepdim=True) == 0.).to(torch.float32)
            mask_filter = torch.permute(mask_filter, (1, 0, 2, 3))  # Shape 1 x L x 1 x 1

            masked_attention = attention * mask_filter
            masked_attention = F.softmax(masked_attention * self.softmax_scale, dim=1)
            masked_attention = masked_attention * mask_filter  # Product softmax attention only for masked region

            x_defilter = x_defilter[0]

            output_i = F.conv_transpose2d(masked_attention, x_defilter, stride=self.attention_rate, padding=1)
            output.append(output_i)

        output = torch.cat(output, dim=0)
        output.contiguous().view(x_shape)

        output = mask_clone * output + (1 - mask_clone) * x_clone

        return output

class RefinementNet(nn.Module):
    def __init__(self, use_gpu=False):
        super(RefinementNet, self).__init__()
        self.encoder_block1 = ResidualGateBlock(4, 64, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_block2 = ResidualGateBlock(32, 128, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck_block = DilationResidualGateBlock(64, 256, 3, [2, 4, 8])
        self.gate1 = GateConv2d(128, 256, 3)
        self.contexual_attetion = ContextualAttention(3, 1, 2, 10, use_gpu)
        self.gate2 = GateConv2d(128, 256, 3)
        # Concat output of bottleneck and attention
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder_block1 = ResidualGateBlock(256, 128, 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder_block2 = ResidualGateBlock(64, 64, 4)
        self.last_layer = GateConv2d(32, 3, 3)

    def forward(self, masked_image, mask):
        x = torch.cat((masked_image, mask), dim=1)  # Shape batch x 4 x W x H

        x = self.encoder_block1(x)
        x = self.maxpool1(x)
        x = self.encoder_block2(x)
        x = self.maxpool2(x)
        x = self.bottleneck_block(x)
        bottleneck_output = x.clone()  # Shape batch x 128 x W/4 x H/4

        x = self.gate1(x)
        x = self.contexual_attetion(x, mask)
        x = self.gate2(x)
        x = torch.cat((bottleneck_output, x), dim=1)  # Shape batch x 256 x W/4 x H/4

        x = self.upsample1(x)
        x = self.decoder_block1(x)
        x = self.upsample2(x)
        x = self.decoder_block2(x)
        output = self.last_layer(x)  # Shape batch x 3 x W x H

        return output


####Discriminator####
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.spectral_conv1 = nn.Sequential(spectral_norm(nn.Conv2d(4, 64, 5, 2, 2, 1)), nn.LeakyReLU())
        self.spectral_conv2 = nn.Sequential(spectral_norm(nn.Conv2d(64, 128, 5, 2, 2, 1)), nn.LeakyReLU())
        self.spectral_conv3 = nn.Sequential(spectral_norm(nn.Conv2d(128, 256, 5, 2, 2, 1)), nn.LeakyReLU())
        self.spectral_conv4 = nn.Sequential(spectral_norm(nn.Conv2d(256, 256, 5, 2, 2, 1)), nn.LeakyReLU())
        self.spectral_conv5 = nn.Sequential(spectral_norm(nn.Conv2d(256, 256, 5, 2, 2, 1)), nn.LeakyReLU())
        self.classifier = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid())

    def forward(self, image, mask):
        x = torch.cat((image, mask), dim=1)  # Shape batch x 4 x 512 x 512

        x = self.spectral_conv1(x)  # Shape batch x 64 x 256 x 256
        x = self.spectral_conv2(x)  # Shape batch x 128 x 128 x 128
        x = self.spectral_conv3(x)  # Shape batch x 256 x 64 x 64
        x = self.spectral_conv4(x)  # Shape batch x 256 x 32 x 32
        x = self.spectral_conv5(x)  # Shape batch x 256 x 16 x 16

        x = x.view(-1, 4096)

        output = self.classifier(x)

        return output