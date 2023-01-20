import torch
from torch import nn

def reduce_sum(tensor, axis, keepdim=False):
    for i in sorted(axis, reverse=True):
        tensor = torch.sum(tensor, dim=i, keepdim=keepdim)

    return tensor

def reduce_mean(tensor, axis, keepdim=False):
    for i in sorted(axis, reverse=True):
        tensor = torch.mean(tensor, dim=i, keepdim=keepdim)

    return tensor

def same_padding(tensor, kernel_size, stride, dilation):
    assert len(tensor.shape) == 4

    image_shape = tensor.shape  # batch x channel x height x width

    output_height = (image_shape[2] + stride - 1) // stride
    output_width = (image_shape[3] + stride - 1) // stride

    padding_height = (stride * (output_height - 1) - image_shape[2] + dilation * (kernel_size - 1) + 1) / 2
    padding_width = (stride * (output_width - 1) - image_shape[3] + dilation * (kernel_size - 1) + 1) / 2

    padding_top = padding_bottom = max(round(padding_height), 0)
    padding_left = padding_right = max(round(padding_width), 0)

    padder = nn.ZeroPad2d((padding_left, padding_right, padding_top, padding_bottom))

    output = padder(tensor)

    return output


def pad_and_unfold(tensor, block_size, stride, dilation=1, padding='same'):
    assert len(tensor.shape) == 4 and padding in ['same', 'valid']

    if padding == 'same':
        tensor = same_padding(tensor, block_size, stride,
                              dilation)  # Make number of block equal to tensor width x tensor height

    unfold = nn.Unfold(block_size, dilation, 0, stride)

    output = unfold(tensor)

    return output  # Shape batch x no of numbers in each block x no of block(L)