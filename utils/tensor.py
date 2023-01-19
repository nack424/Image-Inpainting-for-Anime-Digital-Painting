import math
import torch
from torch import nn
import torch.nn.functional as F

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = F.unfold(x, block_size, stride=block_size)

    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)


def extract_image_patches(x, patch_size, strides):
    if x.is_cuda:
        device = x.device
    else:
        device = 'cpu'

    channels = x.shape[1]
    kernel_size = patch_size * patch_size * channels
    kernel = torch.reshape(torch.eye(kernel_size, dtype=x.dtype),
                           (-1, channels, patch_size, patch_size)).to(device)
    patches = F.conv2d(x, kernel, stride=strides, padding=patch_size // 2)
    return patches


def get_attention_map(x, mask, propagation_size=3, softmax_scale=10, patch_size=3):
    if x.is_cuda:
        device = x.device
    else:
        device = 'cpu'

    b, c, h, w = x.shape

    if mask.shape[2] != x.shape[2] or mask.shape[3] != x.shape[3]:
        mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]))

    patches = extract_image_patches(x, patch_size, 1)  # Shape B x P*P*C x H x W
    patches = patches.permute(0, 2, 3, 1)

    # Normalizing patches.  (B x H x W x P*P*C)
    patches_normalized = patches / torch.max(torch.norm(patches, dim=-1, keepdim=True), torch.tensor(1e-9).to(device))

    # Transpose inverted mask.  (B x P*P*C x H*W)
    patches_transposed = patches.reshape(b, h * w, 3 * 3 * c).permute(0, 2, 1)

    # (B x H*W x P*P*C)
    patches_normalized_reshaped = patches_normalized.reshape(b, h * w, patch_size * patch_size * c)

    # (B x H*W x H*W)
    attention_map = torch.matmul(patches_normalized_reshaped, patches_transposed)

    if propagation_size > 0:
        # Attention propagation.
        prop_weight = torch.eye(propagation_size).reshape(1, 1, propagation_size, propagation_size).to(device)

        proped_horizontally = nn.functional.conv2d(attention_map.unsqueeze(1), prop_weight, stride=1, padding=1)
        proped_horizontally = proped_horizontally.permute(0, 2, 3, 1)

        transposed = proped_horizontally.reshape(b, h, w, h, w).permute(0, 2, 1, 4, 3).reshape(b, w * h, w * h, 1)
        transposed = transposed.permute(0, 3, 1, 2)

        proped_vertically = nn.functional.conv2d(transposed, prop_weight, stride=1, padding=1)
        proped_vertically = proped_vertically.permute(0, 2, 3, 1)

        attention_map = proped_vertically.reshape(b, w, h, w, h).permute(0, 2, 1, 4, 3).reshape(b, h * w, h * w)

    mask_filter = nn.functional.max_pool2d(mask, 3, 1, 1).permute(0, 2, 3, 1).reshape(attention_map.shape[0], 1, -1)

    if softmax_scale > 0.0:
        attention_map_scaled = attention_map * softmax_scale
        attention_map_normalized = attention_map_scaled - torch.max(attention_map_scaled, dim=-1, keepdim=True)[0]
        attention_map_exp = torch.exp(attention_map_normalized) * (1.0 - mask_filter)
        attention_map = attention_map_exp / torch.maximum(torch.sum(attention_map_exp, dim=-1, keepdim=True),
                                                          torch.full((attention_map_exp.shape[0],
                                                                      attention_map_exp.shape[1], 1), 1e-09).to(device))

    return attention_map


def apply_attention_map(x, attention, mask):
    if mask.shape[2] != x.shape[2] or mask.shape[3] != x.shape[3]:
        mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]))

    b, c, h, w = x.shape

    attention_size = attention.shape[1]
    square_block_size = h * w // attention_size

    if square_block_size * attention_size != h * w:
        raise ValueError(
            'Invalid shape. The multiplication of the input height({}) and '
            'width({}) must be a multiple of the second dimension size({}) of '
            'the attention map.'.format(h, w, attention_size))

    block_size = int(math.sqrt(square_block_size))

    if block_size * block_size != square_block_size:
        raise ValueError(
            'Invalid shape. The multiplication of the input height and width '
            'divided by the number of the second dimension of the attention map'
            '({}) must be a square of an integer.'.format(square_block_size))

    if block_size > 1:
        depth = space_to_depth(x, block_size)
    else:
        depth = x

    depth = depth.permute(0, 2, 3, 1)

    h, w = depth.shape[1], depth.shape[2]
    right = depth.reshape(b, -1, c * block_size * block_size)

    mult = torch.matmul(attention, right).reshape(b, h, w, -1)

    if block_size > 1:
        mult = F.pixel_shuffle(mult.permute(0, 3, 1, 2), block_size)

    else:
        mult = mult.permute(0, 3, 1, 2)

    return x * (1.0 - mask) + mult * mask