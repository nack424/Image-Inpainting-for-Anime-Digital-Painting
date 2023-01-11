import os
import torch
from torch.utils.data import Dataset
from turbojpeg import TurboJPEG
from utils.crop import random_crop
from utils.mask import *

def preprocess(array):
    assert len(array.shape) == 3

    tensor = torch.as_tensor(array, dtype=torch.float32)
    tensor = torch.permute(tensor, (2, 0, 1))
    tensor = (tensor - 127.5) / 255.

    return tensor


def postprocess(tensor):
    assert len(tensor.shape) == 4

    output = torch.permute(tensor, (0, 2, 3, 1))
    output = (255. * output) + 127.5
    output = np.array(output, dtype='uint8')
    return output


# For training individual Coarse or Refinement network only
class MaskedDataset(Dataset):
    def __init__(self, path, mask_type):
        self.mask_type = mask_type
        self.image_list = []

        jpeg = TurboJPEG()

        for image_name in os.listdir(path):
            image_path = os.path.join(path, image_name)
            in_file = open(image_path, 'rb')
            original_image = jpeg.decode(in_file.read()).astype('uint8')

            self.image_list.append(original_image)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        original_image = self.image_list[index]

        min_shape = min(original_image.shape[0], original_image.shape[1]) #min height or width

        groundtruth = random_crop(original_image, round(min_shape/2), round(min_shape/2)) #Subject to change
        groundtruth = cv2.resize(groundtruth, (256, 256))

        if self.mask_type == 1:
            brush_amount = round(random.uniform(1, 6))  # Subject to be change

        elif self.mask_type == 2:
            brush_amount = round(random.uniform(1, 4))  # Subject to be change

        else:
            raise Exception("mask_type must be either 1 or 2")

        masked_image, mask = mask_image(groundtruth, mask_type=self.mask_type, brush_amount=brush_amount)

        masked_image = preprocess(masked_image)
        mask = torch.permute(torch.tensor(mask, dtype=torch.float32), (2, 0, 1))
        groundtruth = preprocess(groundtruth)

        return masked_image, mask, groundtruth


# For training individual SuperResolution network only
class SuperResolutionDataset(Dataset):
    def __init__(self, path):
        self.image_list = []

        jpeg = TurboJPEG()

        for image_name in os.listdir(path):
            image_path = os.path.join(path, image_name)
            in_file = open(image_path, 'rb')
            original_image = jpeg.decode(in_file.read()).astype('uint8')

            self.image_list.append(original_image)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        original_image = self.image_list[index]

        min_shape = min(original_image.shape[0], original_image.shape[1]) #min height or width

        groundtruth = random_crop(original_image, round(min_shape/2), round(min_shape/2)) #Subject to change
        groundtruth = cv2.resize(groundtruth, (128, 128))
        resize_image = cv2.resize(groundtruth, (64, 64))

        groundtruth, resize_image = preprocess(groundtruth), preprocess(resize_image)

        return resize_image, groundtruth

class JointDataset(Dataset):
    def __init__(self, path, mask_type):
        self.mask_type = mask_type
        self.image_list = []

        jpeg = TurboJPEG()

        for image_name in os.listdir(path):
            image_path = os.path.join(path, image_name)
            in_file = open(image_path, 'rb')
            original_image = jpeg.decode(in_file.read()).astype('uint8')

            self.image_list.append(original_image)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        original_image = self.image_list[index]

        min_shape = min(original_image.shape[0], original_image.shape[1]) #min height or width

        high_resolution_groundtruth = random_crop(original_image, round(min_shape/2), round(min_shape/2)) #Subject to change
        high_resolution_groundtruth = cv2.resize(high_resolution_groundtruth, (512, 512))
        low_resolution_groundtruth = cv2.resize(high_resolution_groundtruth, (256, 256), interpolation=cv2.INTER_CUBIC)

        if self.mask_type == 1:
            brush_amount = round(random.uniform(1, 6))  # Subject to be change

        elif self.mask_type == 2:
            brush_amount = round(random.uniform(1, 4))  # Subject to be change

        else:
            raise Exception("mask_type must be either 1 or 2")

        masked_image, mask = mask_image(low_resolution_groundtruth, mask_type=self.mask_type,
                                        brush_amount=brush_amount)

        masked_image = preprocess(masked_image)
        mask = torch.permute(torch.tensor(mask, dtype=torch.float32), (2, 0, 1))
        low_resolution_groundtruth = preprocess(low_resolution_groundtruth)
        high_resolution_groundtruth = preprocess(high_resolution_groundtruth)

        return masked_image, mask, low_resolution_groundtruth, high_resolution_groundtruth