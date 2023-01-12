import numpy as np

def random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = y = 0

    if max_x > 0:
        x = np.random.randint(0, max_x)

    if max_y > 0:
        y = np.random.randint(0, max_y)

    crop = image[y : y + crop_height, x : x + crop_width]

    return crop