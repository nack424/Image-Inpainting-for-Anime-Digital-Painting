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


def tile_crop(image):
    """Crop image into multiple tiles with size of short_length x short_length"""

    short_length = min(image.shape[0], image.shape[1])
    long_length = max(image.shape[0], image.shape[1])

    tile_amount = int(math.ceil(long_length / short_length))

    tile_list = []

    if tile_amount == 1:
        tile_list.append(image)

    elif tile_amount > 1:
        crop_amount = tile_amount - 1

        if long_length == image.shape[0]:
            for i in range(crop_amount):
                tile = image[i * short_length:(i + 1) * short_length, 0:short_length]
                tile_list.append(tile)

            last_tile = image[image.shape[0] - short_length: image.shape[0], 0:short_length]
            tile_list.append(last_tile)

        else:
            for i in range(crop_amount):
                tile = image[0:short_length, i * short_length:(i + 1) * short_length]
                tile_list.append(tile)

            last_tile = image[0:short_length, image.shape[1] - short_length: image.shape[1]]
            tile_list.append(last_tile)

    else:
        raise Exception("Amount of tile to cut must be at least 1")

    return tile_list


def inverse_tile_crop(tile_list, long_length, long_side):
    short_length = tile_list[0].shape[0]

    if len(tile_list) == 1:
        image = tile_list[0]

        return image

    elif len(tile_list) > 1:
        if long_side == 0:
            image = np.zeros((long_length, short_length, 3))

            for i in range(len(tile_list) - 1):
                image[i * short_length:(i + 1) * short_length, 0:short_length] = tile_list[i]

            image[image.shape[0] - short_length:image.shape[0], 0:short_length] = tile_list[-1]

            return image

        elif long_side == 1:
            image = np.zeros((short_length, long_length, 3))

            for i in range(len(tile_list) - 1):
                image[0:short_length, i * short_length:(i + 1) * short_length] = tile_list[i]

            image[0:short_length, image.shape[1] - short_length:image.shape[1]] = tile_list[-1]

            return image

        else:
            raise Exception('Long side must be either 0 or 1')

    else:
        raise Exception('Tile list must have at least 1 tile')