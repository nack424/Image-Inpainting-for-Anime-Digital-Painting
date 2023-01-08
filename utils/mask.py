import cv2
import numpy as np
import random


# These function return 0,1 2D mask where 1 is masked area
def create_mask(mask_width, mask_length, brush_amount=1):
    mask = np.zeros((mask_width, mask_length), dtype='uint8')

    for brush in range(brush_amount):
        num_vertices = np.random.randint(1, 12)
        start_x = np.random.randint(0, mask_length)
        start_y = np.random.randint(0, mask_width)
        brush_thickness = np.random.randint(5, 30)
        positive_move_x, negative_move_x, positive_move_y, negative_move_y = 0, 0, 0, 0

        for i in range(num_vertices):
            angle = random.uniform(0, 2 * np.pi)
            if i % 2 == 0:
                angle = angle + random.uniform((7 * np.pi) / 8, (9 * np.pi) / 8)

            length = random.uniform(1, mask_length / 12)

            if positive_move_x:
                end_x = round(start_x + np.abs(length * np.cos(angle)))
            elif negative_move_x:
                end_x = round(start_x - np.abs(length * np.cos(angle)))
            else:
                end_x = round(start_x + length * np.cos(angle))

            if positive_move_y:
                end_y = round(start_y + np.abs(length * np.sin(angle)))
            elif negative_move_y:
                end_y = round(start_y - np.abs(length * np.sin(angle)))
            else:
                end_y = round(start_y + length * np.sin(angle))

            positive_move_x, negative_move_x, positive_move_y, negative_move_y = 0, 0, 0, 0

            end_x = max(end_x, 0)
            end_x = min(end_x, mask_length)
            end_y = max(end_y, 0)
            end_y = min(end_y, mask_width)

            # For prevent moving out of edge again
            if end_x == 0:
                positive_move_x = 1
            if end_x == mask_length:
                negative_move_x = 1
            if end_y == 0:
                positive_move_y = 1
            if end_y == mask_width:
                negative_move_y = 1

            cv2.circle(mask, (start_x, start_y), round(brush_thickness / 2), 255, -1)
            cv2.line(mask, (start_x, start_y), (end_x, end_y), 255, brush_thickness)

            start_x = end_x
            start_y = end_y

        cv2.circle(mask, (start_x, start_y), round(brush_thickness / 2), 255, -1)

    horizon_flip = random.choice([0, 1])
    vertical_flip = random.choice([0, 1])

    if horizon_flip:
        mask = cv2.flip(mask, 1)
    if vertical_flip:
        mask = cv2.flip(mask, 0)

    mask[mask > 0] = 1

    return mask


def create_large_mask(mask_width, mask_length, brush_amount=1):
    mask = np.zeros((mask_width, mask_length), dtype='uint8')

    for brush in range(brush_amount):
        num_vertices = np.random.randint(4, 12)
        start_x = np.random.randint(0, mask_length)
        start_y = np.random.randint(0, mask_width)
        brush_thickness = np.random.randint(12, 40)
        positive_move_x, negative_move_x, positive_move_y, negative_move_y = 0, 0, 0, 0

        for i in range(num_vertices):
            angle = (2 * np.pi) / 5 + random.uniform(random.uniform(-(2 * np.pi) / 15, 0),
                                                     random.uniform(0, (2 * np.pi) / 15))
            if i % 2 == 0:
                angle = 2 * np.pi - angle

            length = np.random.normal(mask_length / 8, mask_length / 16)

            if positive_move_x:
                end_x = round(start_x + np.abs(length * np.cos(angle)))
            elif negative_move_x:
                end_x = round(start_x - np.abs(length * np.cos(angle)))
            else:
                end_x = round(start_x + length * np.cos(angle))

            if positive_move_y:
                end_y = round(start_y + np.abs(length * np.sin(angle)))
            elif negative_move_y:
                end_y = round(start_y - np.abs(length * np.sin(angle)))
            else:
                end_y = round(start_y + length * np.sin(angle))

            positive_move_x, negative_move_x, positive_move_y, negative_move_y = 0, 0, 0, 0

            end_x = max(end_x, 0)
            end_x = min(end_x, mask_length)
            end_y = max(end_y, 0)
            end_y = min(end_y, mask_width)

            # For prevent moving out of edge again
            if end_x == 0:
                positive_move_x = 1
            if end_x == mask_length:
                negative_move_x = 1
            if end_y == 0:
                positive_move_y = 1
            if end_y == mask_width:
                negative_move_y = 1

            cv2.circle(mask, (start_x, start_y), round(brush_thickness / 2), 255, -1)
            cv2.line(mask, (start_x, start_y), (end_x, end_y), 255, brush_thickness)

            start_x, start_y = end_x, end_y

        cv2.circle(mask, (start_x, start_y), round(brush_thickness / 2), 255, -1)

    horizon_flip = random.choice([0, 1])
    vertical_flip = random.choice([0, 1])

    if horizon_flip:
        mask = cv2.flip(mask, 1)
    if vertical_flip:
        mask = cv2.flip(mask, 0)

    mask[mask > 0] = 1

    return mask


# For 3D image only, return masked image and 3D mask
def mask_image(image_array, mask_type=1, brush_amount=1):
    mask_width, mask_length = image_array.shape[0], image_array.shape[1]

    if mask_type == 1:
        mask = create_mask(mask_width, mask_length, brush_amount)
    elif mask_type == 2:
        mask = create_large_mask(mask_width, mask_length, brush_amount)
    else:
        raise Exception("mask_type must be either 1 or 2")

    mask_3D = np.repeat(np.expand_dims(mask, 2), 3, 2)

    masked_image = (1 - mask_3D) * image_array

    return masked_image, mask_3D