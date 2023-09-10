import argparse
from datasets import *
import glob
from network import *
import os
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--image_path', type=str, help='Folder contain test image')
parser.add_argument('--load_model', type=str, help='Folder contain saved model, make sure to leave only desire model')
parser.add_argument('--output_path', type=str, help='Path to save output image')
parser.add_argument('--model', type=str, help='Model to test: coarse, super_resolution, refinement or inpaint (All  model combined)')

if __name__ == '__main__':
    cmd_args = parser.parse_args()

    assert cmd_args.model in ['coarse', 'super_resolution', 'refinement', 'inpaint']

    if cmd_args.model == 'coarse':
        test_dataset = MaskedDataset(cmd_args.image_path, 1)
        model = CoarseNet().to('cuda')
        model.load_state_dict(torch.load(glob.glob(os.path.join(cmd_args.load_model, 'coarse*'))[0], map_location='cuda:0'))

    elif cmd_args.model == 'super_resolution':
        test_dataset = SuperResolutionDataset(cmd_args.image_path)
        model = SuperResolutionNet().to('cuda')
        model.load_state_dict(torch.load(glob.glob(os.path.join(cmd_args.load_model, 'super_resolution*'))[0], map_location='cuda:0'))

    elif cmd_args.model == 'refinement':
        test_dataset = MaskedDataset(cmd_args.image_path, 1)
        model = RefinementNet().to('cuda')
        model.load_state_dict(torch.load(glob.glob(os.path.join(cmd_args.load_model, 'refinement*'))[0], map_location='cuda:0'))

    else:
        test_dataset = JointDataset(cmd_args.image_path, 1)

        coarse_model = CoarseNet().to('cuda')
        super_resolution_model = SuperResolutionNet().to('cuda')
        refinement_model = RefinementNet().to('cuda')

        coarse_model.load_state_dict(torch.load(glob.glob(os.path.join(cmd_args.load_model, 'coarse*'))[0], map_location='cuda:0'))
        super_resolution_model.load_state_dict(
            torch.load(glob.glob(os.path.join(cmd_args.load_model, 'super_resolution*'))[0], map_location='cuda:0'))
        refinement_model.load_state_dict(torch.load(glob.glob(os.path.join(cmd_args.load_model, 'refinement*'))[0], map_location='cuda:0'))

    test_dataloader = DataLoader(test_dataset, batch_size=1)

    for batch, data in enumerate(test_dataloader):
        if cmd_args.model == 'coarse' or cmd_args.model == 'refinement':
            masked_image, mask, groundtruth = data
            masked_image, mask, groundtruth = masked_image.to('cuda'), mask.to('cuda'), groundtruth.to('cuda')

            output = model(masked_image, mask)
            output = mask*output + (1-mask)*masked_image

            masked_image = postprocess(masked_image.cpu())[0]
            output = postprocess(output.cpu().detach())[0]
            groundtruth = postprocess(groundtruth.cpu())[0]

            cv2.imwrite(os.path.join(cmd_args.output_path, 'groundtruth' + str(batch) + '.jpg'), groundtruth)
            cv2.imwrite(os.path.join(cmd_args.output_path, 'masked_img'+str(batch)+'.jpg'), masked_image)
            cv2.imwrite(os.path.join(cmd_args.output_path, 'output' + str(batch) + '.jpg'), output)

        elif cmd_args.model == 'super_resolution':
            resize_image, groundtruth = data
            resize_image, groundtruth = resize_image.to('cuda'), groundtruth.to('cuda')

            output = model(resize_image)

            output = postprocess(output.cpu().detach())[0]
            groundtruth = postprocess(groundtruth.cpu())[0]

            cv2.imwrite(os.path.join(cmd_args.output_path, 'groundtruth' + str(batch) + '.jpg'), groundtruth)
            cv2.imwrite(os.path.join(cmd_args.output_path, 'output' + str(batch) + '.jpg'), output)

        else:
            masked_image, mask, _, hr_groundtruth = data
            masked_image, mask, hr_groundtruth = masked_image.to('cuda'), mask.to('cuda'), hr_groundtruth.to('cuda')

            coarse_output = coarse_model(masked_image, mask)
            coarse_output = mask * coarse_output + (1 - mask) * masked_image

            super_resolution_output = super_resolution_model(coarse_output)

            mask = F.interpolate(mask, size=(512, 512), mode='nearest')
            refinement_output = refinement_model(super_resolution_output, mask)
            refinement_output = mask * refinement_output + (1 - mask) * hr_groundtruth

            coarse_output = postprocess(coarse_output.cpu().detach())[0]
            super_resolution_output = postprocess(super_resolution_output.cpu().detach())[0]
            refinement_output = postprocess(refinement_output.cpu().detach())[0]
            groundtruth = postprocess(hr_groundtruth.cpu())[0]

            cv2.imwrite(os.path.join(cmd_args.output_path, 'groundtruth' + str(batch) + '.jpg'), groundtruth)
            cv2.imwrite(os.path.join(cmd_args.output_path, 'coarse_output' + str(batch) + '.jpg'), coarse_output)
            cv2.imwrite(os.path.join(cmd_args.output_path, 'super_resolution_output' + str(batch) + '.jpg'), super_resolution_output)
            cv2.imwrite(os.path.join(cmd_args.output_path, 'refinement_output' + str(batch) + '.jpg'), refinement_output)