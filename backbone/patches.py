import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from config import ANOMALY_THRESHOLD
import backbone as b



def DivideInPatches(dataset, output_folder, size, stride, masks=False, save_image_patch=False):

    if save_image_patch:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)

    count = 1
    count_patch = 0

    if not masks:
        patches = []
        for n_tensor, i in enumerate(dataset):
            if n_tensor % 5 == 0 or n_tensor == 0:
                if save_image_patch:
                    path_folder = output_folder + "/{}/".format(
                        str(int(count)).zfill(2))
                    if not os.path.exists(path_folder):
                        os.makedirs(path_folder)
                    else:
                        shutil.rmtree(path_folder)
                        os.makedirs(path_folder)
                    count += 1
                    count_patch = 0
            temp = i.unfold(1, size, stride).unfold(2, size, stride)
            temp = temp.contiguous().view(temp.size(0), -1, size, size).permute(1,0,2,3)
            patches.append(temp)
            if save_image_patch:
                for n, t in enumerate(temp):
                    torchvision.utils.save_image(t, path_folder + str(int(count_patch)).zfill(4) + '.jpg')
                    count_patch += 1
        return patches
    else:
        patches = []
        mask_patches = []
        for i in dataset:

            if save_image_patch:
                path_folder = output_folder + "/{}/".format(str(int(count)).zfill(2))
                if not os.path.exists(path_folder):
                    os.makedirs(path_folder)
                else:
                    shutil.rmtree(path_folder)
                    os.makedirs(path_folder)

            temp = i[0].unfold(1, size, stride).unfold(2, size, stride)
            temp = temp.contiguous().view(temp.size(0), -1, size, size).permute(1, 0, 2, 3)
            patches.append(temp)
            temp2 = i[1].unfold(1, size, stride).unfold(2, size, stride)
            temp2 = temp2.contiguous().view(temp2.size(0), -1, size, size).permute(1, 0, 2, 3)
            mask_patches.append(temp2)
            if save_image_patch:
                for idx, t in enumerate(temp):
                    torchvision.utils.save_image(t, path_folder + str(int(idx)).zfill(4) + '.jpg')
            count += 1
        return patches, mask_patches


def AssemblePatches(patches_tensor, number_of_images, channel, height, width, patch_size, stride):
    temp = patches_tensor.contiguous().transpose(1,0).reshape(number_of_images, channel, -1, patch_size*patch_size).transpose(0,1)
    # print(temp.shape) # [number_of_images, C, number_patches_all, patch_size*patch_size]
    temp = temp.permute(0, 1, 3, 2)
    # print(temp.shape) # [number_of_images, C, patch_size*patch_size, number_patches_all]
    temp = temp.contiguous().reshape(number_of_images, channel*patch_size*patch_size, -1)
    # print(temp.shape) # [number_of_images, C*prod(kernel_size), L] as expected by Fold
    output = F.fold(temp, output_size=(height, width), kernel_size=patch_size, stride=stride)
    # print(output.shape) # [number_of_images, C, H, W]
    return output

def getPosition(number_of_images, original_height, original_width, patch_size):
    positions = []
    k = int(original_width/patch_size)
    for n in range(number_of_images):
        for i in range(int(original_height/patch_size)):
            for j in range(k):
                positions.append([n, i * k + j, patch_size * j, patch_size * i])   # number of image, number of patch, x-position, y-position
    return positions

def countAnomalies(test_patches, mask_test_patches, output_dir, save=False):
    number_of_defects = 0
    defective = []
    for idx, item in enumerate(mask_test_patches):
        if int(torch.sum(item)) > ANOMALY_THRESHOLD:
            number_of_defects += 1
            defective.append(True)
            if save:
                torchvision.utils.save_image(mask_test_patches[idx], b.assemble_pathname(str(idx) + '_Mask_patches_image', output_dir))
                torchvision.utils.save_image(test_patches[idx], b.assemble_pathname(str(idx) +'_patches_image', output_dir))
        else:
            defective.append(False)
    np.save(output_dir + 'frame_labels.npy',
            np.expand_dims(np.array([int(e==True) for e in defective]), 0))
    return number_of_defects, defective

def calculateNumberPatches(widths, heights, patch_size):
    number_of_patches = []
    for i in range(len(widths)):
        n = int((widths[i]*heights[i])/(patch_size*patch_size))
        number_of_patches.append(n)
    return number_of_patches