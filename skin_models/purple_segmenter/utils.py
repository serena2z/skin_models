import torch
import numpy as np
import random

def get_image_patches(image_tensor, patch_size=(224, 224)):
    b, c, h, w = image_tensor.size()
    patches = []
    for t in range(b):
        for i in range(0, h, patch_size[1]):
            for j in range(0, w, patch_size[0]):
                patch = image_tensor[t, :, i:i + patch_size[1], j:j + patch_size[0]]
                patches.append(patch)
    patches = torch.stack(patches, dim=0)
    return patches

def stitch_patches(patches, original_shape, patch_size=(224, 224)):
    b, c, h, w = original_shape
    reconstructed_images = torch.zeros((b, c, h, w))
    patch_idx = 0
    for img_idx in range(b):
        for i in range(0, h, patch_size[1]):
            for j in range(0, w, patch_size[0]):
                reconstructed_images[img_idx, :, i:i + patch_size[1], j:j + patch_size[0]] = patches[patch_idx]
                patch_idx += 1
    return reconstructed_images

def drop_empty_patches(images, masks, add_empty, empty_mask_value=0, shuffle=False, min_patches=2):
    idx_nonempty_masks = [x for x in range(len(masks)) if any(masks[x].unique() != empty_mask_value)]

    add_npatches = 0

    if len(idx_nonempty_masks) < min_patches:
        add_npatches = min_patches - len(idx_nonempty_masks)
    if add_empty > add_npatches:
        add_npatches += add_empty - add_npatches
    if (len(idx_nonempty_masks) + add_npatches) > len(masks):
        add_npatches = len(masks) - len(idx_nonempty_masks)

    idx_empty_masks = list(set(range(len(masks))) - set(idx_nonempty_masks))
    idx_addpatches = random.sample(idx_empty_masks, add_npatches)
    idx_nonempty_masks.extend(idx_addpatches)

    if shuffle:
        random.shuffle(idx_nonempty_masks)
    else:
        idx_nonempty_masks.sort()

    return images[idx_nonempty_masks], masks[idx_nonempty_masks]

def dice_score(seg, gt, k=1):
    intersection = np.sum(seg[gt==k]) * 2.0
    denominator = (np.sum(seg) + np.sum(gt))
    if (intersection == 0) and (denominator == 0):
        return 1
    elif (intersection != 0) and (denominator == 0):
        return 0
    else:
        return intersection / denominator