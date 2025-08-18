"""
Helpers for automatic image segmentation, adapted from:
https://github.com/yformer/EfficientSAM/blob/main/notebooks/EfficientSAM_segment_everything_example.ipynb
https://github.com/yformer/EfficientSAM/blob/main/notebooks/EfficientSAM_example.ipynb
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from segment_anything.utils.amg import (
    batched_mask_to_box,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms

def get_segment_from_point_or_box(pil_image, pts_sampled, pts_labels, model, min_area = 200, nms_thresh = 0.7):
    image_np = np.array(pil_image)
    img_tensor = ToTensor()(image_np)
    pts_sampled = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2])
    pts_labels = torch.reshape(torch.tensor(pts_labels), [1, 1, -1])
    predicted_logits, predicted_iou = model(
        img_tensor[None, ...],
        pts_sampled,
        pts_labels,
    )
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )
    mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
    # return mask
    masks = process_small_region([np.array([mask])], min_area, nms_thresh)
    return masks[0]

def process_small_region(masks, min_area = 200, nms_thresh = 0.7):
    rles = [mask_to_rle_pytorch(torch.from_numpy(m[0:1])) for m in masks]
    new_masks = []
    scores = []
    for rle in rles:
        mask = rle_to_mask(rle[0])

        mask, changed = remove_small_regions(mask, min_area, mode="holes")
        unchanged = not changed
        mask, changed = remove_small_regions(mask, min_area, mode="islands")
        unchanged = unchanged and not changed

        new_masks.append(torch.as_tensor(mask).unsqueeze(0))
        # Give score=0 to changed masks and score=1 to unchanged masks
        # so NMS will prefer ones that didn't need postprocessing
        scores.append(float(unchanged))

    # Recalculate boxes and remove any new duplicates
    masks = torch.cat(new_masks, dim=0)
    boxes = batched_mask_to_box(masks)
    keep_by_nms = batched_nms(
        boxes.float(),
        torch.as_tensor(scores),
        torch.zeros_like(boxes[:, 0]),  # categories
        iou_threshold=nms_thresh,
    )

    # Only recalculate RLEs for masks that have changed
    for i_mask in keep_by_nms:
        if scores[i_mask] == 0.0:
            mask_torch = masks[i_mask].unsqueeze(0)
            rles[i_mask] = mask_to_rle_pytorch(mask_torch)
    masks = [rle_to_mask(rles[i][0]) for i in keep_by_nms]
    return masks

def show_anns(mask, ax):
    ax.set_autoscale_on(False)
    img = np.ones((mask.shape[0], mask.shape[1], 4))
    img[:, :, 3] = 0
    color_mask = [0, 1, 0, 0.7]
    img[np.logical_not(mask)] = color_mask
    ax.imshow(img)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="yellow", facecolor=(0, 0, 0, 0), lw=5)
    )

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape((h, w, 1)) * color.reshape((1, 1, -1))
    ax.imshow(mask_image)


def show_points(coords, ax, marker_size=1200):
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )