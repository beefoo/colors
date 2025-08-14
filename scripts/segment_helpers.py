"""
Helpers for automatic image segmentation, adapted from:
https://github.com/yformer/EfficientSAM/blob/main/notebooks/EfficientSAM_segment_everything_example.ipynb
https://github.com/yformer/EfficientSAM/blob/main/notebooks/EfficientSAM_example.ipynb
"""
import shutil

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import requests
     
def bbox_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    x0, x1 = np.where(rows)[0][[0, -1]]
    y0, y1 = np.where(cols)[0][[0, -1]]
    return (x0, y0, x1, y1)

def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        with open(filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return filename


def get_image_segment(image, mask):
    x0, y0, x1, y1 = bbox_mask(mask)
    bounded_mask = mask[y0: y1, x0: x1]
    bounded_mask_img = Image.fromarray(np.uint8(bounded_mask * 255))

def get_segment_from_point_or_box(pil_image, pts_sampled, pts_labels, model):
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
    return torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()

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


def show_points(coords, ax, marker_size=600):
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )