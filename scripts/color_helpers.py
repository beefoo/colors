"""
Helper functions for image processing and color analysis
"""

from colorsys import rgb_to_hsv, hsv_to_rgb
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy.cluster.vq import kmeans, kmeans2

def bbox_mask(mask):
    """
    Retrieve the bounding box of a mask
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = np.where(rows > 0)[0][[0, -1]]
    x0, x1 = np.where(cols > 0)[0][[0, -1]]
    return (x0, y0, x1, y1)

def draw_colors(colors, height = 40):
    """
    Draw colors as an image palette
    """
    width = len(colors) * height
    im = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(im)
    for i, color in enumerate(colors):
        x0 = i * height
        x1 = x0 + height
        y0 = 0
        y1 = height
        draw.rectangle([x0, y0, x1, y1], color)
    return im

def filter_colors(rgbs, property, value_range):
    """
    Filter a color such that the value (brightness or saturation) is within value_range.
    """
    min_v, max_v = value_range
    filtered = []
    for rgb in rgbs:
        h, s, v = rgb_to_hsv(*map(scale_down, rgb))
        if property == "saturation" and s >= min_v and s <= max_v:
            filtered.append(rgb)
        elif v >= min_v and v <= max_v:
            filtered.append(rgb)
    return filtered

def get_clipped_images_from_mask(img, mask, n, smoothing=12):
    """
    Returns clipped images from an image and mask
    """
    components, output, _mask = get_connected_components(mask, n)
    images = []
    for label, _size, _centroid in components:
        mask = np.zeros(output.shape, dtype=np.uint8)
        mask[output == label] = 255
        image = get_image_clip(img, mask, smoothing)
        images.append(image)
    return images

def get_colors(img):
    """
    Returns a list of all the image's colors.
    """
    w, h = img.size
    # convert('RGB') converts the image's pixels info to RGB 
    # getcolors() returns an unsorted list of (count, pixel) values
    # w * h ensures that maxcolors parameter is set so that each pixel could be unique
    # there are three values returned in a list
    img = img.convert('RGB')
    np_img = np.array(img)
    np_img = np.reshape(np_img, (w * h, 3))
    return np_img

def get_connected_components(img, max_number = -1, connectivity = 8):
    """
    Returns the largest components
    """
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity
    )
    sizes = stats[:, -1]
    components = []
    for i in range(1, nb_components):
        components.append((i, sizes[i], centroids[i]))
    components = sorted(components, key=lambda c: -c[1])
    if max_number > 0 and len(components) > max_number:
        components = components[:max_number]
    mask = np.zeros(output.shape, dtype=np.uint8)
    for label, _size, _centroid in components:
        mask[output == label] = 255
    mask = Image.fromarray(mask)
    return components, output, mask

# Adapted from: https://github.com/LibraryOfCongress/data-exploration/blob/master/loc.gov%20JSON%20API/Dominant%20colors.ipynb
def get_dominant_colors(img, n = 6, order_by = 'hue', brightness_range = (0.0, 1.0), saturation_range = (0.0, 1.0), size = 256):
    """
    Return most dominant 'n' colors
    """
    w, h = img.size
    thumb = img.copy()
    thumb_size = (size, size)
    thumb.thumbnail(thumb_size) # replace with a thumbnail with same aspect ratio, no larger than THUMB_SIZE
    tw, th = thumb.size
    # thumb = ImageOps.posterize(thumb, 4)
    rgbs = get_colors(thumb) # gets a list of RGB colors (e.g. (213, 191, 152)) for each pixel
    # adjust the value of each color, if you've chosen to change minimum and maximum values
    # rgbs = filter_colors(rgbs, "saturation", saturation_range)
    # rgbs = filter_colors(rgbs, "brightness", brightness_range)
    # turns the list of colors into a numpy array of floats, then applies scipy's k-means function
    clusters, labels = kmeans2(np.array(rgbs).astype(float), n)
    colors = order_colors(clusters, order_by) if order_by != 'none' else clusters
    # hex_colors = list(map(hexify, colors)) # turn RGB into hex colors for web
    labels = labels.reshape((th, tw))
    labels = Image.fromarray(labels)
    labels = labels.resize((w, h), Image.Resampling.NEAREST)
    labels = np.array(labels)
    return colors, labels

def get_image_clip(image, mask, smoothing=12):
    """
    Retrieve a clipped image from a mask
    """
    x0, y0, x1, y1 = bbox_mask(mask)
    bounded_mask_img = Image.fromarray(mask)
    bounded_mask_img = ImageOps.invert(bounded_mask_img)
    bounded_mask_img = bounded_mask_img.crop((x0, y0, x1, y1))
    # Smooth the edges
    bounded_mask_img = bounded_mask_img.filter(ImageFilter.ModeFilter(size=smoothing))
    cropped_image = image.crop((x0, y0, x1, y1))
    cropped_image = cropped_image.convert("RGBA")
    w, h = cropped_image.size
    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    clip_img = Image.composite(base, cropped_image, bounded_mask_img)
    return clip_img

def get_segments_from_color(img, color, count = 3, distance_threshold = 128):
    """
    Return connected segments that are around a particular color
    """
    ref_color = np.array(color, dtype=np.uint8)
    original_w, original_h = img.size
    thumb = img.copy()
    thumb_size = 512
    thumb.thumbnail((thumb_size, thumb_size))
    np_img = np.array(thumb, dtype=np.uint8)
    h, w, _ = np_img.shape
    color_mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            candidate_color = np_img[y, x]
            dist = np.linalg.norm(candidate_color - ref_color)
            if dist < distance_threshold:
                color_mask[y, x] = 255
    color_mask = Image.fromarray(color_mask)
    color_mask = color_mask.resize((original_w, original_h))
    np_color_mask = np.array(color_mask)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        np_color_mask, connectivity=8
    )
    sizes = stats[:, -1]
    components = []
    for i in range(1, nb_components):
        components.append({
            "label": i,
            "size": sizes[i],
            "centroid": [int(round(v)) for v in centroids[i]]
        })
    components = sorted(components, key=lambda c: -c["size"])
    if len(components) > count:
        components = components[:count]
    for i, c in enumerate(components):
        mask = np.zeros(output.shape, dtype=np.uint8)
        mask[output == c["label"]] = 255
        components[i]["mask"] = mask
    return components


def hexify(rgb):
    """
    Convert RGB to HEX
    """
    return "#{0:02x}{1:02x}{2:02x}".format(*rgb)

def order_colors(colors, order_by):
    """
    Orders colors by hue, saturation, or brightness.
    """
    index = 0
    if order_by == 'saturation':
        index = 1
    elif order_by == 'brightness':
        index = 2
    hsvs = [(i, rgb_to_hsv(*map(scale_down, color))) for i, color in enumerate(colors)]
    invert = -1 if order_by == "saturation" else 1
    hsvs.sort(key=lambda t: invert * t[1][index])
    return [(i, tuple(map(scale_up, hsv_to_rgb(*hsv)))) for i, hsv in hsvs]

def scale_down(x, scale = 255.0):
    """
    Scale down a number
    """
    return x / scale

def scale_up(x, scale = 255.0):
    """
    Scale up a number
    """
    return int(round(x * scale))
