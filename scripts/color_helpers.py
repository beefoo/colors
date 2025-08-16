"""
Helper functions for image processing and color analysis
"""

from colorsys import rgb_to_hsv, hsv_to_rgb
import cv2
import io
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import requests
from scipy.cluster.vq import kmeans

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

def get_clipped_images_from_mask(img, mask, n):
    """
    Returns clipped images from an image and mask
    """
    nb_components, output, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    sizes = stats[:, -1]
    components = []
    for i in range(1, nb_components):
        components.append((i, sizes[i]))
    components = sorted(components, key=lambda c: -c[1])
    if len(components) > n:
        components = components[:n]
    images = []
    for c in components:
        mask = np.zeros(output.shape, dtype=np.uint8)
        mask[output == c[0]] = 255
        image = get_image_clip(img, mask)
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
    return [color for count, color in img.convert('RGB').getcolors(w * h)]

# Adapted from: https://github.com/LibraryOfCongress/data-exploration/blob/master/loc.gov%20JSON%20API/Dominant%20colors.ipynb
def get_dominant_colors(img, n = 6, order_by = 'hue', brightness_range = (0.0, 1.0), saturation_range = (0.0, 1.0), size = 200):
    """
    Return most dominant 'n' colors
    """
    thumb = img.copy()
    thumb_size = (size, size)
    thumb.thumbnail(thumb_size) # replace with a thumbnail with same aspect ratio, no larger than THUMB_SIZE
    rgbs = get_colors(thumb) # gets a list of RGB colors (e.g. (213, 191, 152)) for each pixel
    # adjust the value of each color, if you've chosen to change minimum and maximum values
    rgbs = filter_colors(rgbs, "saturation", saturation_range)
    rgbs = filter_colors(rgbs, "brightness", brightness_range)
    # turns the list of colors into a numpy array of floats, then applies scipy's k-means function
    clusters, _ = kmeans(np.array(rgbs).astype(float), n)
    colors = order_colors(clusters, order_by) if order_by != 'none' else clusters
    # hex_colors = list(map(hexify, colors)) # turn RGB into hex colors for web
    return colors

def get_image_from_url(url):
    """
    Download and open an image from a URL to 
    """
    # Download the image to memory
    response = requests.get(url, timeout=60)
    image_filestream = io.BytesIO(response.content)

    # And read the image data
    im = Image.open(image_filestream)

    return im

def get_image_clip(image, mask):
    """
    Retrieve a clipped image from a mask
    """
    x0, y0, x1, y1 = bbox_mask(mask)
    bounded_mask_img = Image.fromarray(mask)
    bounded_mask_img = ImageOps.invert(bounded_mask_img)
    bounded_mask_img = bounded_mask_img.crop((x0, y0, x1, y1))
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
    hsvs = [rgb_to_hsv(*map(scale_down, color)) for color in colors]
    invert = -1 if order_by == "saturation" else 1
    hsvs.sort(key=lambda t: invert * t[index])
    return [tuple(map(scale_up, hsv_to_rgb(*hsv))) for hsv in hsvs]

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
