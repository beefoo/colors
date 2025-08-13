"""Helper functions to support scripts and notebooks"""

from colorsys import rgb_to_hsv, hsv_to_rgb
import io
from numpy import array
from PIL import Image, ImageDraw
import requests
from scipy.cluster.vq import kmeans

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
def get_dominant_colors(img, n = 6, order_by = 'hue', brightness_range = (0.0, 1.0), saturation_range = (0.0, 1.0), thumb = 200):
    """
    Return most dominant 'n' colors
    """
    thumb_size = (thumb, thumb)
    img.thumbnail(thumb_size) # replace with a thumbnail with same aspect ratio, no larger than THUMB_SIZE
    rgbs = get_colors(img) # gets a list of RGB colors (e.g. (213, 191, 152)) for each pixel
    # adjust the value of each color, if you've chosen to change minimum and maximum values
    rgbs = filter_colors(rgbs, "saturation", saturation_range)
    rgbs = filter_colors(rgbs, "brightness", brightness_range)
    # turns the list of colors into a numpy array of floats, then applies scipy's k-means function
    clusters, _ = kmeans(array(rgbs).astype(float), n)
    colors = order_colors(clusters, order_by) if order_by != 'none' else clusters
    hex_colors = list(map(hexify, colors)) # turn RGB into hex colors for web
    return hex_colors

def get_image_from_url(url):
    """Download and open an image from a URL to memory"""
    # Download the image to memory
    response = requests.get(url, timeout=60)
    image_filestream = io.BytesIO(response.content)

    # And read the image data
    im = Image.open(image_filestream)

    return im

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
