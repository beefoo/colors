"""
Helpers for reading/writing files
"""
import io
import shutil

from PIL import Image
import requests

def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        with open(filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return filename

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