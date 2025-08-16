"""
Configuration for scraping image galleries.
"""
from typing import Literal

# --- General Settings ---
SCRAPER_TYPE: Literal['requests'] = 'requests' # This scraper always uses the requests library

DB_NAME = "galleries.db"
CSV_NAME = "galleries.csv"

# URLs of the galleries to be downloaded
TARGET_GALLERY_URLS = [
    "https://some-gallery.org/1",
    # "https://some-gallery.org/2",
]

# Where to save the images
OUTPUT_FOLDER = "downloaded_images"
# True - create subfolders for each gallery, False - save all in one folder
CREATE_SUBFOLDERS = True 

# --- File Naming Settings ---
NAMING_CONFIG = {
    "mode": "original",  # "custom_prefix" or "original"
    "prefix": "My_Gallery", # Used only if mode = "custom_prefix"
    "start_counter": 1,
    "overwrite": False, # Overwrite existing files
}

# Delay between image downloads in milliseconds
DOWNLOAD_DELAY_MS = 200

# Database table structure
DATA_STRUCTURE = {
    "gallery_url": "TEXT",
    "image_page_url": "TEXT UNIQUE",
    "download_url": "TEXT",
    "original_filename": "TEXT",
    "saved_filepath": "TEXT",
    "status": "TEXT",
    "download_timestamp": "TEXT"
}