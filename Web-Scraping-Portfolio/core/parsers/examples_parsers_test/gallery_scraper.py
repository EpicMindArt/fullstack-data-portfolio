import os
import re
import time
import requests
import sqlite3
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

from utils_scraping.models import Loggers
from utils_scraping.db_manager import save_items_to_db


def sanitize_filename(name: str) -> str:
    """Removes illegal characters from a string to make it a valid folder/file name."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace('～', '-').strip()
    return name[:120].strip() # Limit length to avoid issues with file systems


class GalleryParser:
    """
    Encapsulates parsing logic for a gallery website.
    If a different gallery site is targeted, a new parser class can be created.
    """
    @staticmethod
    def get_gallery_title(soup: BeautifulSoup) -> str | None:
        try:
            return soup.find('h1', id='gn').text.strip()
        except AttributeError:
            return None

    @staticmethod
    def get_gallery_metadata(soup: BeautifulSoup, logger: Loggers) -> dict:
        metadata = {'total_images': 0, 'total_pages': 1}
        try:
            # Find the total number of images
            gpc_text = soup.find('p', class_='gpc').text
            match = re.search(r'of ([\d,]+) images', gpc_text)
            if match:
                metadata['total_images'] = int(match.group(1).replace(',', ''))
        except (AttributeError, ValueError):
            logger.combined.warning("Could not parse the total number of images.")

        try:
            # Find the number of pagination pages
            pagination_cells = soup.select('table.ptt td')
            if len(pagination_cells) > 2:
                # Usually, the second to last element is a link to the last page
                last_page_link = pagination_cells[-2].find('a')
                if last_page_link and last_page_link.text.isdigit():
                    metadata['total_pages'] = int(last_page_link.text)
        except (AttributeError, ValueError, IndexError):
             logger.combined.warning("Could not parse pagination, assuming 1 page.")
        
        return metadata

    @staticmethod
    def get_preview_page_links(soup: BeautifulSoup, base_url: str) -> list[str]:
        """
        Finds all links to image preview pages.
        The current logic is based on finding all links inside a container with id='gdt'.
        """
        links = []
        
        preview_container = soup.find('div', id='gdt')
        if not preview_container:
            return [] 
        
        anchor_tags = preview_container.find_all('a')
    
        for tag in anchor_tags:
            if tag.has_attr('href'):
                # We don't need to do urljoin since the links are already absolute
                links.append(tag['href'])
            
        return links

    @staticmethod
    def get_image_download_url(soup: BeautifulSoup) -> str | None:
        try:
            return soup.find('img', id='img')['src']
        except (AttributeError, TypeError):
            return None


def download_image(session: requests.Session, url: str, save_path: Path, delay_ms: int, loggers: Loggers) -> dict:
    """
    Downloads a single image from a URL to a specified path.
    The file naming logic is handled outside this function.
    """
    result = {'status': 'failed', 'filepath': str(save_path), 'original_filename': None}
    try:
        time.sleep(delay_ms / 1000.0)
        
        # Overwrite logic is handled by the caller, simplifying this function.
        
        with session.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        result['status'] = 'downloaded'
        return result

    except requests.exceptions.HTTPError as e:
        if e.response.status_code in [403, 404]:
             # This can indicate a ban or an expired link
            result['status'] = 'banned_or_expired'
        loggers.combined.error(f"HTTP error downloading {url}: {e.response.status_code} {e.response.reason}")
        return result
    except requests.RequestException as e:
        loggers.combined.error(f"Network error downloading {url}: {e}")
        return result
    except Exception as e:
        loggers.combined.error(f"Unknown error downloading {url}: {e}")
        return result


def scrape_gallery_site(gallery_url: str, config: dict, conn: sqlite3.Connection, output_path: Path, loggers: Loggers):
    """
    Main function to coordinate the scraping of a single gallery.
    Features structured logic and improved error handling.
    """
    logger = loggers.combined
    parser = GalleryParser()
    
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    try:
        logger.info(f"=== Starting gallery processing: {gallery_url} ===")
        response = session.get(gallery_url, timeout=20)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        metadata = parser.get_gallery_metadata(soup, loggers)
        
        if metadata['total_images'] == 0:
            logger.error("Could not determine the number of images. Skipping gallery.")
            return


        # Determine the save folder explicitly
        if config["create_subfolders"]:
            title = parser.get_gallery_title(soup)
            subfolder_name = sanitize_filename(title) if title else f"gallery_{gallery_url.split('/')[-2]}"            
            gallery_folder = output_path / subfolder_name
        else:            
            gallery_folder = output_path
            
        gallery_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving files to: {gallery_folder}")
        logger.info(f"Found ~{metadata['total_images']} images across {metadata['total_pages']} pages.")

        pbar = tqdm(total=metadata['total_images'], unit="img", desc="Gallery", leave=False)
        
        file_counter = config["naming"].get("start_counter", 1)

        # Iterate through all pagination pages
        for page_num in range(metadata['total_pages']):
            pbar.set_description(f"Page {page_num + 1}/{metadata['total_pages']}")
            current_page_soup = soup
            if page_num > 0:
                page_url = f"{gallery_url}?p={page_num}"
                try:
                    response = session.get(page_url, timeout=20)
                    response.raise_for_status()
                    current_page_soup = BeautifulSoup(response.text, 'lxml')
                except requests.RequestException as e:
                    logger.error(f"Failed to load pagination page #{page_num + 1}: {e}. Skipping.")
                    continue

            preview_links = parser.get_preview_page_links(current_page_soup, gallery_url)
            for preview_link in preview_links:
                try:
                    prev_response = session.get(preview_link, timeout=20)
                    prev_response.raise_for_status()
                    prev_soup = BeautifulSoup(prev_response.text, 'lxml')
                    
                    download_url = parser.get_image_download_url(prev_soup)
                    if not download_url:
                        tqdm.write(f"WARNING: Download link not found on {preview_link}")
                        continue
                    
                    original_filename = os.path.basename(urlparse(download_url).path)
                    
                    # File naming logic is handled here, not in download_image
                    naming_mode = config["naming"]["mode"]
                    if naming_mode == 'custom_prefix':
                        ext = Path(original_filename).suffix
                        prefix = config["naming"]["prefix"]
                        final_filename = f"{prefix}_{file_counter:05d}{ext}"
                        file_counter += 1
                    else: # 'original' mode
                        final_filename = original_filename

                    save_path = gallery_folder / final_filename
                    
                    # File existence and overwrite logic is handled here
                    if save_path.exists() and not config["naming"]["overwrite"]:
                        tqdm.write(f"INFO: File {final_filename} already exists. Skipping.")
                        db_record = {"status": "skipped", "saved_filepath": str(save_path)}
                        pbar.update(1)
                    else:
                        pbar.set_description(f"Downloading {final_filename}")
                        download_result = download_image(session, download_url, save_path, config["delay_ms"], loggers)
                        
                        db_record = download_result
                        if download_result['status'] == 'downloaded':
                            pbar.update(1)
                        elif download_result['status'] == 'banned_or_expired':
                            # Handle ban/expired link by waiting
                            wait_time = 20 * 60 # 20 minutes
                            tqdm.write(f"ERROR: Access denied (403/404). Likely a ban or expired link. Waiting for {wait_time/60:.0f} min...")
                            time.sleep(wait_time)
                            tqdm.write("Resuming operation...")
                    
                    # Populate and save the database record
                    full_db_record = {
                        "gallery_url": gallery_url,
                        "image_page_url": preview_link,
                        "download_url": download_url,
                        "original_filename": original_filename,
                        "saved_filepath": db_record.get('saved_filepath'),
                        "status": db_record['status'],
                        "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    save_items_to_db(conn, [full_db_record], loggers)

                except requests.RequestException as e:
                    tqdm.write(f"ERROR: processing preview {preview_link}: {e}")
                except Exception as e:
                    tqdm.write(f"UNHANDLED ERROR: processing preview {preview_link}: {e}")

    except requests.RequestException as e:
        logger.critical(f"Critical error getting the main gallery page {gallery_url}: {e}")
    except Exception as e:
        logger.critical(f"Critical error processing gallery {gallery_url}: {e}", exc_info=True)
    finally:
        if 'pbar' in locals() and pbar:
            pbar.close()
        logger.info(f"=== Finished processing gallery {gallery_url} ===")