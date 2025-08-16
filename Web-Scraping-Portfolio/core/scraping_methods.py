import pkgutil
import inspect
import importlib
import os

import requests
import sqlite3
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup

from utils_scraping.models import Loggers
from utils_scraping.db_manager import save_items_to_db
from .parsers.base_parser import AbstractParser
from .browser_manager import BrowserManager


def _get_parser_instance(parser_name: str, loggers: Loggers, **kwargs) -> AbstractParser:
    """
    Dynamically finds and creates an instance of a parser class.
    It recursively scans all modules inside the 'core.parsers' package and its subdirectories,
    making manual registration of new parsers unnecessary.
    """
    logger = loggers.combined
    parsers_package_path = "core.parsers"
    # Convert the package path to a path understandable to the OS
    os_package_path = parsers_package_path.replace('.', os.sep)
    
    try:
        # 1. Recursively go through all directories and files, starting with 'core/parsers'
        for root, _, files in os.walk(os_package_path):
            for filename in files:                
                if filename.endswith(".py") and filename != "__init__.py":
                    # 3. Collect the full path to the module in Python format (with dots)
                    # Convert 'core/parsers/client1' to 'core.parsers.client1'
                    module_path_in_py_format = root.replace(os.sep, '.')
                    module_name = filename[:-3] # Remove '.py'
                    full_module_path = f"{module_path_in_py_format}.{module_name}"
                    
                    # 4. Dynamically import the module
                    module = importlib.import_module(full_module_path)

                    # 5. We search inside the module for the class we need
                    for member_name, member_obj in inspect.getmembers(module, inspect.isclass):
                        if member_name == parser_name and issubclass(member_obj, AbstractParser) and member_obj is not AbstractParser:
                            logger.info(f"Found and instantiated parser class '{parser_name}' from module '{full_module_path}'.")
                            return member_obj(logger=loggers, **kwargs)

        # If we went through all the files and found nothing
        raise AttributeError(f"Parser class '{parser_name}' not found in any module within '{parsers_package_path}' or its subdirectories.")

    except (AttributeError, ImportError) as e:
        logger.error(f"Failed to find or instantiate parser class '{parser_name}': {e}", exc_info=True)
        raise


def _extract_form_data(soup: BeautifulSoup, loggers: Loggers) -> dict | None:
    """Helper function to extract ViewState form data."""
    form_data = {}
    try:
        form_data['__VIEWSTATE'] = soup.find('input', {'name': '__VIEWSTATE'})['value']
        
        # These fields may not always be present
        if gen_tag := soup.find('input', {'name': '__VIEWSTATEGENERATOR'}):
            form_data['__VIEWSTATEGENERATOR'] = gen_tag['value']
        if val_tag := soup.find('input', {'name': '__EVENTVALIDATION'}):
            form_data['__EVENTVALIDATION'] = val_tag['value']
            
        return form_data
    except (TypeError, KeyError):
        loggers.combined.error("Key field '__VIEWSTATE' not found on the page.")
        return None


def scrape_viewstate_with_requests(conn: sqlite3.Connection, start_url: str, loggers: Loggers, config: dict, parser_class_name: str):
    """
    Scrapes sites with dependent ViewState filters using the `requests` library.
    """
    logger = loggers.combined
    logger.info(f"Starting ViewState scrape via Requests: {start_url}")

    parser = _get_parser_instance(parser_class_name, loggers)

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0",
        "Referer": start_url
    })
    
    filter_url = config["filter_url"]
    primary_conf = config["primary_filter"]
    secondary_conf = config["secondary_filter"]

    try:
        initial_response = session.get(start_url)
        initial_response.raise_for_status()
        initial_soup = BeautifulSoup(initial_response.text, 'lxml')
        initial_form_data = _extract_form_data(initial_soup, loggers)
        if not initial_form_data: return

        author_options = initial_soup.select(primary_conf["option_selector"])
        all_authors = [opt['value'] for opt in author_options if opt.has_attr('value') and opt.get('value') != primary_conf["skip_option_value"]]
        
        authors_to_process = [primary_conf["target_value"]] if primary_conf.get("target_value") else all_authors
        logger.info(f"Found {len(authors_to_process)} authors to process.")

        for i, author in enumerate(authors_to_process):
            logger.info(f"[{i+1}/{len(authors_to_process)}] === Processing author: '{author}' ===")
            
            try:
                payload_for_tags = initial_form_data.copy()
                payload_for_tags[primary_conf["selector"].strip('#')] = author
                
                response_with_tags = session.post(filter_url, data=payload_for_tags, timeout=15)
                soup_with_tags = BeautifulSoup(response_with_tags.text, 'lxml')
                form_data_for_author = _extract_form_data(soup_with_tags, loggers)
                if not form_data_for_author: continue
                
                tag_options = soup_with_tags.select(secondary_conf["option_selector"])
                tags_for_author = [opt['value'] for opt in tag_options if opt.has_attr('value') and opt.get('value') != secondary_conf["skip_option_value"]]
                logger.info(f"  -> Found {len(tags_for_author)} tags.")

                if not tags_for_author:
                    parsed_data = parser.parse_items(soup_with_tags)
                    if parsed_data: save_items_to_db(conn, parsed_data, loggers)
                    continue

                for j, tag in enumerate(tags_for_author):
                    payload_for_quotes = form_data_for_author.copy()
                    payload_for_quotes.update({
                        primary_conf["selector"].strip('#'): author,
                        secondary_conf["selector"].strip('#'): tag,
                        'submit_button': 'Search'
                    })

                    response_with_quotes = session.post(filter_url, data=payload_for_quotes, timeout=15)
                    soup_with_quotes = BeautifulSoup(response_with_quotes.text, 'lxml')
                    
                    parsed_data = parser.parse_items(soup_with_quotes)
                    if parsed_data: save_items_to_db(conn, parsed_data, loggers)
                    
                    updated_form_data = _extract_form_data(soup_with_quotes, loggers)
                    if updated_form_data: form_data_for_author = updated_form_data

            except requests.RequestException as e:
                logger.error(f"Network error while processing author {author}: {e}")
            except Exception as e:
                logger.error(f"Unknown error while processing author {author}: {e}", exc_info=True)

    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)

    logger.info("Requests-based scraping finished.")


def scrape_with_playwright_pagination(conn: sqlite3.Connection, config: dict, parser_class_name: str, base_url: str, base_config: dict, loggers: Loggers):
    """
    Scrapes a site with classic pagination (e.g., a "Next" button) using Playwright.
    """
    logger = loggers.combined
    logger.info("Starting scraper: Pagination with Playwright.")
    
    parser = _get_parser_instance(parser_class_name, loggers, base_url=base_url)
    
    with BrowserManager(base_config, loggers) as context:
        page = context.new_page()
        current_url = config["start_url"]
        page_num = 1

        while current_url:
            if config.get("max_pages") and page_num > config["max_pages"]:
                logger.info(f"Reached page limit of {config['max_pages']}. Stopping.")
                break
            
            logger.info(f"Scraping page #{page_num}: {current_url}")
            try:
                page.goto(current_url, wait_until='domcontentloaded', timeout=30000)
                page.wait_for_selector(config["item_selector"], timeout=20000)
                
                soup = BeautifulSoup(page.content(), 'lxml')
                
                items = parser.parse_items(soup)
                if items:
                    save_items_to_db(conn, items, loggers)
                else:
                    logger.warning(f"No items found on page #{page_num}. This could be the end of the site.")
                
                # Find the next page URL using the parser's logic
                current_url = parser.get_next_page_url(soup, current_url)
                page_num += 1

            except Exception as e:
                logger.error(f"Error processing page {current_url}: {e}")
                break
    
    logger.info("Pagination scraping finished.")


def scrape_with_playwright_scroll(conn: sqlite3.Connection, config: dict, parser_class_name: str, base_url: str, base_config: dict, loggers: Loggers):
    """
    Scrapes a site with infinite scroll or a "Load More" button.
    """
    logger = loggers.combined
    logger.info("Starting scraper: Infinite Scroll / Load More.")

    parser = _get_parser_instance(parser_class_name, loggers, base_url=base_url)

    with BrowserManager(base_config, loggers) as context:
        page = context.new_page()
        page.goto(config["start_url"], wait_until='networkidle')

        item_selector = config["item_selector"]
        page.wait_for_selector(item_selector, timeout=20000)
        
        seen_items_count = 0
        while True:
            # 1. Parse items currently visible on the page
            all_item_handles = page.locator(item_selector).all()
            
            if len(all_item_handles) == seen_items_count:
                logger.info("No new items found after the last action. Ending scrape.")
                break

            new_item_handles = all_item_handles[seen_items_count:]

            for element_handle in new_item_handles:
                # Use inner_html() for a single element, which is faster than page.content()
                html_fragment = element_handle.inner_html()
                
                # We wrap the fragment in a proper structure for BeautifulSoup
                # to parse it correctly as a standalone document.
                soup = BeautifulSoup(f"<html><body>{html_fragment}</body></html>", 'lxml')
                
                # The parser's parse_items should be robust enough to find items in this fragment
                items = parser.parse_items(soup)
                if items:
                    save_items_to_db(conn, items, loggers)

            seen_items_count = len(all_item_handles)
            logger.info(f"Total items processed: {seen_items_count}")

            # 2. Perform an action to load more data
            action_performed = False
            try:
                # Priority is given to a "load more" button
                if load_more_selector := config.get("load_more_selector"):
                    button = page.locator(load_more_selector)
                    if button.is_visible():
                        button.click()
                        action_performed = True
                # If no button, perform a scroll action
                elif config.get("scroll", False):
                    page.keyboard.press("End")
                    action_performed = True

                if not action_performed:
                    logger.info("No more actions to perform (button or scroll). Ending scrape.")
                    break

                # 3. Wait for new content to appear
                # This function waits until the number of items on the page is greater than the last count.
                page.wait_for_function(
                    f"document.querySelectorAll('{item_selector}').length > {seen_items_count}",
                    timeout=15000 # Wait up to 15 seconds for new items
                )
            except Exception:
                logger.info("Timeout while waiting for new content. Assuming end of page.")
                break
    
    logger.info("Incremental scraping finished.")
    
def scrape_with_requests_pagination(conn: sqlite3.Connection, config: dict, parser_class_name: str, base_url: str, loggers: Loggers):
    """
    Scrapes a site with classic pagination using the `requests` library.
    This is a quick strategy for static sites.
    """
    logger = loggers.combined
    logger.info("Starting scraper: Pagination with Requests.")

    parser = _get_parser_instance(parser_class_name, loggers, base_url=base_url)

    session = requests.Session()
    # Set up retry logic for network errors
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    current_url = config["start_url"]
    page_num = 1

    while current_url:
        if config.get("max_pages") and page_num > config["max_pages"]:
            logger.info(f"Reached page limit of {config['max_pages']}. Stopping.")
            break

        logger.info(f"Scraping page #{page_num}: {current_url}")
        try:
            response = session.get(current_url, timeout=20)
            response.raise_for_status() # Will throw an error for 4xx/5xx statuses

            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'lxml')

            items = parser.parse_items(soup)
            if items:
                save_items_to_db(conn, items, loggers)
            else:
                logger.warning(f"No items found on page #{page_num}. This could be the end of the site or a broken page.")

            # Find the URL of the next page
            current_url = parser.get_next_page_url(soup, current_url)
            page_num += 1

        except requests.RequestException as e:
            logger.error(f"Network error on page {current_url}: {e}. Stopping pagination.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred processing page {current_url}: {e}", exc_info=True)
            break

    logger.info("Requests-based pagination scraping finished.")