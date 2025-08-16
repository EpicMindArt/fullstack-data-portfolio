import sqlite3
from utils_scraping.models import Loggers
from pathlib import Path

# Import scraping methods to be dispatched
from .scraping_methods import (   
    scrape_with_requests_pagination,
    scrape_with_playwright_pagination,
    scrape_with_playwright_scroll,
    scrape_viewstate_with_requests
)
from .parsers.examples_parsers_test.gallery_scraper import scrape_gallery_site


def run_scraper(config, base_config, db_connection: sqlite3.Connection, loggers: Loggers, output_path: Path):
    """
    Selects and launches the appropriate scraping strategy based on the configuration.
    This acts as the central dispatcher for the entire scraping process.
    """
    logger = loggers.combined
    scraper_type = getattr(config, 'SCRAPER_TYPE', 'requests').lower()

    # --- Strategy for Gallery Scraping ---
    # Identified by the presence of the unique `TARGET_GALLERY_URLS` attribute.
    if hasattr(config, 'TARGET_GALLERY_URLS'):
        logger.info("Strategy selected: Gallery Scraping.")
        
        gallery_config = {
            "output_folder": config.OUTPUT_FOLDER,
            "create_subfolders": config.CREATE_SUBFOLDERS,
            "naming": config.NAMING_CONFIG,
            "delay_ms": config.DOWNLOAD_DELAY_MS
        }
        
        for url in config.TARGET_GALLERY_URLS:
            scrape_gallery_site(
                gallery_url=url,
                config=gallery_config,
                conn=db_connection,
                output_path=output_path,
                loggers=loggers                
            )
        return # End execution after gallery scraping
    
    # --- Strategy for Pagination-based Scraping via Requests ---
    # Identified by type 'requests' and the presence of `REQUESTS_PAGINATION_CONFIG`.
    if scraper_type == 'requests' and hasattr(config, 'REQUESTS_PAGINATION_CONFIG'):
        logger.info("Strategy selected: Pagination with Requests.")
        scrape_with_requests_pagination(
            conn=db_connection,
            config=config.REQUESTS_PAGINATION_CONFIG,
            parser_class_name=config.PARSER_CLASS,
            base_url=config.BASE_URL,
            loggers=loggers
        )
        return
    
        # --- Strategy for Playwright-based Scraping (Pagination) ---
    # Identified by type 'playwright' and the presence of `PAGINATION_CONFIG`.
    if scraper_type == 'playwright' and hasattr(config, 'PAGINATION_CONFIG'):
        logger.info("Strategy selected: Pagination with Playwright.")
        scrape_with_playwright_pagination(
             conn=db_connection,
             config=config.PAGINATION_CONFIG, # Pass the specific config block
             parser_class_name=config.PARSER_CLASS,
             base_url=config.BASE_URL,
             base_config=base_config,
             loggers=loggers
        )
        return
        
    # --- Strategy for Playwright-based Scraping (Infinite Scroll) ---
    if scraper_type == 'playwright' and hasattr(config, 'SCROLL_CONFIG'):
        logger.info("Strategy selected: Infinite Scroll with Playwright.")
        scrape_with_playwright_scroll(
           conn=db_connection,
            config=config.SCROLL_CONFIG, # Pass the specific config block
            parser_class_name=config.PARSER_CLASS,
            base_url=config.BASE_URL,
            base_config=base_config,
            loggers=loggers
        )
        return



    # --- Strategy for ViewState-based Scraping via Requests ---
    # Identified by type 'requests' and the presence of `VIEWSTATE_CONFIG`.
    if scraper_type == 'requests' and hasattr(config, 'VIEWSTATE_CONFIG'):
        logger.info("Strategy selected: ViewState with Requests.")
        scrape_viewstate_with_requests(
        conn=db_connection,
        start_url=config.BASE_URL,
        loggers=loggers,
        config=config.VIEWSTATE_CONFIG,
        parser_class_name=config.PARSER_CLASS 
        )
        return


    # If no strategy was matched
    logger.error(f"Could not determine a suitable scraping strategy for configuration '{config.__name__}'.")
    logger.error("Check that the config specifies a correct SCRAPER_TYPE and contains the required blocks (e.g., VIEWSTATE_CONFIG, PAGINATION_CONFIG, etc.).")