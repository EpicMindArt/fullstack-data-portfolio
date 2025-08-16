from typing import Literal

# --- 1. General Settings ---
SCRAPER_TYPE: Literal['playwright'] = 'playwright'
DB_NAME = "quotes_js_delayed.db"
CSV_NAME = "quotes_js_delayed.csv"
BASE_URL = "http://quotes.toscrape.com/"

# --- 2. Data structure ---
# The best unique identifier will be the URL or SKU.
DATA_STRUCTURE = {
    "quote": "TEXT",
    "author": "TEXT",
    "author_about_url": "TEXT",
    "tags": "TEXT",
    "unique_id": "TEXT UNIQUE"
}

# --- 3. Parser settings ---
PARSER_CLASS = "QuotesDefaultParser"

# --- 4. Scraping STRATEGY settings ---
PAGINATION_CONFIG = {
    "start_url": "http://quotes.toscrape.com/js-delayed/",
    "item_selector": "div.quote",
    "next_page_selector": "li.next > a",
    # We can add an optional timeout here,
    # if we want to override the default (20000 ms in the code).
    # But in this case the default timeout is enough.
    # "wait_for_selector_timeout": 15000, # Example: wait 15 seconds
    "max_pages": None,
}