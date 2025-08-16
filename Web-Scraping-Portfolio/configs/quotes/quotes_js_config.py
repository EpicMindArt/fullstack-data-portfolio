from typing import Literal

# --- 1. General Settings ---
# Requires JS execution, so type 'playwright'
SCRAPER_TYPE: Literal['playwright'] = 'playwright'
DB_NAME = "quotes_js.db"
CSV_NAME = "quotes_js.csv"
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
    "start_url": "http://quotes.toscrape.com/js/",
    "item_selector": "div.quote",
    "next_page_selector": "li.next > a",
    "max_pages": None,
}