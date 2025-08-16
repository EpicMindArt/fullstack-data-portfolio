from typing import Literal

# --- 1. General Settings ---
SCRAPER_TYPE: Literal['playwright'] = 'playwright'
DB_NAME = "quotes_scroll.db"
CSV_NAME = "quotes_scroll.csv"
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
PARSER_CLASS = "QuotesScrollParser"

# --- 4. Scraping STRATEGY settings ---
SCROLL_CONFIG = {
    "start_url": "http://quotes.toscrape.com/scroll",
    "item_selector": "div.quote",
    "scroll": True, # We explicitly indicate that we need to scroll
    # "load_more_selector" is not needed, since there is no "Load More" button
}