from typing import Literal

# --- 1. General Settings ---
SCRAPER_TYPE: Literal['requests'] = 'requests'
DB_NAME = "quotes_table.db"
CSV_NAME = "quotes_table.csv"
BASE_URL = "http://quotes.toscrape.com/"

# --- 2. Data structure ---
# The best unique identifier will be the URL or SKU.
DATA_STRUCTURE = {
    "quote": "TEXT",
    "author": "TEXT",
    "tags": "TEXT",
    "unique_id": "TEXT UNIQUE"
}

# --- 3. Parser settings ---
PARSER_CLASS = "QuotesTableParser"

# --- 4. Scraping STRATEGY settings ---
REQUESTS_PAGINATION_CONFIG = {
    "start_url": "http://quotes.toscrape.com/tableful/",
    # Here item_selector is more complex. We need to select all lines containing a quote.
    # These are lines that do not have the style="border-bottom: 0px; " attribute
    # but do have a td with text. It would be more reliable to select all <tr> and filter in the parser.
    "item_selector": "table tr",
    "next_page_selector": "a[href*='/page/']",
    "max_pages": None,
}