from typing import Literal

# --- 1. General Settings ---
SCRAPER_TYPE: Literal['requests'] = 'requests'
DB_NAME = "quotes_viewstate.db"
CSV_NAME = "quotes_viewstate.csv"
BASE_URL = "http://quotes.toscrape.com/search.aspx"

# --- 2. Data structure ---
# The best unique identifier will be the URL or SKU.
DATA_STRUCTURE = {
    "author": "TEXT",
    "quote": "TEXT",
    "tags": "TEXT",
    "unique_id": "TEXT UNIQUE" 
}

# --- 3. Parser settings ---
PARSER_CLASS = "QuotesViewStateParser"

# --- 4. Scraping STRATEGY settings ---
VIEWSTATE_CONFIG = {
    "filter_url": "http://quotes.toscrape.com/filter.aspx",
    "primary_filter": {
        "selector": "#author",
        "option_selector": "#author > option",
        "skip_option_value": "----------",
        # You can change target_value to the author name for the test, or None for full scraping
        "target_value": None,  # e.g., "Albert Einstein" or None
    },
    "secondary_filter": {
        "selector": "#tag",
        "option_selector": "#tag > option",
        "skip_option_value": "----------",
    }
}