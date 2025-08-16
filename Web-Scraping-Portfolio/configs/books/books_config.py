from typing import Literal

# --- 1. General Settings ---
SCRAPER_TYPE: Literal['requests'] = 'requests'
DB_NAME = "books.db"
CSV_NAME = "books_results.csv"
BASE_URL = "http://books.toscrape.com/"

# --- 2. Data structure ---
# The best unique identifier will be the URL or SKU.
DATA_STRUCTURE = {
    "title": "TEXT",
    "price": "REAL",
    "book_url": "TEXT UNIQUE",
    "image_url": "TEXT",
    "rating": "INTEGER",
    "availability": "TEXT"
}

# --- 3. Parser settings ---
PARSER_CLASS = "BooksParser"

# --- 4. Scraping STRATEGY settings ---
# Configuration block for pagination via Requests
REQUESTS_PAGINATION_CONFIG = {
    "start_url": "http://books.toscrape.com/catalogue/page-1.html",
    "item_selector": "article.product_pod",
    "next_page_selector": "li.next > a", # Selector for <a href="..."> inside li.next
    "max_pages": None, # None - for all pages
}