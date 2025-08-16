# ⚙️ Production-Ready Web Scraping Framework

This is a professional, scalable, and resilient web scraping framework built in Python. It is designed based on the **Open/Closed Principle**, allowing for the rapid development of new scrapers without modifying the core system. This project demonstrates not just web scraping techniques, but also proficiency in software architecture, design patterns, and creating production-ready tools.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [How to Scrape a New Site](#how-to-scrape-a-new-site)
- [Advanced Usage: Handling Logins](#advanced-usage-handling-logins)
- [Deployment & Portability (Docker)](#deployment--portability-docker)
- [License](#license)

## Features

-   **Strategy Pattern**: The framework uses a configuration-driven approach to dynamically select the appropriate scraping strategy (e.g., **`requests`** for static sites, **`playwright`** for dynamic JS-heavy sites).
-   **Separation of Concerns**: A clear distinction between scraping logic (`core/scraping_methods.py`), data parsing (`core/parsers/`), and scraper configuration (`configs/`).
-   **Dynamic Parser Loading**: New parsers are automatically discovered and loaded by the framework, even from subdirectories, making the system highly extensible.
-   **Resilience and Error Handling**:
    -   **Network Retries**: Implemented `requests.Retry` for handling temporary network issues.
    -   **Defensive Parsing**: Parsers are designed to handle missing HTML elements without crashing.
    -   **Persistent Storage**: Data is immediately saved to an SQLite database, preventing data loss on critical errors.
    -   **UNIQUE Constraints**: The database schema prevents duplicate entries automatically at the database level.
-   **Efficiency**: Utilizes `requests` for static content for maximum speed and `playwright` for complex JavaScript-driven websites, with resource blocking (`image`, `font`) to speed up page loads.
-   **Session Management**: Includes a utility to generate and use authenticated sessions, allowing you to scrape sites that require a login.

## Tech Stack

-   **Scraping Engines**: Playwright, Requests
-   **HTML Parsing**: BeautifulSoup4, lxml
-   **Database**: SQLite
-   **CLI & Utilities**: Colorama, TQDM

## Project Structure

```
Web-Scraping-Portfolio/
├── output/                     # Directory for all generated files (CSVs, db, images)
├── main.py                     # Main application entrypoint
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Instructions to build the Docker image
├── docker-compose.yml          # Configuration for Docker Compose
├── generate_session.py         # Utility for saving login sessions
├── configs/                    # Configuration files for each target site
│   ├── books/
│   └── quotes/
├── core/
│   ├── scraping_methods.py     # Implementations of scraping strategies
│   ├── orchestrator.py         # Logic to select the correct strategy
│   └── parsers/                # Site-specific parser classes
│       ├── base_parser.py      # Abstract base class for all parsers
│       └── examples_parsers_test/
└── utils_scraping/             # Reusable utility modules (DB, CSV, Logger)
```

## Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Web-Scraping-Portfolio.git
    cd Web-Scraping-Portfolio
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Playwright browsers:**
    This is a one-time setup step required by Playwright.
    ```bash
    playwright install
    ```

5.  **Configure and run the scraper (see next section).**

## How to Use

This framework is designed for rapid development. To scrape a new website, follow these steps:

#### Step 1: Create a Configuration File

In the `configs/` directory, create a new file (e.g., `my_site_config.py`). This file acts as the control panel for your scraper.

```python
# configs/my_site_config.py
from typing import Literal

# 1. Choose scraper type: 'requests' (fast) or 'playwright' (for JS sites)
SCRAPER_TYPE: Literal['requests'] = 'requests'
DB_NAME = "my_site.db"
CSV_NAME = "my_site.csv"
BASE_URL = "https://example.com"

# 2. Define the data structure for the database table
DATA_STRUCTURE = {
    "product_name": "TEXT",
    "price": "REAL",
    "product_url": "TEXT UNIQUE", # UNIQUE constraint prevents duplicates
}

# 3. Specify the name of the parser class you will create
PARSER_CLASS = "MySiteParser"

# 4. Select and configure the scraping strategy (e.g., pagination)
REQUESTS_PAGINATION_CONFIG = {
    "start_url": "https://example.com/products?page=1",
    "item_selector": "div.product-card",
    "next_page_selector": "a.next-page",
    "max_pages": 50,
}
```

#### Step 2: Create a Parser

In the `core/parsers/` directory, create a new file (`my_site_parser.py`) with a class that inherits from `AbstractParser`.

```python
# core/parsers/my_site_parser.py
from bs4 import BeautifulSoup, Tag
from core.parsers.base_parser import AbstractParser

class MySiteParser(AbstractParser):
    def parse_items(self, soup: BeautifulSoup) -> list[dict]:
        items = soup.select("div.product-card")
        return [self._parse_single_item(item) for item in items if item]

    def _parse_single_item(self, item_tag: Tag) -> dict | None:
        try:
            name = item_tag.select_one('.name').text
            price = float(item_tag.select_one('.price').text.replace('$', ''))
            url = item_tag.select_one('a')['href']
            return {"product_name": name, "price": price, "product_url": url}
        except (AttributeError, TypeError):
            self.logger.warning("Failed to parse an item card.")
            return None

    def get_next_page_url(self, soup: BeautifulSoup, current_url: str) -> str | None:
        next_page = soup.select_one("a.next-page")
        return next_page['href'] if next_page else None
```

#### Step 3: Run the Scraper

Open `main.py`, update the `ACTIVE_CONFIG_MODULE` variable to point to your new configuration file, and execute the script.

```python
# main.py
ACTIVE_CONFIG_MODULE = "configs.my_site_config"
```
```bash
python main.py
```

## Advanced Usage: Handling Logins

The framework supports scraping sites that require authentication by using a saved browser session.
#### Step 0: Preparation
1. Launch regular Google Chrome with your relatively clean profile. Turn it off. You won't need it anymore
2. Copy User Data (the entire folder) from `C:\Users\USERNAME\AppData\Local\Google\Chrome\User Data` to `C:\`
3. Rename `User Data` to `ChromeDebugProfile`

#### Step 1: Launch Chrome in Debug Mode

Close all other Chrome instances and run the following command in your terminal. This opens a special browser instance that our script can connect to.

```bash
# On Windows (adjust path if needed)
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\ChromeDebugProfile"
```

#### Step 2: Generate the Session File

1.  With the debug browser open, run the `generate_session.py` script:
    ```bash
    python generate_session.py
    ```
2.  In the debug browser window, manually log into the target website.
3.  Once logged in, return to the terminal and press `Enter`.

This will save your session (cookies, local storage) to `auth_state.json`.

#### Step 3: Use the Session in Your Scraper

The framework will automatically detect and use the `auth_state.json` file when running a Playwright-based scraper, allowing it to operate as a logged-in user.

## Deployment & Portability (Docker)

#### How to use Docker:

1.  **Build the Docker image:**
    This command needs to be run only once, or whenever you change `requirements.txt`.
    ```bash
    docker-compose build
    ```

2.  **Run a scraper:**
    Use `docker-compose run` to execute the `main.py` script inside the container. The `--rm` flag automatically cleans up the container after it exits. Ensure your desired config is set in main.py first!
    ```bash    
    docker-compose run --rm scraper python main.py
    ```
    All generated files (`.db`, `.csv`, logs) will appear in your local project directory because the volume is synced.

## License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/license/mit) file for details.