## Architecture for Real-World Challenges

This project is not just a script, but a complete data extraction platform built on principles of extensibility and resilience. It allows for rapid adaptation to any target website, minimizing development time for new scrapers.

### Key Features

*   **Works with Any Website:** Supports both static sites (via `requests`) and complex, dynamic sites that heavily use JavaScript (via `Playwright`).
*   **Anti-Bot Circumvention:** The framework is ready for User-Agent and proxy rotation, as well as configurable delays to mimic human behavior.
*   **Authentication and Sessions:** Includes a utility to save a browser session after manual login, enabling scraping of private areas and user-specific data.
*   **Fault Tolerance:**
    *   **Incremental Saving:** Data is immediately saved to an SQLite database, preventing data loss in case of a critical failure.
    *   **Deduplication:** A `UNIQUE` constraint at the database level guarantees no duplicate records.
    *   **Automatic Retries:** Built-in retry mechanism for network requests to handle temporary issues.
*   **Flexibility:** New scrapers are added by creating just two files (a config and a parser class) without modifying the core framework.
*   **Reproducibility:** Fully containerized with Docker. The scraper can be run with a single command on any machine.

### Disclaimer
All web scraping and data processing is carried out exclusively in accordance with applicable law, the terms of use (ToS) of a particular site and official API interfaces.

- Social networks and services are collected **only through official APIs**, if provided.
- Confidential information is processed **only within the framework of the customer's clients**.
- Personal data of **third parties outside the customer's business** is not collected.
- Copyrights, access restrictions and technical protection measures are not violated.
- The methods used are transparent, safe and do not harm third-party services.