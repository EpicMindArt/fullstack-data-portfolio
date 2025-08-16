# Full-Stack Data Solutions Portfolio

This monorepository contains a portfolio of three interconnected, production-ready projects that demonstrate a complete, end-to-end data lifecycle: from resilient data extraction to in-depth analysis and interactive presentation.

Each project is a standalone, fully containerized application designed with principles of clean architecture, scalability, and reproducibility.

### The Data Lifecycle Demonstrated Here:

```
[ ⚙️ Web Scraping Framework ]  ->  [ 🔬 Data Analysis Pipeline ]  ->  [ 🖥️ Streamlit Portfolio UI ]
 (Extracts raw data)             (Cleans, analyzes, finds insights)   (Presents results & skills)
```
*(This diagram illustrates how the components can work together as a full-stack data intelligence system)*

---

## 1. [Web Scraping Framework](./Web-Scraping-Portfolio/)

A production-grade framework for extracting data from any website. Built on the Strategy design pattern, it allows for the rapid development of new scrapers without modifying the core system.

*   **Universal Compatibility:** Handles static sites (`requests`) and dynamic, JavaScript-heavy sites (`Playwright`).
*   **Resilience by Design:** Features automatic network retries, incremental saving to `SQLite` to prevent data loss, and database-level deduplication.
*   **Advanced Capabilities:** Includes utilities for managing authenticated sessions to scrape sites requiring a login and is architected to bypass common anti-bot measures.

**Tech Stack:** Python, Playwright, Requests, BeautifulSoup4, SQLite, Docker.

---

## 2. [Data Analysis Pipeline](./Data-Analytics-Portfolio/)

A configurable CLI framework for executing reproducible data analysis. It can perform a rapid, automated Exploratory Data Analysis (EDA) or a deep-dive analysis to answer specific, complex business questions.

*   **Reproducibility:** All pipelines are controlled via a central `config.yaml`, ensuring that any analysis can be rerun with identical results.
*   **Business-Oriented Case Studies:** Includes a deep-dive analysis of HR data using statistical models (`statsmodels`) to assess pay equity, salary compression, and calculate cost-to-parity — demonstrating the ability to solve tangible business problems.
*   **Rich, Automated Outputs:** Generates a suite of artifacts including CSV/Excel reports, static and interactive visualizations (`Matplotlib`, `Seaborn`, `Plotly`), and comprehensive HTML profiles (`ydata-profiling`).

**Tech Stack:** Python, Pandas, Statsmodels, Plotly, ydata-profiling, Docker.

---

## 3. [Streamlit Site Portfolio Engine](./Streamlit-Site-Portfolio/)

A lightweight, extensible, and multilingual portfolio website engine built with Streamlit. It serves as the presentation layer for this portfolio, dynamically rendering content from a modular "skills" system.

*   **Modular & Multilingual:** Each portfolio piece is a self-contained "skill" with its own metadata and content, fully supporting internationalization (i18n) via YAML files.
*   **Dynamic Content Renderers:** Supports both static Markdown and interactive Python-based renderers, allowing for live demos, charts, and forms directly within the portfolio.
*   **Secure & Configurable:** Employs Pydantic for robust configuration validation and secure asset loading to prevent vulnerabilities like Path Traversal.

**Tech Stack:** Streamlit, Pydantic, YAML, Docker.

---

## Core Architectural Principles

This portfolio adheres to modern software development best practices across all projects:

*   **Containerization First:** Every application is fully dockerized with `Docker` and `Docker Compose` for seamless setup and deployment.
*   **Configuration as Code:** Settings are externalized from application logic (`.yaml`, `.py` configs), making the systems flexible and easy to maintain.
*   **Separation of Concerns:** Each project follows a clean, modular architecture that separates data, business logic, and presentation layers.
*   **Developer Experience:** Includes tools for automation (scaffolding scripts) and enhanced CLI interfaces (`rich`, `loguru`) to improve the development workflow.

## How to Run

Each of the three directories (`Web-Scraping-Portfolio/`, `Data-Analytics-Portfolio/`, `Streamlit-Site-Portfolio/`) is a standalone application. Please refer to the `README.md` file inside each respective directory for detailed setup and execution instructions.

## About Me

My name is Oleksii Y. I build data-driven systems. My methodology centers on an **AI-accelerated workflow**, allowing me to architect and deliver robust, production-quality solutions with exceptional speed.

---

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/license/mit) file for details.