# Dynamic Streamlit Portfolio

A lightweight, extensible, and multilingual portfolio website engine built with Streamlit. It supports navigation by skills/projects, search, tagging, content localization, secure asset handling, and Python-based renderers for interactive demos.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Adding New Content (Skills)](#adding-new-content-skills)
- [Content Types](#content-types)
- [Deployment](#deployment)
- [Customization](#customization)
- [Roadmap](#roadmap)
- [License](#license)

## Features

-   **Multilingual Support**: Built-in internationalization using YAML files with language fallbacks. Includes a UI language switcher.
-   **Modular Content System ("Skills")**: Each portfolio piece is a self-contained "skill" with its own metadata (`meta.yaml`), content, and assets.
-   **Dynamic Content Renderers**: Supports both static Markdown (`.md`) and interactive Python (`.py`) content, allowing for live demos, charts, and forms.
-   **Search and Filtering**: Full-text search across skill names, summaries, and tags. Filter skills by category and tags.
-   **Developer-Friendly Scaffolding**: A command-line script (`scripts/new_skill.py`) to quickly create the necessary file structure for a new skill.
-   **Secure & Efficient**: Uses caching for I/O operations and ensures secure asset loading to prevent Path Traversal vulnerabilities.
-   **Centralized Configuration**: All site settings are managed in a single `config/site.yaml` file, with support for environment variable overrides via Pydantic.
-   **Deployment-Ready**: Includes instructions for deploying to Streamlit Community Cloud and via Docker.

## Tech Stack

-   **Framework**: [Streamlit](https://streamlit.io/)
-   **Data Validation**: [Pydantic](https://pydantic.dev/)
-   **Configuration**: PyYAML, pydantic-settings
-   **Dependencies**: Pillow, python-slugify

## Project Structure

The project is organized to separate configuration, core logic, content, and presentation.

```
Streamlit-Site-Portfolio/
├── app.py                   # Main application entrypoint
├── requirements.txt         # Project dependencies
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration
├── assets/
│   └── style.css            # Global CSS styles
├── config/
│   └── site.yaml            # Main site configuration
├── core/                    # Core application logic and models
├── i18n/                    # Language translation files (e.g., en.yaml)
├── pages/                   # Main page content (e.g., summary.en.md)
├── scripts/
│   └── new_skill.py         # Scaffolding script for new skills
├── skills/
│   └── your-skill-id/       # Each skill is a folder
│       ├── meta.yaml        # Skill metadata (name, tags, etc.)
│       ├── content.en.md    # Skill content (Markdown)
│       ├── content.py       # OR Skill content (Python renderer)
│       └── assets/          # Skill-specific images, videos, etc.
└── view/                    # UI components and presentation logic
```

## Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Streamlit-Site-Portfolio.git
    cd Streamlit-Site-Portfolio
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows:
    # .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure your site:**
    -   Edit `config/site.yaml` to set your name, social links, and other details.
    -   Add your own content in the `skills/` directory.

5.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## Configuration

All site-wide settings are in `config/site.yaml`.

```yaml
# config/site.yaml
email: "john@example.com"

branding:
  page_icon: "💼"

social:
  github: "https://github.com/johndoe"
  linkedin: "https://www.linkedin.com/in/johndoe/"
  telegram: "https://t.me/johndoe"

features:
  use_sidebar_nav: true
  developer_mode: false # Set to true to show a "Clear Cache" button

i18n:
  default: "en"
  languages:
    - code: "en"
      label: "English"
      flag: "us"
    - code: "ru"
      label: "Русский"
      flag: "ru"
    - code: "ua"
      label: "Українська"
      flag: "ua"

search:
  tag_aliases:
    nlp: "natural-language-processing" # Alias 'nlp' tag to a canonical form

resources:
  resume_pdf: "assets/resume.pdf" # Path to enable the "Download Resume" button
```

## Adding New Content (Skills)

#### Method 1: Use the CLI Script (Recommended)

The easiest way to add a new skill is with the `new_skill.py` script.

```bash
# Create a Markdown-based skill
python scripts/new_skill.py "My New Project" --category "data-analytics" --summary "A brief description of this project."

# Create a skill with a Python-based interactive renderer
python scripts/new_skill.py "Interactive Demo" --category "demos" --python
```

This will create a new folder `skills/my-new-project/` with a `meta.yaml` file, a content file, and an `assets` folder.

#### Method 2: Manually

1.  Create a new folder inside `skills/`. The folder name will be the skill's ID (e.g., `web-scraping`).
2.  Inside, create a `meta.yaml` file.
3.  Add a content file: `content.md` (or `content.en.md`) for Markdown, or `content.py` for a Python renderer.
4.  Place any images, videos, or other files in the `skills/web-scraping/assets/` subfolder.
5.  Update `meta.yaml` to link to your assets.

---

**Example `meta.yaml`:**

This file defines all the metadata for a skill. For multilingual fields like `name` and `summary`, you can provide different strings for each language code.

```yaml
# skills/telegram-bots/meta.yaml
name:
  en: "Telegram Bots"
  ru: "Телеграм-боты"
category: "bots" # Must match a key in i18n/*.yaml for translation
summary:
  en: "FSM, inline mode, payments, and deployment strategies."
  ru: "FSM, инлайн-режим, платежи и стратегии деплоя."
tags: ["telegram", "bots", "asyncio"]
tech_stack: ["aiogram", "aiohttp", "redis"]
order: 10 # Lower numbers appear first

# Paths are relative to this skill's folder
hero_image: "assets/hero.png"
gallery:
  - "assets/screen1.png"
  - "assets/screen2.png"
video: "assets/demo.webm" # Can also be a YouTube/Vimeo URL

# External links
demo_url: "https://my-bot-demo.example.com"
repo_url: "https://github.com/user/my-bot-project"
```

## Content Types

#### Markdown (`content.<lang>.md`)

-   Standard Markdown is fully supported.
-   You can include HTML, as it's rendered with `unsafe_allow_html=True`.
-   Create separate files for each language (e.g., `content.en.md`, `content.ru.md`). The system will automatically serve the correct one based on the selected language, with a fallback to the default language.

#### Python Renderer (`content.py`)

-   For interactive content, create a `content.py` file.
-   This file must contain a function `render(st, skill, ctx)`.
-   `st`: The Streamlit module object.
-   `skill`: The Pydantic `Skill` model instance for this skill.
-   `ctx`: A context dictionary containing `{'site_config': ..., 'lang': ...}`.

**Example `content.py`:**
```python
import streamlit as st
import pandas as pd

def render(st, skill, ctx):
    st.markdown("### Interactive Data Demo")
    
    # You can access skill metadata
    st.write(f"Data from repository: {skill.repo_url}")

    # Example of an interactive chart
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': [10, 20, 30, 40]
    })
    st.line_chart(df)
```

## Deployment

#### Option A: Streamlit Community Cloud

1.  Push your repository to GitHub.
2.  Go to [share.streamlit.io](https://share.streamlit.io/) and create a new app.
3.  Connect your GitHub repository.
4.  Ensure the main file is `app.py`.
5.  Deploy.

#### Option B: Docker Run (Simple Method)

This method is quick for launching the container directly.

1.  **Build the Docker image:**
    First, ensure you have a `Dockerfile` in your project root. Then, build the image from your terminal:
    ```bash
    docker build -t streamlit-portfolio .
    ```

2.  **Run the container in the background:**
    The `-d` flag runs the container in "detached" mode, so you can close your console.
    ```bash
    docker run -d -p 8501:8501 streamlit-portfolio
    ```
    -   To view logs: `docker logs <container_id>`
    -   To stop the container: `docker stop <container_id>`

The app will be available at `http://localhost:8501`.

---

#### Option C: Docker Compose (Recommended Method)

Using Docker Compose is the recommended approach as it makes managing your application's lifecycle (start, stop, restart) much easier by defining its configuration in a file.

1.  **Build the image (if you haven't already):**
    This command is the same as in the previous method.
    ```bash
    docker build -t streamlit-portfolio .
    ```

2.  **Start the application:**
    This command reads the `docker-compose.yml` file and starts your service in the background (`-d`).
    ```bash
    docker-compose up -d
    ```

3.  **Stop the application:**
    To stop the service and remove the container, run:
    ```bash
    docker-compose down
    ```

The app will be available at `http://localhost:8501`.

## Customization

#### Theming

Modify `.streamlit/config.toml` to change the colors, fonts, and base theme of the application.

```toml
# .streamlit/config.toml
[theme]
primaryColor="#6C63FF"
backgroundColor="#0E1117"
secondaryBackgroundColor="#1B1F2A"
textColor="#FAFAFA"
font="sans serif"
```

#### Global Styles

Edit `assets/style.css` to add custom CSS rules. This file is loaded into every page. For example, you can change the appearance of skill cards or chips.

```css
/* assets/style.css */
:root {
    --card-bg: #1b1f2a;
    --chip-bg: #2a3241;
}

.skill-card {
    background: var(--card-bg);
    /* ... more styles ... */
}
```

## Roadmap

-   [ ] Shareable filter URLs (to link to a specific category/tag search).
-   [ ] Add thumbnail images to skill cards on the summary page.
-   [ ] Add unit tests for core services (search, i18n, security).

## License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/license/mit) file for details.
