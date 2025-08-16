from pathlib import Path

# --- Project Paths ---
# Defines the absolute root path of the project.
ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "config"
PAGES_DIR = ROOT_DIR / "pages"
SKILLS_DIR = ROOT_DIR / "skills"
I18N_DIR = ROOT_DIR / "i18n"
ASSETS_DIR = ROOT_DIR / "assets"

# --- Special Identifiers ---
# Used throughout the app to refer to the main summary/landing page.
SUMMARY_SKILL_ID = "summary"
# Identifier for the "All Categories" filter option in the navigation.
ALL_CATEGORIES_ID = "__all__"

# --- Default Values ---
DEFAULT_LANGUAGE = "en"