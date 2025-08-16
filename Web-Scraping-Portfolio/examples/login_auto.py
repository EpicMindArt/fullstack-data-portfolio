"""
Example script for demonstrating automated login and session saving.
This script showcases:
1. Using the BrowserManager to create a configured browser instance.
2. The process of filling out a login form.
3. Saving the context state (cookies, local storage) to a file for later use.
"""

# --- Environment initialization for correct console output ---
import sys
try:    
    from colorama import init, just_fix_windows_console
    init()
    just_fix_windows_console()
    sys.stdout.reconfigure(encoding='utf-8')
except ImportError:
    print("Warning: colorama library not found. Colored output will be disabled.")
except Exception as e:
    print(f"Warning: Failed to configure the console: {e}")
# --- End of initialization block ---

import time
from playwright.sync_api import Page
from types import SimpleNamespace

# Import our custom framework modules
from core.browser_manager import BrowserManager
from configs.base_config import BROWSER_CONFIG, SESSION_FILE_PATH
from utils_scraping.logger_config import initialize_loggers

# --- Example-specific Settings ---
LOGIN_URL = "http://quotes.toscrape.com/login"
USERNAME = "admin" # Replace with real credentials if needed
PASSWORD = "admin" # Replace with real credentials if needed

# Selectors for the login page elements
USERNAME_SELECTOR = "#username"
PASSWORD_SELECTOR = "#password"
SUBMIT_BUTTON_SELECTOR = 'input[type="submit"]'

# An element that ONLY appears AFTER a successful login.
# This is the most reliable way to confirm a successful login.
LOGIN_SUCCESS_SELECTOR = 'a[href="/logout"]'


def perform_login(page: Page, logger) -> bool:
    """
    Executes the steps to fill out the form and log into the site.
    Returns True on success, False on failure.
    """
    logger.info(f"Navigating to login page: {LOGIN_URL}")
    try:
        page.goto(LOGIN_URL, wait_until='networkidle')

        logger.info(f"Filling username field: {USERNAME_SELECTOR}")
        page.locator(USERNAME_SELECTOR).fill(USERNAME)

        logger.info(f"Filling password field: {PASSWORD_SELECTOR}")
        page.locator(PASSWORD_SELECTOR).fill(PASSWORD)

        logger.info(f"Clicking submit button: {SUBMIT_BUTTON_SELECTOR}")
        page.locator(SUBMIT_BUTTON_SELECTOR).click()

        # Crucial step: wait for confirmation of a successful login.
        logger.info(f"Waiting for success element to appear: '{LOGIN_SUCCESS_SELECTOR}'...")
        page.wait_for_selector(LOGIN_SUCCESS_SELECTOR, timeout=10000)

        logger.info("[SUCCESS] Login successful!")
        return True

    except Exception as e:
        logger.error(f"An error occurred during the login process: {e}")
        screenshot_path = "login_error.png"
        page.screenshot(path=screenshot_path)
        logger.error(f"Error screenshot saved to: {screenshot_path}")
        return False


def main():
    """Main logic for the example script."""
    loggers = initialize_loggers()
    logger = loggers.combined

    # We can modify the base config specifically for this example.
    # For instance, we definitely want to see what's happening, so headless=False.
    example_browser_config = BROWSER_CONFIG.copy()
    example_browser_config["headless"] = False
    example_browser_config["slow_mo"] = 200 # Increase delay for better visibility
    
    # Create a 'mock' config object to pass to the BrowserManager.
    # This avoids creating a full config file for a simple example.
    base_cfg = SimpleNamespace(
        BROWSER_CONFIG=example_browser_config,
        SESSION_FILE_PATH=SESSION_FILE_PATH
    )

    logger.info("--- Running Automated Login Example ---")

    try:
        # Use our BrowserManager! It handles the setup, configuration,
        # and teardown of the browser for us.
        with BrowserManager(base_cfg, loggers) as context:
            page = context.new_page()

            if perform_login(page, logger):
                # If login is successful, save the session state
                logger.info(f"Saving session state to file: {SESSION_FILE_PATH}")
                context.storage_state(path=SESSION_FILE_PATH)
                logger.info("State saved successfully. This file can now be used by the main scraper.")
            else:
                logger.error("Login process failed. Session file will not be saved.")
            
            # A short pause to see the result before the browser closes
            logger.info("Script will terminate in 5 seconds...")
            time.sleep(5)

    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)

    logger.info("--- Example Finished ---")


if __name__ == "__main__":
    main()