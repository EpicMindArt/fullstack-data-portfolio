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

import json
from pathlib import Path
from playwright.sync_api import sync_playwright, Playwright, Browser
from typing import Optional

# Import configuration and loggers
from configs.session_config import DEBUGGING_PORT, SESSION_FILE_PATH
from utils_scraping.logger_config import initialize_loggers


def connect_to_browser(playwright: Playwright, port: int) -> Optional[Browser]:
    """
    Attempts to connect to a browser instance launched with a debugging port.
    Returns a Browser object on success, or None on failure.
    """
    try:
        browser = playwright.chromium.connect_over_cdp(f"http://localhost:{port}")
        return browser
    except Exception:
        return None


def save_storage_state(browser: Browser, output_path: str, logger) -> bool:
    """Saves the storage state of the first browser context to a file."""
    try:
        # The main profile is usually the first context
        context = browser.contexts[0]
        storage_state = context.storage_state()

        # Use pathlib for robust and clean file operations
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist
        output_file.write_text(json.dumps(storage_state, indent=4), encoding='utf-8')
        
        logger.info(f"Session successfully saved to: '{output_path}'")
        return True
    except IndexError:
        logger.error("No active contexts found in the browser. Have you opened at least one tab?")
        return False
    except Exception as e:
        logger.error(f"Failed to save session state: {e}")
        return False


def main():
    """Main logic for the session generation utility."""
    loggers = initialize_loggers()
    logger = loggers.combined

    logger.info("--- Interactive Session File Generator ---")
    logger.info("Please ensure Chrome is running with the remote debugging port enabled.")
    logger.info(f"Example command: chrome.exe --remote-debugging-port={DEBUGGING_PORT} --user-data-dir=\"C:\\ChromeDebugProfile\"")

    with sync_playwright() as playwright:
        browser = connect_to_browser(playwright, DEBUGGING_PORT)

        if not browser:
            logger.critical(f"Failed to connect to the browser on port {DEBUGGING_PORT}.")
            logger.critical("Please check that:")
            logger.critical("1. All other Chrome instances (except the debug one) are closed.")
            logger.critical(f"2. Chrome was launched with the flag: --remote-debugging-port={DEBUGGING_PORT}")
            return

        logger.info("[SUCCESS] Successfully connected to your browser instance!")
        
        # We don't own this browser instance, so we do not close it.
        # We are merely a 'guest'.

        try:
            input("\n>>> After you have logged into all necessary websites, return to this console and PRESS ENTER...")
        except KeyboardInterrupt:
            logger.warning("\nOperation aborted by user. Exiting.")
            return

        save_storage_state(browser, SESSION_FILE_PATH, logger)

    logger.info("Script has finished. You can now close the browser.")


if __name__ == "__main__":
    main()