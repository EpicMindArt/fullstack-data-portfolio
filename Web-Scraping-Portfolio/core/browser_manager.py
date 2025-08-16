from __future__ import annotations
import contextlib
from typing import TYPE_CHECKING
from playwright.sync_api import sync_playwright, Playwright, Browser, BrowserContext

from utils_scraping.models import Loggers

import os

if TYPE_CHECKING:
    # This allows you to use type hints without cyclic imports
    from configs import base_config

@contextlib.contextmanager
def BrowserManager(base_config: base_config, loggers: Loggers):
    """
    Context manager to handle the lifecycle of a Playwright browser instance.
    Ensures that the browser is launched, configured, and properly closed.

    Usage:
    with BrowserManager(cfg, loggers) as context:
        page = context.new_page()
        ...
    """
    logger = loggers.combined
    playwright_instance = None
    browser = None
    
    def block_resources(route):
        """Blocks unnecessary resources to speed up page loads."""
        if route.request.resource_type in base_config.BROWSER_CONFIG.get("blocked_types", []):
            route.abort()
        else:
            route.continue_()

    try:
        playwright_instance = sync_playwright().start()
        
        launch_mode = base_config.BROWSER_CONFIG.get("launch_mode", "launch")
        
        if launch_mode == "connect":
            port = base_config.BROWSER_CONFIG.get("debug_port", 9222)
            logger.info(f"Connecting to an existing browser instance on port {port}...")
            browser = playwright_instance.chromium.connect_over_cdp(f"http://localhost:{port}")
            # When connecting, we use the existing context instead of creating a new one.
            context = browser.contexts[0]
            
        else: # launch_mode == "launch"
            logger.info(f"Launching a new browser instance (headless={base_config.BROWSER_CONFIG.get('headless', True)})...")
            browser = playwright_instance.chromium.launch(
                headless=base_config.BROWSER_CONFIG.get("headless", True),
                slow_mo=base_config.BROWSER_CONFIG.get("slow_mo", 0),
                args=['--disable-blink-features=AutomationControlled'] # Helps avoid bot detection
            )

            # Create a new context with the specified parameters
            context_options = {
                "user_agent": base_config.BROWSER_CONFIG.get("user_agent"),
                "viewport": base_config.BROWSER_CONFIG.get("viewport"),
                # ... other emulation options can be added here ...
            }
            # Load session state if a path is provided
            session_path = getattr(base_config, 'SESSION_FILE_PATH', None)
            if session_path and os.path.exists(session_path):
                logger.info(f"Found session file. Attempting to load state from '{session_path}'")
                context_options["storage_state"] = session_path
            elif session_path:
                logger.warning(f"Session file specified at '{session_path}', but it was not found. Starting a clean session.")

            context = browser.new_context(**context_options)

        # Apply resource blocking if enabled
        if base_config.BROWSER_CONFIG.get("block_resources", False):
            logger.info(f"Blocking resource types: {base_config.BROWSER_CONFIG.get('blocked_types')}")
            context.route("**/*", block_resources)

        yield context  # Yield the prepared CONTEXT, not the browser

    except Exception as e:
        logger.critical(f"Error during browser initialization: {e}", exc_info=True)
        # Re-raise the exception so the main script can handle it
        raise
    finally:
        if browser:
            # If we connected to an existing browser, we should not close it
            if launch_mode == "launch":
                logger.info("Closing browser...")
                browser.close()
        if playwright_instance:
            playwright_instance.stop()
        logger.info("Playwright resources have been released.")