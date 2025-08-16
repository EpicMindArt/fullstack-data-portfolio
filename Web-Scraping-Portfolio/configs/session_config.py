"""
Configuration for the session file generation utility (generate_session.py).
"""

# The port on which Chrome should be running in debug mode.
# Ensure this port is free and matches the one in the launch command.
# Example command for Windows:
# "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\ChromeDebugProfile"
DEBUGGING_PORT = 9222

# The file path where the session state (cookies, local storage, etc.) will be saved.
# This file will then be used by the main scraper to load an authenticated state.
SESSION_FILE_PATH = "auth_state.json"