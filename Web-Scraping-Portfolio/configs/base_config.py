"""
Base application settings that rarely change.
"""
# Path for saving the session file. Must match the setting in session_config.py
SESSION_FILE_PATH = "auth_state.json"

# Settings for the Playwright browser instance
BROWSER_CONFIG = {
    "headless": False,
    "slow_mo": 50,  # Delay between actions for observation
    "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    "viewport": {'width': 1920, 'height': 1080},
    
    # Block specified resource types to speed up page loads
    "block_resources": True,
    "blocked_types": ["image", "font", "media"], # "stylesheet" can be added for more speed
    
    # "launch" a new browser or "connect" to an existing one
    "launch_mode": "launch", 
    "debug_port": 9222,      # Only used if launch_mode = "connect"
}