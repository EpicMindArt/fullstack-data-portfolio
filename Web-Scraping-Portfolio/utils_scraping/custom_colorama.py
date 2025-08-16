import os

# ANSI escape codes for colors
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
GREEN_BG = "\033[102m\033[30m" # Green background, black text
RESET = "\033[0m"

def print_yellow(text: str):
    """Prints text in yellow."""
    print(f"{YELLOW}{text}{RESET}")

def print_green(text: str):
    """Prints text in green."""
    print(f"{GREEN}{text}{RESET}")

def print_red(text: str):
    """Prints text in red."""
    print(f"{RED}{text}{RESET}")

def print_green_bg(text: str):
    """Prints text with a green background."""
    print(f"{GREEN_BG}{text}{RESET}")

def separator_str() -> str:
    """Returns a separator string that fits the terminal width."""
    try:
        columns = os.get_terminal_size().columns
    except OSError:
        # Fallback for environments where terminal size cannot be determined (e.g., in some IDEs)
        columns = 80
    return '-' * columns

def print_separator():
    """Prints a separator line in yellow."""
    print_yellow(separator_str())