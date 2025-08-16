import csv
from pathlib import Path
from .models import Loggers # Use relative import within the same package

def save_to_csv(data: list[dict], filepath: Path, loggers: Loggers):
    """
    Saves a list of dictionaries to a CSV file.

    :param data: A list of dictionaries to save.
    :param filepath: The full path (pathlib.Path object) to the output CSV file.
    :param loggers: The logger instances.
    """
    if not data:
        loggers.combined.warning("No data provided to save to CSV.")
        return
        
    try:        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            # Dynamically get fieldnames from the first item in the list
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(data)
            
        loggers.combined.info(f"Data successfully saved to CSV file: {filepath}")
    except (IOError, IndexError) as e:
        loggers.combined.error(f"Error saving data to {filepath}: {e}")