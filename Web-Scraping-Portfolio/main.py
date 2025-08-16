# Step 1: Import the required configuration.
# To run a different scraper, change the import path below.
# We use dynamic imports for maximum flexibility.

# ======================= MASTER CONTROL SWITCH =======================
# Specify the path to the configuration module to be used.
# Example: "configs.quotes_config" or "configs.gallery_config"
# Future use: "configs.works.my_project_config"
ACTIVE_CONFIG_MODULE = "configs.quotes.quotes_scroll_config"
# =======================================================================

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


import importlib
from pathlib import Path
from utils_scraping.logger_config import initialize_loggers
from utils_scraping.db_manager import init_db, read_all_items_from_db, close_db_connection
from utils_scraping.csv_manager import save_to_csv
from core import run_scraper

# Dynamically import the selected configuration module
try:
    cfg = importlib.import_module(ACTIVE_CONFIG_MODULE)
    base_cfg = importlib.import_module("configs.base_config")
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to load configuration module: {e}")
    exit(1)


def main():
    """
    Main entry point of the application.
    Orchestrates the scraping process based on the loaded configuration.
    """
    loggers = initialize_loggers()
    logger = loggers.combined
    
    logger.info("=" * 50)
    logger.info(f"Starting scraper with configuration: {ACTIVE_CONFIG_MODULE}")
    logger.info(f"Scraper type: {cfg.SCRAPER_TYPE}")
    logger.info(f"Database file: {cfg.DB_NAME}")
    logger.info("=" * 50)
    
    output_dir_name = Path(cfg.CSV_NAME).stem
    output_path = Path("output") / output_dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    db_full_path = output_path / cfg.DB_NAME
    csv_full_path = output_path / cfg.CSV_NAME
    
    logger.info(f"All output files will be saved to: {output_path}")

    # Initialize the database
    conn = init_db(loggers, db_full_path, table_structure=cfg.DATA_STRUCTURE)
    if not conn:
        logger.critical("Failed to initialize the database. Exiting.")
        return

    try:
        # --- Select and run the scraping strategy ---
        run_scraper(cfg, base_cfg, conn, loggers, output_path)
        
        # --- Post-processing and saving results ---
        logger.info("Data collection finished. Reading results from the database...")
        
        final_data = read_all_items_from_db(conn, loggers)
        if final_data:            
            logger.info(f"Found {len(final_data)} unique records in the database.")            
            save_to_csv(final_data, csv_full_path, loggers)
        else:
            logger.info("No data in the database to save to CSV.")

    except Exception as e:
        logger.critical(f"An unexpected error occurred in main: {e}", exc_info=True)
    finally:
        # It's crucial to always close the DB connection
        close_db_connection(conn, loggers)
        logger.info("=" * 50)
        logger.info("SCRIPT EXECUTION FINISHED")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()