import sqlite3
import re
from .models import Loggers

def _sanitize_name(name: str) -> str:
    """Sanitizes a column or table name to prevent SQL injection issues."""
    return re.sub(r'[^a-zA-Z0-9_]', '', name)

def init_db(loggers: Loggers, db_name: str, table_structure: dict) -> sqlite3.Connection | None:
    """
    Initializes the database. If the 'items' table doesn't exist, it creates it
    dynamically based on the provided table_structure dictionary.

    :param loggers: The logger instances.
    :param db_name: The name of the database file.
    :param table_structure: A dict where keys are column names and values are SQL data types (e.g., 'TEXT UNIQUE').
    :return: A database connection object or None if an error occurs.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        columns_definitions = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
        for key, definition in table_structure.items():
            clean_key = _sanitize_name(key)
            # The definition already contains the type and constraints (e.g., "TEXT UNIQUE")
            columns_definitions.append(f'"{clean_key}" {definition}') 
        
        # 'CREATE TABLE IF NOT EXISTS' is idempotent and won't cause an error if the table already exists.
        create_table_sql = f"CREATE TABLE IF NOT EXISTS items ({', '.join(columns_definitions)})"
        
        loggers.file.debug(f"Executing SQL for table creation: {create_table_sql}")
        cursor.execute(create_table_sql)
        conn.commit()
        return conn

    except sqlite3.Error as e:
        loggers.combined.critical(f"Database initialization error: {e}")
        return None

def save_items_to_db(conn: sqlite3.Connection, items_data: list[dict], loggers: Loggers):
    """
    Saves a list of items to the database. Works with any dictionary structure.
    Uses 'INSERT OR IGNORE' to automatically handle duplicates if a UNIQUE constraint is set.
    """
    if not items_data:
        return
        
    try:
        cursor = conn.cursor()
        
        # Use the keys from the first dictionary as the columns
        sample_keys = list(items_data[0].keys())
        clean_keys = [_sanitize_name(key) for key in sample_keys]
        
        columns_str = ", ".join(f'"{k}"' for k in clean_keys)
        placeholders_str = ", ".join(["?"] * len(clean_keys))
        
        # Prepare a list of tuples for executemany
        records_to_insert = [tuple(item.get(key) for key in sample_keys) for item in items_data]
        
        # 'INSERT OR IGNORE' will skip inserting rows that violate a UNIQUE constraint,
        # which is the desired behavior for deduplication.
        sql = f"INSERT OR IGNORE INTO items ({columns_str}) VALUES ({placeholders_str})"
        
        cursor.executemany(sql, records_to_insert)
        conn.commit()
        
        newly_inserted_count = cursor.rowcount
        if newly_inserted_count > 0:
            loggers.combined.info(f"Successfully saved {newly_inserted_count} new records to DB (out of {len(records_to_insert)} provided).")
        else:
            loggers.file.info(f"{len(records_to_insert)} records were provided, but all were duplicates already in the DB.")

    except (sqlite3.Error, IndexError, KeyError) as e:
        loggers.combined.error(f"Error saving data to the database: {e}")

def read_all_items_from_db(conn: sqlite3.Connection, loggers: Loggers) -> list[dict]:
    """
    Reads all records from the 'items' table and returns them as a list of dictionaries.
    """
    try:
        # sqlite3.Row provides both index-based and case-insensitive name-based access to columns.
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM items")
        rows = cursor.fetchall()
        
        # Convert sqlite3.Row objects to standard Python dictionaries
        data = [dict(row) for row in rows]
        loggers.combined.info(f"Successfully fetched {len(data)} records from DB for final export.")
        return data
        
    except sqlite3.Error as e:
        loggers.combined.error(f"Error reading data from the database: {e}")
        return []
    
def close_db_connection(conn: sqlite3.Connection, logger: Loggers):
    """Closes the database connection."""
    if conn:
        conn.close()
        logger.combined.info("Database connection closed.")