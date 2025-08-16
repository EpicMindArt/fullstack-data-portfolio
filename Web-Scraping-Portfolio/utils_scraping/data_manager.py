from .models import Loggers

def remove_exact_duplicates(data: list[dict], unique_field: str, loggers: Loggers) -> list[dict]:
    """
    Removes duplicate dictionaries from a list based on a unique key, preserving order.
    
    NOTE: This is a post-processing utility. The recommended primary method for
    deduplication is to use a `UNIQUE` constraint in the database schema, which
    prevents duplicates from being inserted in the first place. This function

    can serve as a final cleanup step if needed.
    """
    loggers.combined.info("Performing final deduplication check...")
    
    if not data:
        return []
        
    seen = set()
    unique_data = []
    
    for item in data:
        identifier = item.get(unique_field)
        if identifier is None:
            # Handle cases where the unique field might be missing
            unique_data.append(item)
            continue
            
        if identifier not in seen:
            seen.add(identifier)
            unique_data.append(item)
            
    original_count = len(data)
    final_count = len(unique_data)
    
    if original_count != final_count:
        loggers.combined.info(f"Deduplication complete. Original items: {original_count}, final unique items: {final_count}.")
    else:
        loggers.combined.info("No duplicates found.")
        
    return unique_data