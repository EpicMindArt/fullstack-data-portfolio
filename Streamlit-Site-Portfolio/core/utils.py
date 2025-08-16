from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import yaml
import logging

import streamlit as st
from PIL import Image
from slugify import slugify as python_slugify # Using a robust library

# Setup a logger for this module
logger = logging.getLogger(__name__)

# --- Caching Primitives ---

@st.cache_data(show_spinner=False)
def read_text_file(path: str | Path) -> str:
    """Cached function to read a text file."""
    p = Path(path)
    return p.read_text(encoding="utf-8")

@st.cache_data(show_spinner=False)
def read_yaml_file(path: str | Path) -> Dict[str, Any]:
    """Cached function to read a YAML file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

@st.cache_resource(show_spinner=False)
def load_image(path: str | Path) -> Image.Image:
    """Cached function to load an image object."""
    p = Path(path)
    return Image.open(str(p))

def clear_all_caches() -> None:
    """Clears all Streamlit caches."""
    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("All Streamlit caches have been cleared.")


# --- Security & Path Utilities ---

def secure_path_resolve(base_dir: Path, user_path: str) -> Path:
    """
    Safely resolves a path, ensuring it doesn't escape the base directory.
    Prevents Path Traversal attacks.
    """
    abs_base = base_dir.resolve()
    abs_res = (abs_base / user_path).resolve()
    
    if not str(abs_res).startswith(str(abs_base)):
        logger.warning(
            "Path Traversal attempt detected: base='%s', path='%s'",
            base_dir, user_path
        )
        raise ValueError("Path Traversal attempt detected")
        
    if not abs_res.exists():
        raise FileNotFoundError(f"Asset not found at resolved path: {abs_res}")

    return abs_res


# --- String Utilities ---

def slugify(text: str) -> str:
    """
    Generates a URL-friendly slug from a string, with Cyrillic support.
    """
    return python_slugify(text)