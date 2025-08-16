from __future__ import annotations
from typing import Dict, Any, Optional

import streamlit as st

from .models import SiteConfig
from .utils import read_yaml_file
from .constants import I18N_DIR, DEFAULT_LANGUAGE


@st.cache_data(show_spinner=False)
def _load_translation_file(lang: str) -> Dict[str, Any]:
    """Loads a single translation YAML file."""
    path = I18N_DIR / f"{lang}.yaml"
    if not path.exists():
        return {}
    return read_yaml_file(path)

def t(key: str, lang: str, default: Optional[str] = None) -> str:
    """
    Translates a given key using the loaded i18n data.
    It follows a fallback chain: current lang -> default lang -> provided default -> key itself.
    """
    # 1. Try to get the translation for the current language
    data = _load_translation_file(lang)
    value = _get_nested_key(data, key)
    if value:
        return str(value)

    # 2. Fallback to the default language if different
    if lang != DEFAULT_LANGUAGE:
        default_data = _load_translation_file(DEFAULT_LANGUAGE)
        value = _get_nested_key(default_data, key)
        if value:
            return str(value)

    # 3. Fallback to the provided default string or the key itself
    return default or key

def _get_nested_key(data: Dict[str, Any], key: str) -> Optional[Any]:
    """Helper to access a nested dictionary key like 'ui.title'."""
    keys = key.split('.')
    current_level = data
    for k in keys:
        if isinstance(current_level, dict) and k in current_level:
            current_level = current_level[k]
        else:
            return None
    return current_level

def get_initial_language(site_config: SiteConfig) -> str:
    """Determines the initial language for the session."""
    return site_config.i18n.default or DEFAULT_LANGUAGE