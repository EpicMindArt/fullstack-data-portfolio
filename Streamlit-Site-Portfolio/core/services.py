from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging

import streamlit as st

from .models import SiteConfig, Skill
from .utils import read_yaml_file
from .constants import CONFIG_DIR, SKILLS_DIR, PAGES_DIR

logger = logging.getLogger(__name__)


# --- Configuration Loading ---

@st.cache_data(show_spinner=False)
def load_site_config() -> SiteConfig:
    """Loads the main site configuration from site.yaml."""
    config_path = CONFIG_DIR / "site.yaml"
    if not config_path.exists():
        logger.error("site.yaml not found, using default SiteConfig.")
        return SiteConfig()
    
    data = read_yaml_file(config_path)
    return SiteConfig.model_validate(data)


# --- Skill Loading & Processing ---

@st.cache_data(ttl=3600)
def _scan_raw_skills_data() -> List[Dict[str, Any]]:
    """
    Scans the SKILLS_DIR for skill metadata.
    This is the slow I/O part and is heavily cached.
    It returns a list of raw dictionaries, not validated models.
    """
    raw_skills = []
    if not SKILLS_DIR.exists():
        logger.warning("Skills directory not found: %s", SKILLS_DIR)
        return []

    for folder in sorted(SKILLS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        
        meta_file = folder / "meta.yaml"
        if not meta_file.exists():
            continue

        meta_dict = read_yaml_file(meta_file)
        meta_dict["id"] = folder.name
        meta_dict["folder"] = folder
        raw_skills.append(meta_dict)
        
    logger.info("Scanned %d raw skills from disk.", len(raw_skills))
    return raw_skills

def load_and_localize_skills(lang: str, default_lang: str) -> List[Skill]:
    """
    Takes raw skill data, localizes it, and validates it into Skill models.
    This part is fast and runs on every script rerun to handle language changes.
    """
    raw_skills_data = _scan_raw_skills_data()
    skills: List[Skill] = []
    
    for meta_dict in raw_skills_data:
        skill_id = meta_dict.get("id", "unknown")
        folder = meta_dict.get("folder")
        
        # 1. Detect content file based on language
        content_type, content_path = _find_content_file(folder, lang, default_lang)
        
        # 2. Localize translatable fields
        processed_meta = meta_dict.copy()
        processed_meta["name"] = _get_i18n_field(meta_dict, "name", lang, default_lang, fallback=skill_id)
        processed_meta["summary"] = _get_i18n_field(meta_dict, "summary", lang, default_lang)
        
        # 3. Add computed fields for validation
        processed_meta["content_type"] = content_type
        processed_meta["content_path"] = content_path
        
        try:
            skill = Skill.model_validate(processed_meta)
            skills.append(skill)
        except Exception as e:
            st.warning(f"Skipping skill '{skill_id}' due to validation error: {e}")
            logger.error("Pydantic validation failed for skill '%s'", skill_id, exc_info=True)

    skills.sort(key=lambda s: (s.order, s.name.lower()))
    return skills


# --- Helper functions for data loading ---

def _get_i18n_field(data: Dict, key: str, lang: str, default_lang: str, fallback: str = "") -> str:
    """Extracts a translated string from a potentially multilingual dictionary field."""
    value = data.get(key, fallback)
    if isinstance(value, dict):
        return value.get(lang, value.get(default_lang, fallback))
    return str(value)

def _find_localized_path(base_path: Path, ext: str, lang: str, default_lang: str) -> Optional[Path]:
    """Finds a localized file with fallback logic (e.g., content.ru.md -> content.en.md -> content.md)."""
    # 1. Try language-specific file
    lang_file = base_path.with_suffix(f".{lang}{ext}")
    if lang_file.exists():
        return lang_file
    
    # 2. Try default language file
    if lang != default_lang:
        default_file = base_path.with_suffix(f".{default_lang}{ext}")
        if default_file.exists():
            return default_file
            
    # 3. Try generic file
    generic_file = base_path.with_suffix(ext)
    if generic_file.exists():
        return generic_file
        
    return None

def _find_content_file(folder: Path, lang: str, default_lang: str) -> Tuple[str, Optional[Path]]:
    """Detects the appropriate content file (md or py) for a skill."""
    md_path = _find_localized_path(folder / "content", ".md", lang, default_lang)
    if md_path:
        return "md", md_path
    
    py_path = folder / "content.py"
    if py_path.exists():
        return "python", py_path
        
    return "none", None


# --- Single Entity Retrieval ---

def get_skill_by_id(skills: List[Skill], skill_id: str) -> Optional[Skill]:
    """Finds a skill in a list by its ID."""
    normalized_id = (skill_id or "").strip().lower()
    for skill in skills:
        if skill.id == normalized_id:
            return skill
    return None

def get_summary_path(lang: str, default_lang: str) -> Optional[Path]:
    """Returns the path to the localized summary.md file."""
    return _find_localized_path(PAGES_DIR / "summary", ".md", lang, default_lang)