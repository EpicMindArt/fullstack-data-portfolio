# core/models.py

from __future__ import annotations
from typing import List, Optional, Literal, Dict, Callable
from pathlib import Path
import importlib.util
import logging

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings

from .utils import read_text_file, secure_path_resolve

logger = logging.getLogger(__name__)

# Type definition for the kind of content a skill can have.
ContentType = Literal["md", "python", "none"]


# --- Site Configuration Models ---

class LanguageEntry(BaseModel):
    code: str
    label: str
    flag: str = ""

class I18nConfig(BaseModel):
    default: str = "en"
    languages: List[LanguageEntry] = [LanguageEntry(code="en", label="English", flag="gb")]

class ResourcesConfig(BaseModel):
    resume_pdf: str = ""

class SearchConfig(BaseModel):
    tag_aliases: Dict[str, str] = {}

class SiteFeatures(BaseModel):
    use_sidebar_nav: bool = True
    default_skill: str = "summary"
    developer_mode: bool = False
    show_summary_in_category: bool = True

class Branding(BaseModel):
    page_icon: str = "💼"
    hero_image: str = ""

class SocialLinks(BaseModel):
    github: Optional[str] = None
    linkedin: Optional[str] = None
    telegram: Optional[str] = None

class SiteConfig(BaseSettings):
    site_name: str = "My Portfolio"
    tagline: str = ""
    author: str = ""
    nickname: str = ""
    email: str = ""
    social: SocialLinks = SocialLinks()
    features: SiteFeatures = SiteFeatures()
    branding: Branding = Branding()
    i18n: I18nConfig = I18nConfig()
    search: SearchConfig = SearchConfig()
    resources: ResourcesConfig = ResourcesConfig()


    class Config:
        env_prefix = 'APP_'
        env_nested_delimiter = '__'


# --- Core Domain Model ---

class Skill(BaseModel):
    """
    Represents a single skill, project, or portfolio piece.
    This model is not just a DTO; it contains methods to interact with its own data.
    """
    id: str
    name: str
    category: str
    summary: str = ""
    tags: List[str] = []
    tech_stack: List[str] = []
    order: int = 9999
    hero_image: str = ""
    gallery: List[str] = []
    video: str = ""
    demo_url: str = ""
    repo_url: str = ""
    
    content_type: ContentType = "none"
    content_path: Optional[Path] = None
    folder: Path

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        # Ensures skill IDs are consistent and URL-friendly.
        return v.strip().lower().replace(" ", "-")

    def get_content(self) -> str:
        """Reads and returns the skill's content from its markdown file."""
        if self.content_type == "md" and self.content_path and self.content_path.exists():
            return read_text_file(self.content_path)
        return ""

    def get_renderer(self) -> Optional[Callable]:
        """Loads and returns the render function from a skill's content.py file."""
        if self.content_type != "python" or not self.content_path or not self.content_path.exists():
            return None

        module_name = f"skills.{self.id}.content"
        spec = importlib.util.spec_from_file_location(module_name, self.content_path)
        
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                return getattr(module, "render", None)
            except Exception as e:
                logger.error("Failed to execute renderer module for skill '%s'", self.id, exc_info=True)
                return None
        return None

    def get_asset_path(self, asset_name: str) -> Optional[Path]:
        """
        Returns a secure, absolute path to an asset (like an image or video).
        Prevents path traversal attacks.
        """
        if not asset_name:
            return None
        try:
            return secure_path_resolve(self.folder, asset_name)
        except (ValueError, FileNotFoundError) as e:
            logger.warning(
                "Could not resolve asset '%s' for skill '%s': %s", 
                asset_name, self.id, e
            )
            return None