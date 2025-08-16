# core/state.py

from __future__ import annotations
import streamlit as st
from pydantic import BaseModel, PrivateAttr

from .constants import SUMMARY_SKILL_ID, ALL_CATEGORIES_ID, DEFAULT_LANGUAGE


class AppState(BaseModel):
    """
    A single source of truth for the application's request-scoped state.
    It encapsulates query parameters and session state, making data flow explicit.
    """

    language: str = DEFAULT_LANGUAGE
    skill_id: str = SUMMARY_SKILL_ID
    category_id: str = ALL_CATEGORIES_ID

    _category_changed: bool = PrivateAttr(default=False)
    _lang_changed: bool = PrivateAttr(default=False)

    @classmethod
    def from_streamlit(cls) -> AppState:
        """
        Factory method to initialize the state from Streamlit's query and session state.
        This should be called once at the beginning of each script run.
        """
        # Priority: Query Param > Session State > Default
        lang = _get_query_param("lang") or st.session_state.get("lang", DEFAULT_LANGUAGE)
        skill_id = _get_query_param("skill", default=SUMMARY_SKILL_ID)
        category_id = st.session_state.get("category_filter", ALL_CATEGORIES_ID)
        
        # Update session state to be in sync
        st.session_state['lang'] = lang
        st.session_state['category_filter'] = category_id
        
        return cls(language=lang, skill_id=skill_id, category_id=category_id)

    def apply_to_streamlit(self) -> None:
        """
        Writes the current state back to Streamlit's query parameters.
        This is called when a state change needs to be reflected in the URL.
        """
        _set_query_param("lang", self.language)
        _set_query_param("skill", self.skill_id)
        # category_id is kept in session_state, not query params, for simplicity.
        st.session_state['category_filter'] = self.category_id

    def reset_skill_id(self):
        """Resets the skill ID to the default summary page."""
        self.skill_id = SUMMARY_SKILL_ID
        self._category_changed = True

    def set_language(self, lang_code: str):
        """Updates the language and flags that a change occurred."""
        if self.language != lang_code:
            self.language = lang_code
            self._lang_changed = True
            
    @property
    def category_changed(self) -> bool:
        return self._category_changed

    @property
    def language_changed(self) -> bool:
        return self._lang_changed

# --- Helper functions to interact with Streamlit's query params ---
def _get_query_param(name: str, default: str = "") -> str:
    """A robust way to get a single query parameter."""
    params = st.query_params
    value = params.get(name)
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value) if value is not None else default

def _set_query_param(name: str, value: str):
    """Sets a query parameter."""
    st.query_params[name] = value