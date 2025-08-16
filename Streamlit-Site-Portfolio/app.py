import streamlit as st
import logging

from core.state import AppState
from core.services import load_site_config, load_and_localize_skills, get_skill_by_id, get_summary_path
from core.constants import SUMMARY_SKILL_ID, ASSETS_DIR
from core.utils import read_text_file
from view.components import render_header, render_footer, render_sidebar_navigation
from view.presentation import render_summary_page, render_skill_details

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def apply_global_styles():
    """Reads and injects global CSS styles."""
    try:
        css = read_text_file(ASSETS_DIR / "style.css")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        logging.warning("assets/style.css not found. Global styles will not be applied.")

def main():
    """
    The main execution flow of the Streamlit application.
    """
    # --- 1. Initial Setup ---
    site_config = load_site_config()
    
    st.set_page_config(
        page_title=site_config.site_name,
        page_icon=site_config.branding.page_icon or "💼",
        layout="wide",
        initial_sidebar_state="expanded" if site_config.features.use_sidebar_nav else "collapsed",
    )
    apply_global_styles()

    # --- 2. State Initialization ---
    state = AppState.from_streamlit()
    
    # --- 3. Data Loading ---
    skills = load_and_localize_skills(lang=state.language, default_lang=site_config.i18n.default)

    # --- 4. Sidebar Rendering and State Update ---
    original_state = state.model_copy(deep=True)

    if site_config.features.use_sidebar_nav:        
        render_sidebar_navigation(skills, site_config, state) 
    else:        
        pass

    # --- 5. State Reconciliation and Rerun ---    
    if state != original_state:
        state.apply_to_streamlit()
        st.rerun()

    # --- 6. Main Content Rendering ---
    render_header(site_config, state.language)
    
    current_skill = get_skill_by_id(skills, state.skill_id)

    if state.skill_id == SUMMARY_SKILL_ID or not current_skill:
        summary_path = get_summary_path(state.language, site_config.i18n.default)
        summary_content = read_text_file(summary_path) if summary_path else ""
        render_summary_page(summary_content, skills, site_config, state.language)
    else:
        render_skill_details(current_skill, site_config, state.language)

    # --- 7. Footer ---
    render_footer(site_config)


if __name__ == "__main__":
    main()