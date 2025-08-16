# view/components.py

from __future__ import annotations
from typing import List, Dict

import streamlit as st

from core.models import SiteConfig, Skill
from core.state import AppState
from core.i18n import t
from core.search import collect_all_tags, skill_matches_filters
from core.constants import SUMMARY_SKILL_ID, ALL_CATEGORIES_ID
from core.utils import clear_all_caches


def render_header(site_config: SiteConfig, lang: str):
    """Renders the main page header with title, tagline and social links."""
    left, right = st.columns([2, 1])
    with left:        
        st.title(f"{site_config.branding.page_icon} {t('site.site_name', lang)}")
        if site_config.tagline:
             st.caption(t('site.tagline', lang))
             
    with right:
        links = []
        if site_config.nickname:
            links.append(f"**@{t('site.nickname', lang, default=site_config.nickname)}**")
        if site_config.social.github:
            links.append(f"[GitHub]({site_config.social.github})")
        if site_config.social.linkedin:
            links.append(f"[LinkedIn]({site_config.social.linkedin})")
        if site_config.social.telegram:
            links.append(f"[Telegram]({site_config.social.telegram})")
        
        if links:
            st.write(" &nbsp;•&nbsp; ".join(links))

        if site_config.features.developer_mode:
            st.caption("Developer mode is ON")


def render_footer(site_config: SiteConfig):
    """Renders the page footer."""
    st.write("---")
    st.caption(f"© {site_config.author or site_config.site_name}")


def render_language_switcher(site_config: SiteConfig, current_lang: str) -> str:
    """Renders the language selection dropdown WITH a vertically centered flag."""
    languages = site_config.i18n.languages
    if len(languages) <= 1:
        return current_lang

    codes = [lang.code for lang in languages]
    labels = [lang.label for lang in languages]
    flags = {lang.code: lang.flag for lang in languages}
    
    try:
        current_idx = codes.index(current_lang)
    except ValueError:
        current_idx = 0

    col1, col2 = st.sidebar.columns([1, 4])
    
    with col1:
        flag_code = flags.get(current_lang)
        if flag_code:
            st.markdown(
                f'<div style="height: 38px; display: flex; align-items: center; justify-content: center;">'
                f'<img src="https://flagcdn.com/24x18/{flag_code}.png">'
                f'</div>', 
                unsafe_allow_html=True
            )

    with col2:
        selected_label = st.selectbox(
            label=t("ui.language", current_lang),
            options=labels,
            index=current_idx,
            label_visibility="collapsed"
        )
    
    return codes[labels.index(selected_label)]


def render_sidebar_navigation(skills: List[Skill], site_config: SiteConfig, state: AppState) -> AppState:
    """Renders the entire sidebar for navigation and filtering, returning the new app state."""
    st.sidebar.header(t("ui.navigation", state.language))
    
    new_lang = render_language_switcher(site_config, state.language)
    if new_lang != state.language:
        state.set_language(new_lang)
    
    category_ids = sorted(set(s.category for s in skills if s.category))
    category_labels = {cid: t(f"categories.{cid}", state.language, default=cid) for cid in category_ids}
    inv_category_labels = {v: k for k, v in category_labels.items()}
    
    def on_category_change():
        selected_label = st.session_state.category_selector
        category_id = inv_category_labels.get(selected_label, ALL_CATEGORIES_ID)
        st.session_state.category_filter = category_id
        st.query_params["skill"] = SUMMARY_SKILL_ID

    all_cat_label = t("ui.all_categories", state.language)
    options = [all_cat_label] + list(category_labels.values())
    
    current_label = category_labels.get(state.category_id, all_cat_label)
    try:
        cat_index = options.index(current_label)
    except ValueError:
        cat_index = 0
        
    st.sidebar.selectbox(
        label=t("ui.category", state.language),
        options=options,
        index=cat_index,
        key="category_selector", 
        on_change=on_category_change 
    )
    
    new_category_id = st.session_state.get("category_filter", ALL_CATEGORIES_ID)
    if new_category_id != state.category_id:
        state.category_id = new_category_id

    search_query = st.sidebar.text_input(t("ui.search", state.language))
    aliases = site_config.search.tag_aliases
    all_tags = collect_all_tags(skills, aliases)
    selected_tags = st.sidebar.multiselect(t("ui.tags", state.language), options=all_tags, default=[])

    filtered_skills = []
    for s in skills:
        if state.category_id != ALL_CATEGORIES_ID and s.category != state.category_id:
            continue
        if not skill_matches_filters(s, search_query, selected_tags, aliases):
            continue
        filtered_skills.append(s)

    if not filtered_skills and state.category_id != ALL_CATEGORIES_ID:
        st.sidebar.warning("No skills found for this filter.")
    
    skill_options: Dict[str, str] = {}
    if state.category_id == ALL_CATEGORIES_ID or site_config.features.show_summary_in_category:
        skill_options[SUMMARY_SKILL_ID] = t("ui.summary", state.language)
    
    for s in filtered_skills:
        skill_options[s.id] = s.name
        
    if state.skill_id not in skill_options:
        if skill_options:
            state.skill_id = list(skill_options.keys())[0]
        else:
            state.skill_id = SUMMARY_SKILL_ID 
            if state.skill_id not in skill_options:
                 skill_options[SUMMARY_SKILL_ID] = t("ui.summary", state.language)

    selected_skill_id = st.sidebar.selectbox(
        label=t("ui.select_skill", state.language),
        options=list(skill_options.keys()),
        index=list(skill_options.keys()).index(state.skill_id),
        format_func=lambda skill_id: skill_options.get(skill_id, "???")
    )
    state.skill_id = selected_skill_id
    
    if site_config.features.developer_mode:
        st.sidebar.button("Clear App Caches", on_click=clear_all_caches, use_container_width=True)

    return state