from __future__ import annotations
from typing import List
import html
from pathlib import Path

import streamlit as st

from core.models import SiteConfig, Skill
from core.i18n import t
from core.utils import load_image, read_text_file


def _render_skill_card(skill: Skill, lang: str):
    """Renders a single skill card as a self-contained HTML block."""
    
    # Escape all user data for security
    skill_name_safe = html.escape(skill.name)
    summary_safe = html.escape(skill.summary)
    tags_safe = [html.escape(tag) for tag in skill.tags]
    tags_title_safe = html.escape(t("ui.tags", lang))
    button_text_safe = html.escape(t("ui.view_details", lang))

    # Generate HTML for chips with tags
    tags_html = ""
    if tags_safe:
        chips_html = "".join([f'<span class="chip">{tag}</span>' for tag in tags_safe])
        tags_html = f'<div class="chips-line"><strong>{tags_title_safe}:</strong> {chips_html}</div>'

    # Assemble the entire card as one F-string
    # ?lang={lang} in the link to preserve the language when switching
    card_html = f"""
    <a href="?skill={skill.id}&lang={lang}" target="_self" class="skill-card-link">
        <div class="skill-card">
            <div class="skill-card-content">
                <h5>{skill_name_safe}</h5>
                <p class="caption">{summary_safe}</p>
                {tags_html}
            </div>
            
        </div>
    </a>
    """
    
    # Unfortunately, we can't put a button inside a link.
    # So let's make the entire card clickable.
    # If you need a button, the code is below.
    
    button_html = f'<a href="?skill={skill.id}&lang={lang}" target="_self" class="skill-card-button">{button_text_safe}</a>'
    
    final_card_html = f"""
    <div class="skill-card">
        <div class="skill-card-content">
            <h5>{skill_name_safe}</h5>
            <p class="caption">{summary_safe}</p>
            {tags_html}
        </div>
        {button_html}
    </div>
    """

    st.markdown(final_card_html, unsafe_allow_html=True)


def render_summary_page(summary_content: str, skills: List[Skill], site_config: SiteConfig, lang: str):
    """Renders the main summary/landing page."""
    if summary_content:
        st.markdown(summary_content, unsafe_allow_html=True)
    
    if site_config.resources.resume_pdf:
        resume_path = Path(site_config.resources.resume_pdf)
        if resume_path.exists():
            with resume_path.open("rb") as f:
                st.download_button(
                    label=t("ui.download_resume", lang, default="Download Resume"),
                    data=f.read(),
                    file_name=resume_path.name,
                    mime="application/pdf",
                )
    
    st.markdown(f"## {t('ui.my_skills', lang)}")
    
    num_columns = 3
    skills_to_show = [s for s in skills if s.id != "summary"]    
    
    rows = [skills_to_show[i:i + num_columns] for i in range(0, len(skills_to_show), num_columns)]
    
    for row_skills in rows:
        cols = st.columns(num_columns)
        for i, skill in enumerate(row_skills):
            with cols[i]:
                _render_skill_card(skill, lang)


def render_skill_details(skill: Skill, site_config: SiteConfig, lang: str):
    """Renders the detailed page for a single skill."""
    st.header(skill.name)
    st.caption(f"{t('ui.category', lang)}: {t(f'categories.{skill.category}', lang, default=skill.category)}")
    
    if skill.summary:
        st.markdown(f"> {skill.summary}")

    links = []
    if skill.demo_url:
        links.append(f"[{t('ui.demo', lang)}]({skill.demo_url})")
    if skill.repo_url:
        links.append(f"[{t('ui.repo', lang)}]({skill.repo_url})")
    if links:
        st.write(" | ".join(links))

    _render_badges_line(t("ui.tags", lang), skill.tags)
    _render_badges_line(t("ui.tech_stack", lang), skill.tech_stack)
    st.write("---")

    hero_path = skill.get_asset_path(skill.hero_image)
    if hero_path:
        st.image(load_image(hero_path), use_column_width=True)

    content = skill.get_content()
    if content:
        st.markdown(content, unsafe_allow_html=True)

    renderer = skill.get_renderer()
    if renderer:
        renderer_context = {"site_config": site_config, "lang": lang}
        renderer(st, skill, renderer_context)
    
    gallery_paths = [p for p in (skill.get_asset_path(img) for img in skill.gallery) if p]
    if gallery_paths:
        st.subheader(t("ui.gallery", lang))
        cols = st.columns(min(3, len(gallery_paths)))
        for i, path in enumerate(gallery_paths):
            cols[i % 3].image(load_image(path), use_column_width=True)

    video_path = skill.get_asset_path(skill.video)
    if video_path:
        st.subheader(t("ui.video", lang))
        st.video(str(video_path))
    elif skill.video.startswith("http"):
        st.subheader(t("ui.video", lang))
        st.video(skill.video)


def _render_badges_line(title: str, items: List[str]):
    """Helper to render a line of styled chips/badges."""
    if not items:
        return
    chips = " ".join([f'<span class="chip">{html.escape(i)}</span>' for i in items])
    st.markdown(
        f'<div class="chips-line"><strong>{title}:</strong> {chips}</div>', 
        unsafe_allow_html=True
    )