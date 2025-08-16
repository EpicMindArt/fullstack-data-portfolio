#!/usr/bin/env python3
# scripts/new_skill.py

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import textwrap

# Add project root to path to allow importing from 'core'
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.utils import slugify

SKILLS_DIR = ROOT / "skills"

META_TEMPLATE = """name: "{name}"
category: "{category}"
summary: "{summary}"
tags: []
tech_stack: []
order: 9999
hero_image: "assets/hero.png" # Example path
gallery: []
video: ""
demo_url: ""
repo_url: ""
"""

MD_TEMPLATE = """## {name}

A brief description of the skill and examples.

- Point 1
- Point 2
- Point 3
"""

PY_TEMPLATE = '''
import streamlit as st

# The `skill` and `ctx` arguments are passed by the renderer.
# `skill` is the Skill Pydantic model instance.
# `ctx` is a dictionary containing context like {'site_config': ..., 'lang': ...}
def render(st, skill, ctx):
    st.markdown("## {name}")
    st.write("This is a Python renderer. You can add interactive elements, charts, demos, etc.")
    
    # Example of using context:
    # lang = ctx.get("lang", "en")
    # st.write(f"Current language is: {lang}")
'''

def main():
    parser = argparse.ArgumentParser(description="Create a new skill scaffold.")
    parser.add_argument("name", help="The name of the skill, e.g., 'Web Scraping'")
    parser.add_argument("--category", default="misc", help="Category ID (e.g., 'development')")
    parser.add_argument("--summary", default="A short description.", help="A one-line summary")
    parser.add_argument("--python", action="store_true", help="Create content.py instead of content.md")
    args = parser.parse_args()

    skill_slug = slugify(args.name)
    if not skill_slug:
        print("Error: Could not generate a valid slug from the provided name.")
        sys.exit(1)

    target_dir = SKILLS_DIR / skill_slug
    if target_dir.exists():
        print(f"Error: Skill directory '{target_dir}' already exists.")
        sys.exit(1)
        
    target_dir.mkdir(parents=True)
    (target_dir / "assets").mkdir()

    meta_content = META_TEMPLATE.format(name=args.name, category=args.category, summary=args.summary)
    (target_dir / "meta.yaml").write_text(meta_content, encoding="utf-8")

    if args.python:
        code_content = PY_TEMPLATE.format(name=args.name)
        (target_dir / "content.py").write_text(code_content, encoding="utf-8")
        print(f"Created Python-based skill: {skill_slug}")
    else:
        md_content = MD_TEMPLATE.format(name=args.name)
        (target_dir / "content.md").write_text(md_content, encoding="utf-8")
        print(f"Created Markdown-based skill: {skill_slug}")

    print(f"Scaffold created at: {target_dir}")

if __name__ == "__main__":
    main()