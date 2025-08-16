from __future__ import annotations
from typing import List, Dict, Set

from .models import Skill


def normalize_tag(tag: str) -> str:
    """Converts a tag into a standardized format."""
    return (tag or "").strip().lower().replace(" ", "-")

def canonicalize_tags(tags: List[str], aliases: Dict[str, str]) -> List[str]:
    """Converts a list of tags to their canonical form using aliases."""
    canonical_tags = []
    for tag in tags:
        normalized = normalize_tag(tag)
        canonical_tags.append(aliases.get(normalized, normalized))
    return canonical_tags

def collect_all_tags(skills: List[Skill], aliases: Dict[str, str]) -> List[str]:
    """Gathers all unique, canonical tags from a list of skills."""
    all_normalized_tags: Set[str] = set()
    for skill in skills:
        for tag in skill.tags:
            all_normalized_tags.add(normalize_tag(tag))
    
    # Resolve aliases to get the final set of canonical tags
    canonical_set = {aliases.get(tag, tag) for tag in all_normalized_tags}
    return sorted(list(canonical_set))

def skill_matches_filters(
    skill: Skill,
    query: str,
    required_tags: List[str],
    aliases: Dict[str, str]
) -> bool:
    """Checks if a skill matches the search query and required tags."""
    # Normalize query and tags for comparison
    normalized_query = (query or "").strip().lower()
    skill_tags = set(canonicalize_tags(skill.tags, aliases))
    required_canonical_tags = set(canonicalize_tags(required_tags, aliases))

    # Tag filter: all selected tags must be present in the skill's tags
    if required_canonical_tags and not required_canonical_tags.issubset(skill_tags):
        return False

    # If there's no text query, and tags match (or no tags required), it's a match
    if not normalized_query:
        return True

    # Text search: check against name, category, summary, tags, and tech stack
    haystack = " ".join([
        skill.name.lower(),
        skill.category.lower(),
        skill.summary.lower(),
        " ".join(skill_tags),
        " ".join([t.lower() for t in skill.tech_stack])
    ])

    return normalized_query in haystack