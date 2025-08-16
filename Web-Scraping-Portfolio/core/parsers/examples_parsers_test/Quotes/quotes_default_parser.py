from urllib.parse import urljoin
from bs4 import BeautifulSoup, Tag

# Use absolute import for reliability
from core.parsers.base_parser import AbstractParser
from utils_scraping.models import Loggers

class QuotesDefaultParser(AbstractParser):
    """
    Parser for the main endpoint of quotes.toscrape.com.
    """
    # Constructor and self.logger/self.base_url inherit from AbstractParser

    def parse_items(self, soup: BeautifulSoup) -> list[dict]:
        """Finds all .quote containers and passes them to the single element parser."""
        items_data = []
        item_tags = soup.select(
            "div.quote"
        ) # Selector from config

        for item_tag in item_tags:
            parsed_item = self._parse_single_item(item_tag)
            if parsed_item:
                items_data.append(parsed_item)
        
        return items_data

    def _parse_single_item(self, item_tag: Tag) -> dict | None:
        """
        Extracts data from a single <div class="quote"> block.
        If some required element is not found, returns None,
        without interrupting the entire scraper.
        """
        try:
            # --- Required fields: if they are missing, the whole entry is meaningless ---
            quote_text = item_tag.find('span', class_='text').text.strip()
            author_text = item_tag.find('small', class_='author').text.strip()
            
            # --- Optional fields: if none, use default value ---
            
            # About link of the author
            author_about_tag = item_tag.find('a', href=lambda href: href and href.startswith('/author'))
            author_about_url = urljoin(self.base_url, author_about_tag['href']) if author_about_tag else "N/A"
            
            # Tags
            tags_list = []
            tags_container = item_tag.find('div', class_='tags')
            if tags_container:
                tag_elements = tags_container.find_all('a', class_='tag')
                tags_list = [tag.text for tag in tag_elements]
            
            tags_str = ', '.join(tags_list) if tags_list else "N/A"
            
            unique_id = f"{author_text}_{len(quote_text)}"
            
            return {
                'quote': quote_text,
                'author': author_text,
                'author_about_url': author_about_url,
                'tags': tags_str,
                'unique_id': unique_id
            }
        # This block will catch the error if .find() returns None and we try to get .text from it
        except (AttributeError, TypeError) as e:
            self.logger.warning(f"Error parsing quote. Key tag may be missing. Error: {e}. Tag: {item_tag}")
            return None

    def get_next_page_url(self, soup: BeautifulSoup, current_url: str) -> str | None:
        """Finds the URL of the next page."""
        next_page_tag = soup.select_one("li.next > a")
        if next_page_tag and next_page_tag.has_attr('href'):
            next_page_url = next_page_tag['href']
            return urljoin(current_url, next_page_url)
        return None