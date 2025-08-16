from urllib.parse import urljoin
from bs4 import BeautifulSoup, Tag

from core.parsers.base_parser import AbstractParser
from utils_scraping.models import Loggers


class QuotesViewStateParser(AbstractParser):
    """Parser for the /filter.aspx pages of quotes.toscrape.com."""

    def parse_items(self, soup: BeautifulSoup) -> list[dict]:
        items_data = []
        item_tags = soup.find_all('div', class_='quote')

        if not item_tags:
            self.logger.info("No quote blocks found on the page.")
            return []

        for item_tag in item_tags:
            parsed_item = self._parse_single_item(item_tag)
            if parsed_item:
                items_data.append(parsed_item)
        
        return items_data

    def _parse_single_item(self, item_tag: Tag) -> dict | None:
        """Parses a single <div class="quote"> element."""
        try:
            quote_text = item_tag.find('span', class_='content').text.strip()
            author_text = item_tag.find('span', class_='author').text.strip()
            # This page has only one tag per quote
            tag_text = item_tag.find('span', class_='tag').text.strip()
            
            # Create a unique ID to prevent duplicates in the database
            unique_id = f"{author_text}_{len(quote_text)}"

            return {
                'author': author_text,
                'quote': quote_text,
                'tags': tag_text,
                'unique_id': unique_id
            }
        except AttributeError as e:
            self.logger.warning(f"Error parsing a single item: {e}. Tag: {item_tag}")
            return None


class QuotesPaginationParser(AbstractParser):
    """Parser for the main paginated quotes.toscrape.com site."""
    
    # The __init__ is inherited from AbstractParser, so we don't need to redeclare it
    # if we only need to call super(). The `base_url` will be set automatically.

    def parse_items(self, soup: BeautifulSoup) -> list[dict]:
        items_data = []
        item_tags = soup.find_all('div', class_='quote')

        for item_tag in item_tags:
            parsed_item = self._parse_single_item(item_tag)
            if parsed_item:
                items_data.append(parsed_item)
        
        return items_data

    def _parse_single_item(self, item_tag: Tag) -> dict:
        """Parses a single <div class="quote"> element from the main site."""
        quote_text = item_tag.find('span', class_='text').text.strip()
        author_text = item_tag.find('small', class_='author').text.strip()
        tags_list = [tag.text for tag in item_tag.find_all('a', class_='tag')]
        tags_str = ', '.join(tags_list)
        
        # Create a unique ID. A simple hash of the quote text could also work.
        unique_id = f"{author_text}_{len(quote_text)}"
        
        return {
            'author': author_text,
            'quote': quote_text,
            'tags': tags_str,
            'unique_id': unique_id
        }

    def get_next_page_url(self, soup: BeautifulSoup, current_url: str) -> str | None:
        """Finds the link to the next page."""
        next_page_tag = soup.find('li', class_='next')
        if next_page_tag and next_page_tag.a and next_page_tag.a.has_attr('href'):
            relative_url = next_page_tag.a['href']
            # Use urljoin to correctly combine the base URL and the relative path
            return urljoin(self.base_url, relative_url)
        return None