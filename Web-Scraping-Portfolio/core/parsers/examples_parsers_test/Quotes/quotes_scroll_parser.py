from urllib.parse import urljoin
from bs4 import BeautifulSoup, Tag

from core.parsers.base_parser import AbstractParser
from utils_scraping.models import Loggers

class QuotesScrollParser(AbstractParser):
    """
    A specialized parser for the /scroll endpoint of quotes.toscrape.com.
    It is designed to work with the HTML fragment of ONE quote,
    obtained via inner_html().
    """

    def parse_items(self, soup: BeautifulSoup) -> list[dict]:
        """
        Accepts a soup object containing the HTML of a single quote.
        Extracts the data and returns it as a list with one element.
        """
        try:
            # --- Since we are being passed the contents of ONE .quote, we are looking for elements directly ---
            quote_text = soup.find('span', class_='text').text.strip()
            author_text = soup.find('small', class_='author').text.strip()
            
            # --- Optional fields ---
            
            # There are no links to the author on this endpoint, so we put a stub
            author_about_url = "N/A"
            
            # Tags
            tags_list = []
            # div.tags is the root tag here, find() will find it
            tags_container = soup.find('div', class_='tags')
            if tags_container:
                tag_elements = tags_container.find_all('a', class_='tag')
                tags_list = [tag.text for tag in tag_elements]
            
            tags_str = ', '.join(tags_list) if tags_list else "N/A"
            
            unique_id = f"{author_text}_{len(quote_text)}"
            
            parsed_item = {
                'quote': quote_text,
                'author': author_text,
                'author_about_url': author_about_url,
                'tags': tags_str,
                'unique_id': unique_id
            }
            # The method must return a list, even if it has one element
            return [parsed_item]
            
        except (AttributeError, TypeError) as e:
            self.logger.warning(f"Error parsing quote fragment. Error: {e}. Fragment: {soup}")
            return []

    # The get_next_page_url method is not needed for the scroll strategy, 
    # so we don't implement it. 