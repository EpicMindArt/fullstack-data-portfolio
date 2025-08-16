from urllib.parse import urljoin
from bs4 import BeautifulSoup, Tag
import re

from core.parsers.base_parser import AbstractParser
from utils_scraping.models import Loggers

class QuotesTableParser(AbstractParser):
    def parse_items(self, soup: BeautifulSoup) -> list[dict]:
        """
        Finds all cells with quotes, 
        starting from them, finds related cells with tags.
        """
        items_data = []
        # Find all <td> cells containing "Author:", these are our "anchors".
        quote_cells = soup.find_all('td', string=re.compile(r'Author:'))

        for quote_cell in quote_cells:
            quote_row = quote_cell.find_parent('tr')
            tags_row = quote_row.find_next_sibling('tr') if quote_row else None
            
            if quote_row and tags_row:
                parsed_item = self._parse_single_item(quote_row, tags_row)
                if parsed_item:
                    items_data.append(parsed_item)
        
        return items_data

    def _parse_single_item(self, quote_row: Tag, tags_row: Tag) -> dict | None:
        """
        Extracts data from a guaranteed valid pair of <tr> strings.
        """
        try:
            full_text = quote_row.find('td').text.strip()
            
            match = re.search(r'^(“.*?”)\s*Author:\s*(.*)$', full_text, re.DOTALL)
            if not match:
                return None
            
            quote_text = match.group(1).strip()
            author_text = match.group(2).strip()

            tag_elements = tags_row.find_all('a')
            tags_list = [tag.text for tag in tag_elements]
            tags_str = ', '.join(tags_list) if tags_list else "N/A"
            
            unique_id = f"{author_text}_{len(quote_text)}"
            
            return {
                'quote': quote_text,
                'author': author_text,
                'tags': tags_str,
                'unique_id': unique_id
            }
        except (AttributeError, TypeError, IndexError) as e:
            self.logger.warning(f"Error parsing table record: {e}")
            return None

    def get_next_page_url(self, soup: BeautifulSoup, current_url: str) -> str | None:
        """
        A final, reliable pagination method based on a combination of features.
        """
        # 1. Find all links that can be paginated (contain /page/).
        potential_next_links = soup.select("a[href*='/page/']")
        
        # 2. We iterate through them to find the one that contains the text "Next".
        for link in potential_next_links:
            # Use .get_text() to reliably extract all text
            if "Next" in link.get_text():
                # 3. Once found, return its href.
                if link.has_attr('href'):
                    return urljoin(self.base_url, link['href'])
        
        # 4. If the cycle has ended and we have not returned anything, there is no "Next" button.
        return None