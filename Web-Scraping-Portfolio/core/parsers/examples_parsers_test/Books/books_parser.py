from urllib.parse import urljoin
from bs4 import BeautifulSoup, Tag

from core.parsers.base_parser import AbstractParser
from utils_scraping.models import Loggers

class BooksParser(AbstractParser):
    """
    Parser for the site books.toscrape.com.
    """
    
    # __init__ inherits from AbstractParser, self.base_url will be available

    def parse_items(self, soup: BeautifulSoup) -> list[dict]:
        """
        Finds all book cards on the page and runs 
        individual parsing for each one.
        """
        items_data = []
        
        item_tags = soup.select("article.product_pod")

        for item_tag in item_tags:
            parsed_item = self._parse_single_item(item_tag)
            if parsed_item:
                items_data.append(parsed_item)
        
        return items_data

    def _parse_single_item(self, item_tag: Tag) -> dict | None:
        """
        Extracts all the information you need from ONE book card.
        Dictionary keys must match the keys in DATA_STRUCTURE.
        """
        rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

        try:
            # Book Title and URL
            title_anchor = item_tag.find('h3').find('a')
            title = title_anchor['title']
            relative_book_url = title_anchor['href']
            book_url = urljoin(self.base_url, relative_book_url)

            # Image URL
            img_tag = item_tag.find('img', class_='thumbnail')
            relative_img_url = img_tag['src']
            image_url = urljoin(self.base_url, relative_img_url)

            # Price
            price_tag = item_tag.find('p', class_='price_color')
            # Clear the price from currency and convert to float
            price = float(price_tag.text.strip().replace('£', ''))

            # Rating
            rating_p = item_tag.find('p', class_='star-rating')
            # Class contains rating: "star-rating Three" -> extract "Three"
            rating_text = rating_p['class'][1]
            rating = rating_map.get(rating_text, 0)

            #Availability
            availability_tag = item_tag.find('p', class_='instock availability')
            availability = availability_tag.text.strip()
            
            return {
                'title': title,
                'price': price,
                'book_url': book_url,
                'image_url': image_url,
                'rating': rating,
                'availability': availability
            }
        except (AttributeError, TypeError, KeyError, IndexError) as e:
            self.logger.warning(f"Error parsing element: {e}. Tag: {item_tag}")
            return None

    def get_next_page_url(self, soup: BeautifulSoup, current_url: str) -> str | None:
        """
        Finds a link to the next page.
        """
        next_page_tag = soup.select_one("li.next > a")
        if next_page_tag and next_page_tag.has_attr('href'):
            next_page_url = next_page_tag['href']
            # urljoin will correctly handle both relative and absolute paths
            return urljoin(current_url, next_page_url)
        return None