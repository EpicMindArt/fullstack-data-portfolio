from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from utils_scraping.models import Loggers

class AbstractParser(ABC):
    """
    Abstract Base Class for all parsers.
    It defines the interface that every concrete parser must implement,
    ensuring consistency across the framework.
    """
    def __init__(self, logger: Loggers, **kwargs):
        """
        Initializes the parser.
        
        :param logger: An instance of the Loggers dataclass.
        :param kwargs: Catches any extra arguments passed during dynamic instantiation,
                       such as `base_url`.
        """
        self.logger = logger.combined
        # Store base_url if provided, useful for resolving relative links
        self.base_url = kwargs.get("base_url")

    @abstractmethod
    def parse_items(self, soup: BeautifulSoup) -> list[dict]:
        """
        The primary method that takes a BeautifulSoup object of a page
        (or a page fragment) and returns a list of dictionaries, where
        each dictionary represents a scraped item.
        """
        raise NotImplementedError

    def get_next_page_url(self, soup: BeautifulSoup, current_url: str) -> str | None:
        """
        An optional method to find the URL of the next page for pagination strategies.
        Returns the full URL of the next page, or None if it's not found.
        This method must be implemented by parsers used with the pagination strategy.
        By default, it is not implemented.
        """
        # self.logger.warning(
        #     f"The 'get_next_page_url' method was called but is not implemented in {self.__class__.__name__}."
        # )
        return None