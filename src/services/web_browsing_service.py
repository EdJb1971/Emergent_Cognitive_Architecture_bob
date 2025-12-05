import logging
import json
from typing import Dict, Any, List, Optional
from uuid import UUID
import aiohttp
from bs4 import BeautifulSoup
try:
    from serpapi import GoogleSearch  # Optional dependency
except Exception:  # ImportError or other
    GoogleSearch = None

from src.core.exceptions import LLMServiceException, APIException
from src.services.llm_integration_service import LLMIntegrationService
from src.core.config import settings

logger = logging.getLogger(__name__)

class WebBrowsingService:
    """
    A sandboxed backend service that enables autonomous web exploration.
    It can perform web searches, navigate pages, extract information, and summarize content.
    """
    SERVICE_ID = "web_browsing_service"
    MODEL_NAME = settings.LLM_MODEL_NAME

    def __init__(self, llm_service: LLMIntegrationService):
        self.llm_service = llm_service
        # Provider detection
        self.serpapi_api_key: Optional[str] = settings.SERPAPI_API_KEY
        self.google_api_key: Optional[str] = settings.GOOGLE_API_KEY
        self.google_cse_id: Optional[str] = settings.GOOGLE_CSE_ID

        if self.serpapi_api_key:
            self.provider = "serpapi"
            if GoogleSearch is None:
                logger.warning(f"{self.SERVICE_ID}: SERPAPI_API_KEY provided but 'serpapi' package is not installed. Falling back to disabled state.")
        elif self.google_api_key and self.google_cse_id:
            self.provider = "google_cse"
        else:
            self.provider = None
            logger.warning(f"{self.SERVICE_ID} is not configured. Provide SERPAPI_API_KEY or GOOGLE_API_KEY + GOOGLE_CSE_ID in .env to enable web browsing.")

        logger.info(f"{self.SERVICE_ID} initialized. Provider: {self.provider or 'disabled'}")

    def is_enabled(self) -> bool:
        return bool(self.provider == "serpapi" and GoogleSearch is not None) or bool(self.provider == "google_cse")

    async def _search(self, query: str) -> List[Dict[str, Any]]:
        """Perform a web search via configured provider and return a list of result dicts
        with at least: {'link': str, 'title': str, 'snippet': str}.
        """
        try:
            if self.provider == "serpapi":
                if GoogleSearch is None:
                    raise APIException(detail="SerpAPI provider unavailable (missing package).", status_code=501)
                params = {
                    "q": query,
                    "api_key": self.serpapi_api_key
                }
                search = GoogleSearch(params)
                results = search.get_dict()
                out: List[Dict[str, Any]] = []
                for item in results.get('organic_results', []) or []:
                    link = item.get('link')
                    if not link:
                        continue
                    out.append({
                        "link": link,
                        "title": item.get('title', ''),
                        "snippet": item.get('snippet', '')
                    })
                return out

            if self.provider == "google_cse":
                # Google Programmable Search Engine (Custom Search JSON API)
                endpoint = "https://www.googleapis.com/customsearch/v1"
                params = {
                    "key": self.google_api_key,
                    "cx": self.google_cse_id,
                    "q": query
                }
                async with aiohttp.ClientSession() as session:
                    async with session.get(endpoint, params=params, timeout=10) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        items = data.get("items", []) or []
                        out: List[Dict[str, Any]] = []
                        for it in items:
                            link = it.get("link")
                            if not link:
                                continue
                            out.append({
                                "link": link,
                                "title": it.get("title", ""),
                                "snippet": it.get("snippet", "")
                            })
                        return out

            # No provider configured
            raise APIException(detail="Web browsing provider not configured.", status_code=501)
        except APIException:
            raise
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during search: {e}", exc_info=True)
            raise APIException(detail="Search HTTP error.", status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error during web search: {e}", exc_info=True)
            raise APIException(detail="An unexpected error occurred during web search.", status_code=500)

    async def _scrape(self, url: str) -> str:
        """
        Scrapes the content of a given URL and returns the text.
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9"
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                    return soup.get_text(separator='\n', strip=True)
        except aiohttp.ClientError as e:
            logger.error(f"aiohttp.ClientError during scraping {url}: {e}", exc_info=True)
            return "" # Return empty string if scraping fails
        except Exception as e:
            logger.error(f"Unexpected error during scraping {url}: {e}", exc_info=True)
            return "" # Return empty string for any other scraping errors

    async def browse_and_scrape(self, query: str, user_id: UUID) -> Dict[str, Any]:
        """
        Performs a web search, scrapes the top results, and returns summarized content.
        """
        if not self.is_enabled():
            raise APIException(detail="Web browsing service is not configured.", status_code=501)

        logger.info(f"AUDIT: Web browsing initiated for user {user_id} with query: '{query}' via provider: {self.provider}")

        results = await self._search(query)
        if not results:
            raise APIException(detail="No search results found for the query.", status_code=404)
        urls = [r.get("link") for r in results if r.get("link")]

        scraped_content = []
        for url in urls:
            content = await self._scrape(url)
            if content:
                scraped_content.append(content)

        if not scraped_content:
            # Fallback: synthesize context from search titles + snippets
            fallback_blocks: List[str] = []
            for r in results[:10]:
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                link = r.get("link", "")
                if title or snippet:
                    fallback_blocks.append(f"Title: {title}\nSnippet: {snippet}\nSource: {link}")
            if not fallback_blocks:
                raise APIException(detail="Could not scrape any content or snippets from the search results.", status_code=500)
            scraped_content.append("\n\n".join(fallback_blocks))

        combined_content = "\n\n".join(scraped_content)
        
        prompt = f"""
        Summarize the following content scraped from the web based on the query "{query}".
        Focus on extracting key information, insights, and relevant data points.
        
        Content:
        {combined_content[:8000]}
        """

        try:
            summary = await self.llm_service.generate_text(
                prompt=prompt,
                model_name=self.MODEL_NAME,
                temperature=0.4,
                max_output_tokens=1500
            )

            moderation_result = await self.llm_service.moderate_content(summary)
            if not moderation_result.get("is_safe"):
                logger.warning(f"WebBrowsingService: LLM-generated summary blocked due to safety concerns for query '{query[:50]}...': {moderation_result.get('block_reason', 'N/A')}")
                raise APIException(
                    detail="Generated web content summary was flagged as unsafe and cannot be processed.",
                    status_code=400
                )

            logger.info(f"WebBrowsingService: Successfully summarized web content for query '{query[:50]}...'.")
            return {
                "query": query,
                "summary": summary,
                "sources": urls
            }

        except LLMServiceException as e:
            logger.error(f"{self.SERVICE_ID} failed to get LLM response for query '{query}': {e.detail}", exc_info=True)
            raise APIException(
                detail=f"Failed to summarize web content: {e.detail}",
                status_code=e.status_code
            )
        except APIException:
            raise
        except Exception as e:
            logger.exception(f"{self.SERVICE_ID} encountered an unexpected error during summarization for query '{query}'.")
            raise APIException(
                detail=f"An unexpected error occurred during web content summarization: {e}",
                status_code=500
            )
