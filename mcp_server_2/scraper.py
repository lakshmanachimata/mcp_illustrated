"""Web scraping using Bright Data proxy (or direct HTTP)."""
import logging
import re

import httpx

from config import BRIGHT_DATA_PROXY

logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


def scrape_url(url: str, timeout: float = 30.0) -> dict:
    """
    Fetch a URL and return extracted text. Uses Bright Data proxy if configured.
    Returns dict with keys: success, url, text, error (if failed), title (if parsed).
    """
    url = (url or "").strip()
    if not url:
        return {"success": False, "url": "", "text": "", "error": "URL is required"}
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        kwargs = {"timeout": timeout, "follow_redirects": True}
        if BRIGHT_DATA_PROXY:
            kwargs["proxy"] = BRIGHT_DATA_PROXY
        with httpx.Client(**kwargs) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content_type = (resp.headers.get("content-type") or "").lower()
            text = resp.text
            if "text/html" in content_type and BeautifulSoup:
                soup = BeautifulSoup(text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                text = re.sub(r"\n{3,}", "\n\n", text)
                title = ""
                if soup.title:
                    title = soup.title.get_text(strip=True)
                return {"success": True, "url": url, "text": text[:500_000], "title": title}
            return {"success": True, "url": url, "text": text[:500_000], "title": ""}
    except httpx.HTTPStatusError as e:
        logger.warning("Scrape HTTP error %s: %s", e.response.status_code, url)
        return {"success": False, "url": url, "text": "", "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        logger.exception("Scrape failed for %s", url)
        return {"success": False, "url": url, "text": "", "error": str(e)}
