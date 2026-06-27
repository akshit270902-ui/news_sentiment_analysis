"""
Resolve real headlines for URLs whose source data only gives a slug/path,
via concurrent async HTTP requests (OG/Twitter meta tags, <h1>, <title>),
with a regex-based URL-slug fallback when fetching fails or is skipped.
"""
import asyncio
import re
from urllib.parse import urlparse

import aiohttp
import async_timeout
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from config import FETCH_HEADERS, FETCH_TIMEOUT, MAX_CONCURRENT_REQUESTS, MAX_PER_HOST


def extract_title_from_url(url):
    """Best-effort fallback: turn a URL path slug into a readable title."""
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url.strip())
        path = parsed.path
        parts = [p for p in path.split("/") if p]
        candidates = []
        for part in reversed(parts):
            clean = re.sub(r"\.(html?|php|aspx?|cfm|htm)$", "", part, flags=re.I)
            clean = re.sub(r"[-_]", " ", clean)
            clean = re.sub(r"\d{8,}", "", clean)
            clean = re.sub(r"\s{2,}", " ", clean).strip()
            if len(clean.split()) >= 4:
                candidates.append(clean)
        if candidates:
            return max(candidates, key=len)
        slug = parts[-1] if parts else ""
        slug = re.sub(r"\.(html?|php|aspx?|cfm|htm)$", "", slug, flags=re.I)
        slug = re.sub(r"[-_]", " ", slug)
        slug = re.sub(r"\d{8,}", "", slug)
        slug = re.sub(r"\s{2,}", " ", slug).strip()
        return slug
    except Exception:
        return ""


async def _fetch_one_headline_async(session, url):
    if not url or not isinstance(url, str) or not url.startswith("http"):
        return url, extract_title_from_url(url)
    try:
        async with async_timeout.timeout(FETCH_TIMEOUT):
            async with session.get(url, headers=FETCH_HEADERS, allow_redirects=True) as resp:
                if resp.status == 200:
                    ct = resp.headers.get("Content-Type", "")
                    if "text/html" not in ct and "text/plain" not in ct:
                        return url, extract_title_from_url(url)

                    html = await resp.text(errors="ignore")
                    soup = BeautifulSoup(html, "lxml")

                    for prop in ("og:title", "twitter:title"):
                        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
                        if tag:
                            content = tag.get("content", "").strip()
                            if content and len(content.split()) >= 4:
                                return url, content

                    h1 = soup.find("h1")
                    if h1:
                        text = h1.get_text(" ", strip=True)
                        if text and len(text.split()) >= 4:
                            return url, text

                    title_tag = soup.find("title")
                    if title_tag:
                        text = title_tag.get_text(" ", strip=True)
                        text = re.split(r"\s*[|\-\u2013\u2014]\s*", text)[0].strip()
                        if text and len(text.split()) >= 4:
                            return url, text

    except Exception:
        pass
    return url, extract_title_from_url(url)


async def _fetch_all_urls_async(unique_urls):
    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS,
        limit_per_host=MAX_PER_HOST,
        ssl=False,
    )
    result_map = {}
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def worker(url):
            async with semaphore:
                return await _fetch_one_headline_async(session, url)

        tasks = [worker(url) for url in unique_urls]
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching headlines asynchronously"):
            url, title = await future
            result_map[url] = title
    return result_map


def fetch_headlines_concurrent(urls):
    """Resolve a list of URLs to headlines, deduplicating requests for repeated URLs."""
    url_list = list(urls)
    unique_urls = list(set(url_list))

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    result_map = loop.run_until_complete(_fetch_all_urls_async(unique_urls))

    return [result_map.get(u, extract_title_from_url(u)) for u in url_list]
