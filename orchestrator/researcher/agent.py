import asyncio
from ddgs import DDGS
from google.adk.agents import Agent
from google.adk.models import Gemini
from ..utils.resilience import ResilientGemini

__all__ = ["researcher_agent"]

import httpx
from bs4 import BeautifulSoup

async def fetch_website_content(url: str) -> str:
    """Scrapes paragraph text content from a given URL."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            text = " ".join([p.get_text(separator=' ', strip=True) for p in soup.find_all('p')])
            if not text:
                return "No readable text found on the page."
            return f"Source URL: {url}\nContent:\n{text[:4000]}"
    except Exception as e:
        return f"Failed to fetch content from {url}: {str(e)}"

async def duckduckgo_web_search(query: str) -> str:
    """Searches the internet for historical and geographical context."""
    try:
        def _search():
            # Use the provided historical logic
            results = DDGS().text(query, max_results=4)
            results = list(results)
            if not results:
                return "No results found."
            return "\n\n".join([f"Source: {r.get('title', '')}\nLink: {r.get('href', '')}\nSnippet: {r.get('body', '')}" for r in results])
            
        return await asyncio.to_thread(_search)
    except Exception as e:
        return f"Search failed: {str(e)}"

researcher_agent = Agent(
    name="ResearcherAgent",
    model=ResilientGemini(
        model="models/gemma-4-26b-a4b-it",
        fallbacks=["models/gemma-4-31b-it"]
    ),
    description="Agent: Checks the archive URL for extra context, or searches the internet if it's missing.",
    tools=[fetch_website_content, duckduckgo_web_search],
    output_key="research_context", 
    instruction="""
ROLE:
You are an Elite Historical & Geographical Researcher for the Igbo Archives.

GOAL:
Read the completely untouched archive obtained by the discoverer. 
RAW ARCHIVE METADATA: {discovered_archive}

There are NO notes on this archive. You need to find extra context and place it in the state EXACTLY as you see it. 
DO NOT rewrite anything or use your mind because you will hallucinate. DO NOT try to identify missing context.

STRICT RULES:
1. If the archive has an `original_url` or a website link, use `fetch_website_content` to read the exact text from that URL.
2. Also, formulate ONE search query using `duckduckgo_web_search` and find extra context for the archives not in description or original url because the more the extra context, the more notes we can write.
3. If no `original_url` proceed to get context from duckduckgo_web_search.
4. Your FINAL output MUST be exactly what you caught/fetched (the exact text and snippet), untouched and un-rewritten, with the Link/URL attached to it.
5. DO NOT add conversational padding. Just output the verbatim context and its URL.
"""
)
