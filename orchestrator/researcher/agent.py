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
ROLE: Elite Historical & Geographical Researcher.

GOAL: Extract verbatim supplemental context. 

AVAILABLE DATA:
- Metadata: {discovered_archive}
- Blind Vision Report: {vision_report}

STRICT WORKFLOW:
1. If `original_url` exists, call `fetch_website_content`. 
2. Formulate 1 or 2 or 3 targeted search query via `duckduckgo_web_search`. Use details from the Metadata and Vision Report to find additional context not in the igbo archives metadata.
3. Output the exact text caught, untouched and un-rewritten, with the Source Link/URL. 

STRICT RULES:
- No rewriting.
- No conversational padding.
- No general cultural trivia.
""".strip()
)
