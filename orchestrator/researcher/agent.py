import asyncio
from ddgs import DDGS
from google.adk.agents import Agent
from google.adk.models import Gemini
from ..utils.resilience import ResilientGemini

__all__ = ["researcher_agent"]

import httpx
from bs4 import BeautifulSoup

async def fetch_website_content(url: str) -> str:
    """Scrapes all readable body text content from a given URL, ignoring scripts and styles."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Remove noisy elements that don't contain core content
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.extract()
                
            # Extract text with spaces to avoid words running together
            text = soup.get_text(separator=' ', strip=True)
            
            if not text:
                return "No readable text found on the page."
            return f"Source URL: {url}\nContent:\n{text[:6000]}"
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
    description="Agent: Gathers maximum supplemental context by scraping the original archive URL and performing multiple targeted web searches.",
    tools=[fetch_website_content, duckduckgo_web_search],
    output_key="research_context", 
    instruction="""
ROLE: Elite Historical Researcher and Context Gatherer.

GOAL: Gather as much highly specific supplemental context as possible using BOTH the original archive URL (if available) AND multiple targeted web searches.

AVAILABLE DATA:
- Metadata: {discovered_archive}
- Contextual Vision Report: {vision_report}

STRICT WORKFLOW:
1. ORIGINAL URL SCRAPING: Check the Metadata for an `original_url`. If it exists, call `fetch_website_content` to retrieve the hidden context. (Note: Many archives do not have an original URL. If missing, skip this step).
2. ENTITY EXTRACTION: Identify specific Names, Dates, precise Locations, or specific Events from the Metadata, the Vision Report, AND the scraped original URL.
3. EXHAUSTIVE WEB SEARCH: Call `duckduckgo_web_search` multiple times if necessary. Build targeted queries using the exact entities extracted in step 2 to find as much additional context as possible. 
4. FILTER & COMBINE: Combine the useful facts from the original URL and the web searches. If the web search results DO NOT explicitly mention the specific entities you queried, discard those specific results.
5. OUTPUT: Output ALL the exact text/snippets caught from both the URL and searches, untouched and un-rewritten, with their Source Link/URLs. 

STRICT RULES:
- NO REWRITING. Do not summarize. Provide verbatim text.
- NO GENERAL TRIVIA. If the search returns generic encyclopedia data about "The Igbo people", discard it.
- MAXIMIZE CONTEXT: Your goal is to find as many specific facts as possible to enrich the archive. Do not stop at just one search if more entities can be explored.
- If no specific entities exist to search, and no original URL exists, output EXACTLY: "No specific supplemental context found."
""".strip()
)
