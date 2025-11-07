# ============================================================
#  Custom Tools for Company Research Crew
# ============================================================
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os

# ------------------------------------------------------------
# SerpAPI-based Search Tool
# ------------------------------------------------------------
from serpapi import GoogleSearch


class SerpApiQuery(BaseModel):
    """Input schema for performing Google search via SerpAPI"""
    query: str = Field(..., description="The search query to perform on Google.")


class SerpApiSearchTool(BaseTool):
    """Tool to perform a Google search using SerpAPI and return top results"""

    name: str = "serp_api_tool"  # must match YAML
    description: str = (
        "Performs a Google search using SerpAPI and returns the top search results "
        "including title, link, and snippet."
    )
    args_schema: Type[BaseModel] = SerpApiQuery

    def _run(self, query: str) -> str:
        """Execute the Google search and return formatted results."""
        api_key = os.getenv("SERP_API_KEY")

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
        }

        print(f"🔍 Searching Google for: {query}\n")

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            organic_results = results.get("organic_results", [])

            if not organic_results:
                print("⚠️ No search results found.")
                return f"⚠️ No results found for '{query}'."

            formatted_results = []
            for i, result in enumerate(organic_results[:5], start=1):
                title = result.get("title", "No title")
                link = result.get("link", "No link")
                snippet = result.get("snippet", "No description available.")
                formatted_results.append(f"{i}. {title}\n   {link}\n   {snippet}")

            output = "\n\n".join(formatted_results)
            print("✅ Top Results:\n", output)
            return f"🔍 Google Search Results for '{query}':\n\n{output}"

        except Exception as e:
            print("🚨 Error while fetching search results:", e)
            return f"❌ Error while performing search: {str(e)}"


# ✅ CrewAI expects this callable factory
def serp_api_tool():
    """Factory returning a SerpApiSearchTool instance"""
    return SerpApiSearchTool()


# ------------------------------------------------------------
# Safe imports from crewai_tools (with fallbacks if unavailable)
# ------------------------------------------------------------
try:
    from crewai_tools import (
        YahooFinanceTool,
        NewsAPITool,
        TwitterScraperTool,
    )
except ImportError:
    YahooFinanceTool = None
    NewsAPITool = None
    TwitterScraperTool = None


# ------------------------------------------------------------
# Tool Initializations (with stubs/fallbacks)
# ------------------------------------------------------------
# ✅ Yahoo Finance Tool
if YahooFinanceTool:
    yahoo_finance_tool = YahooFinanceTool()
else:
    class YahooFinanceToolStub(BaseTool):
        name: str = "yahoo_finance_tool"
        description: str = "Stub: YahooFinanceTool not installed."

        def _run(self, query: str) -> str:
            return f"(Stub) YahooFinanceTool unavailable. Simulated finance info for '{query}'."

    yahoo_finance_tool = YahooFinanceToolStub()

# ✅ News API Tool
if NewsAPITool:
    news_api_tool = NewsAPITool()
else:
    news_api_tool = serp_api_tool()  # fallback to web search

# ✅ Twitter Scraper Tool
if TwitterScraperTool:
    twitter_scraper_tool = TwitterScraperTool()
else:
    twitter_scraper_tool = serp_api_tool()  # fallback if not installed

# ------------------------------------------------------------
# Custom Wikipedia Tool
# ------------------------------------------------------------
import wikipedia
import wikipediaapi


class WikipediaToolInput(BaseModel):
    """Input schema for the Wikipedia tool."""
    query: str = Field(..., description="Topic or company name to look up on Wikipedia.")


class WikipediaTool(BaseTool):
    """Fetch summaries or related topics from Wikipedia."""
    name: str = "wikipedia_tool"  # must match YAML reference
    description: str = "Fetches company or topic summaries from Wikipedia."
    args_schema: Type[BaseModel] = WikipediaToolInput

    def _run(self, query: str = None, **kwargs) -> str:
        """Search Wikipedia for a topic and return a summary."""
        if not query:
            query = kwargs.get("query")
        if not query:
            return "❌ No query provided."

        query = str(query).strip()
        print(f"Using Tool: {self.name}\nSearching for: {query}")

        # Initialize Wikipedia APIs with polite identification
        wikipedia.set_lang("en")
        wiki_api = wikipediaapi.Wikipedia(
            language="en",
            user_agent="CompanyResearchBot/1.0 (contact: srivarshith@iiit.ac.in)"
        )

        # Try direct page lookup
        try:
            page = wiki_api.page(query)
            if page.exists():
                summary = page.summary or page.text
                url = f"https://en.wikipedia.org/wiki/{page.title.replace(' ', '_')}"
                return f"✅ Found: {page.title}\n🔗 {url}\n\n🔹 Summary:\n{summary[:1000]}"
        except Exception:
            pass

        # Fallback: keyword search
        try:
            candidates = wikipedia.search(query)
            if not candidates:
                return f"❌ No results found for '{query}'."

            for title in candidates[:5]:
                try:
                    page = wikipedia.page(title, auto_suggest=False, redirect=True)
                    summary = getattr(page, "summary", "")
                    if summary:
                        url = page.url if hasattr(page, "url") else f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        return f"⚠️ Closest match: {page.title}\n🔗 {url}\n\n🔹 Summary:\n{summary[:1000]}"
                except wikipedia.DisambiguationError as e:
                    return "⚠️ Ambiguous query. Options:\n" + "\n".join(f"- {opt}" for opt in e.options[:10])
                except wikipedia.PageError:
                    continue
        except Exception as e:
            return f"❌ Error fetching from Wikipedia: {e}"

        return f"❌ Could not find any page or summary for '{query}'."


# Initialize the tool instance (for YAML usage)
wikipedia_tool = WikipediaTool()

# ------------------------------------------------------------
# Example Custom Tool (for template/demo)
# ------------------------------------------------------------
class MyCustomToolInput(BaseModel):
    argument: str = Field(..., description="Test argument for demo.")


class MyCustomTool(BaseTool):
    name: str = "my_custom_tool"
    description: str = "Example tool to demonstrate custom tool creation."
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        return f"✅ Custom tool received argument: {argument}"
