# ======================================================
# company_research/tools/__init__.py
# CrewAI tool registry (factory-based, runtime-safe)
# ======================================================

from .custom_tool import (
    serp_api_tool,
    yahoo_finance_tool,
    news_api_tool,
    twitter_scraper_tool,
    wikipedia_tool,
    MyCustomTool,
)

# ✅ CrewAI expects these to exist by name — no `_factory` suffix
# These are callable functions or ready-to-use tool objects
__all__ = [
    "serp_api_tool",
    "yahoo_finance_tool",
    "news_api_tool",
    "twitter_scraper_tool",
    "wikipedia_tool",
    "MyCustomTool",
]

# ✅ Optional: keep a central registry (useful for debugging)
TOOL_FACTORIES = {
    "serp_api_tool": serp_api_tool,
    "yahoo_finance_tool": yahoo_finance_tool,
    "news_api_tool": news_api_tool,
    "twitter_scraper_tool": twitter_scraper_tool,
    "wikipedia_tool": wikipedia_tool,
}

if __name__ == "__main__":
    print("✅ Tool factories ready:", list(TOOL_FACTORIES.keys()))
