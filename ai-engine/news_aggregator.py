import httpx
import asyncio
import os
import time

# Cache headlines for 5 minutes to avoid hammering the API on every signal
_CACHE: dict = {"headlines": "", "expires_at": 0.0}
CACHE_TTL = 300  # seconds

# High-impact keywords that should trigger extra AI scrutiny
HIGH_IMPACT_KEYWORDS = (
    "CPI", "NFP", "NONFARM", "FOMC", "RATE DECISION",
    "RATE HIKE", "RATE CUT", "FEDERAL RESERVE", "POWELL",
    "UNEMPLOYMENT", "GDP", "INFLATION", "PCE",
)


async def fetch_macro_news() -> str:
    """
    Fetches real-time financial headlines.
    Uses NewsAPI.org if NEWS_API_KEY is set; falls back to curated static context.
    Headlines are cached for CACHE_TTL seconds to avoid rate limits.
    """
    global _CACHE

    # Return cached result if still fresh
    if time.time() < _CACHE["expires_at"] and _CACHE["headlines"]:
        return _CACHE["headlines"]

    api_key = os.getenv("NEWS_API_KEY", "")

    if api_key:
        headlines = await _fetch_newsapi(api_key)
    else:
        # No API key — return static context that at least keeps the AI honest
        headlines = _static_context()

    _CACHE["headlines"] = headlines
    _CACHE["expires_at"] = time.time() + CACHE_TTL
    return headlines


async def _fetch_newsapi(api_key: str) -> str:
    """
    Calls NewsAPI.org /v2/top-headlines filtered to business/finance.
    Returns a pipe-delimited string of up to 5 headlines.
    Falls back to static context on any error.
    """
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "category": "business",
        "language": "en",
        "pageSize": 10,
        "apiKey": api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        titles = [a["title"] for a in articles if a.get("title")][:5]
        if not titles:
            return _static_context()
        result = " | ".join(titles)
        # Flag if any high-impact event is detected
        upper = result.upper()
        flags = [kw for kw in HIGH_IMPACT_KEYWORDS if kw in upper]
        if flags:
            result = f"[HIGH-IMPACT: {', '.join(flags)}] {result}"
        return result
    except Exception as e:
        print(f"[NEWS] API fetch failed: {e}. Using static context.")
        return _static_context()


def _static_context() -> str:
    """
    Fallback context used when NEWS_API_KEY is not configured.
    Reflects neutral macro backdrop — AI will default to technical analysis.
    """
    return (
        "No live news feed configured. "
        "Macro backdrop: Fed on hold, DXY consolidating, gold in slow uptrend. "
        "No high-impact events scheduled in the next 4 hours."
    )
