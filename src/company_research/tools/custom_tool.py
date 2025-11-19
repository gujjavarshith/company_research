from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Type

import pandas as pd
import plotly.graph_objects as go
import requests
import wikipedia
import wikipediaapi
from plotly.subplots import make_subplots
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, field_validator


def _request_json(
    url: str,
    params: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to fetch data from {url}. Details: {exc}"
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            f"Received a non-JSON response from {url}. Details: {exc}"
        ) from exc


class SerpApiToolInput(BaseModel):
    query: str = Field(..., description="Search query to run on Google via SerpAPI.")
    num_results: int = Field(
        5, ge=1, le=10, description="Maximum number of organic results to include."
    )
    gl: Optional[str] = Field(
        None, description="Geographic location code (e.g. 'us', 'in')."
    )
    hl: str = Field(
        "en", description="Interface language for the search results (e.g. 'en')."
    )


class SerpApiTool(BaseTool):
    name: str = "serp_api_tool"
    description: str = (
        "Runs a Google search through SerpAPI and returns the top organic results, "
        "including titles, links, and snippets."
    )
    args_schema: Type[BaseModel] = SerpApiToolInput

    def _run(self, query: str, num_results: int = 5, gl: str | None = None, hl: str = "en") -> str:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "SERPAPI_API_KEY is not set. Please add it to your environment or .env file."
            )

        params: Dict[str, Any] = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": num_results,
            "hl": hl,
        }
        if gl:
            params["gl"] = gl

        data = _request_json("https://serpapi.com/search.json", params)
        results = data.get("organic_results", [])[:num_results]
        if not results:
            return f"No organic Google results were returned for '{query}'."

        lines = [f"Top {len(results)} Google results for '{query}':"]
        for idx, result in enumerate(results, start=1):
            title = result.get("title", "Untitled result")
            link = result.get("link", "No URL provided")
            snippet = (result.get("snippet") or "").strip()
            lines.append(f"{idx}. {title} — {link}")
            if snippet:
                lines.append(f"   {snippet}")
        return "\n".join(lines)


class WikipediaToolInput(BaseModel):
    topic: str = Field(..., description="Topic or entity to look up on Wikipedia.")
    max_sentences: int = Field(
        5, ge=1, le=10, description="Maximum number of summary sentences to include."
    )


class WikipediaTool(BaseTool):
    name: str = "wikipedia_tool"
    description: str = (
        "Fetches structured information from Wikipedia, returning a concise summary "
        "and the canonical article URL."
    )
    args_schema: Type[BaseModel] = WikipediaToolInput

    def _run(self, topic: str, max_sentences: int = 5) -> str:
        wikipedia.set_lang("en")
        wiki_api = wikipediaapi.Wikipedia(
            language="en",
            user_agent="CompanyResearchAgent/1.0 (company_research@example.com)",
        )

        page = wiki_api.page(topic)
        if page.exists():
            summary_sentences = page.summary.split(". ")
            trimmed = ". ".join(summary_sentences[:max_sentences]).strip()
            if trimmed and not trimmed.endswith("."):
                trimmed += "."
            return "\n".join(
                [
                    f"Wikipedia summary for '{page.title}':",
                    trimmed or "No summary available.",
                    f"URL: {page.fullurl}",
                ]
            )

        related = wikipedia.search(topic, results=5)
        if not related:
            return f"No Wikipedia article or related pages were found for '{topic}'."

        lines = [
            f"No exact Wikipedia page found for '{topic}'. Related suggestions:",
        ]
        for idx, title in enumerate(related, start=1):
            lines.append(f"{idx}. {title}")
        return "\n".join(lines)


class YahooFinanceToolInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol to fetch data for (e.g. AAPL).")


class YahooFinanceTool(BaseTool):
    name: str = "yahoo_finance_tool"
    description: str = (
        "Fetches quote summary information for a ticker symbol directly from Yahoo Finance."
    )
    args_schema: Type[BaseModel] = YahooFinanceToolInput

    def _run(self, symbol: str) -> str:
        modules = "price,summaryDetail,financialData,defaultKeyStatistics"
        url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        try:
            data = _request_json(url, params={"modules": modules})
        except RuntimeError as exc:
            if "429" in str(exc):
                return (
                    "Yahoo Finance is rate limiting requests right now. "
                    "Please wait a moment and try again."
                )
            raise
        result = data.get("quoteSummary", {}).get("result")
        if not result:
            return f"Yahoo Finance did not return data for '{symbol}'."

        payload = result[0]
        price = payload.get("price", {})
        summary = payload.get("summaryDetail", {})
        financial = payload.get("financialData", {})

        name = price.get("longName") or price.get("shortName") or symbol.upper()
        current_price = price.get("regularMarketPrice", {}).get("fmt", "N/A")
        currency = price.get("currency") or ""
        market_cap = summary.get("marketCap", {}).get("fmt", "N/A")
        pe_ratio = summary.get("trailingPE", {}).get("fmt", "N/A")
        forward_pe = summary.get("forwardPE", {}).get("fmt", "N/A")
        revenue = financial.get("totalRevenue", {}).get("fmt", "N/A")
        profit_margins = summary.get("profitMargins", {}).get("fmt", "N/A")
        target_mean_price = financial.get("targetMeanPrice", {}).get("fmt", "N/A")

        return "\n".join(
            [
                f"Yahoo Finance snapshot for {name} ({symbol.upper()}):",
                f"- Price: {current_price} {currency}".strip(),
                f"- Market Cap: {market_cap}",
                f"- Trailing P/E: {pe_ratio} | Forward P/E: {forward_pe}",
                f"- Total Revenue: {revenue}",
                f"- Profit Margins: {profit_margins}",
                f"- Analyst Target (mean): {target_mean_price}",
            ]
        )


class GoogleTrendsToolInput(BaseModel):
    keyword: str = Field(..., description="Search keyword to analyze in Google Trends.")
    geo: str = Field(
        "US",
        description="Geographic region code (ISO-3166). Use 'GLOBAL' for worldwide data.",
    )
    trailing_days: int = Field(
        365,
        ge=7,
        le=1095,
        description="Number of trailing days to include in the analysis.",
    )


class GoogleTrendsTool(BaseTool):
    name: str = "google_trends_tool"
    description: str = (
        "Retrieves historical interest-over-time data for a keyword using SerpAPI's "
        "Google Trends endpoint and summarizes the trend."
    )
    args_schema: Type[BaseModel] = GoogleTrendsToolInput

    def _run(self, keyword: str, geo: str = "US", trailing_days: int = 365) -> str:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "SERPAPI_API_KEY is required to call the Google Trends tool."
            )

        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=trailing_days)
        params: Dict[str, Any] = {
            "engine": "google_trends",
            "data_type": "TIMESERIES",
            "q": keyword,
            "time_range": f"{start_date} {end_date}",
            "hl": "en",
            "api_key": api_key,
        }
        if geo and geo.upper() != "GLOBAL":
            params["geo"] = geo.upper()

        data = _request_json("https://serpapi.com/search.json", params)
        interest = data.get("interest_over_time", {})
        points = interest.get("timeline_data", [])
        if not points:
            return f"Google Trends did not return timeline data for '{keyword}'."

        values = [point.get("values", [{}])[0].get("value", 0) for point in points]
        avg_interest = sum(values) / len(values)
        peak_index = max(range(len(values)), key=lambda idx: values[idx])
        peak_value = values[peak_index]
        peak_time = points[peak_index].get("time")

        latest_point = points[-1]
        latest_value = latest_point.get("values", [{}])[0].get("value", 0)
        latest_time = latest_point.get("time")

        return "\n".join(
            [
                f"Google Trends analysis for '{keyword}' ({geo.upper()}):",
                f"- Period: {start_date.isoformat()} → {end_date.isoformat()}",
                f"- Average interest: {avg_interest:.1f}",
                f"- Peak interest: {peak_value} on {peak_time}",
                f"- Latest interest ({latest_time}): {latest_value}",
            ]
        )


class NewsApiToolInput(BaseModel):
    query: str = Field(..., description="News search query or boolean expression.")
    language: str = Field("en", description="Two-letter language code.")
    days_back: int = Field(
        7, ge=1, le=30, description="How many days back to search for articles."
    )
    page_size: int = Field(
        10, ge=1, le=50, description="Maximum number of articles to include."
    )
    sort_by: str = Field(
        "relevancy",
        description="Sort order: relevancy, popularity, or publishedAt.",
    )


class NewsApiTool(BaseTool):
    name: str = "news_api_tool"
    description: str = (
        "Queries the NewsAPI.org Everything endpoint to surface recent media coverage "
        "about the specified topic."
    )
    args_schema: Type[BaseModel] = NewsApiToolInput

    def _run(
        self,
        query: str,
        language: str = "en",
        days_back: int = 7,
        page_size: int = 10,
        sort_by: str = "relevancy",
    ) -> str:
        api_key = os.getenv("NEWSAPI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NEWSAPI_API_KEY is not set. Obtain a key from https://newsapi.org."
            )

        from_date = (datetime.utcnow() - timedelta(days=days_back)).date().isoformat()
        params = {
            "q": query,
            "language": language,
            "sortBy": sort_by,
            "pageSize": page_size,
            "from": from_date,
            "apiKey": api_key,
        }

        data = _request_json("https://newsapi.org/v2/everything", params)
        articles = data.get("articles", [])
        if not articles:
            return f"No recent news articles found for '{query}'."

        lines = [
            f"Top {min(len(articles), page_size)} news articles for '{query}' "
            f"(since {from_date}):"
        ]
        for idx, article in enumerate(articles[:page_size], start=1):
            title = article.get("title", "Untitled article")
            source = article.get("source", {}).get("name", "Unknown source")
            url = article.get("url", "No URL provided")
            published = article.get("publishedAt", "Unknown date")
            lines.append(f"{idx}. {title} — {source} ({published})")
            lines.append(f"   {url}")
        return "\n".join(lines)


class StockChartToolInput(BaseModel):
    ticker: Optional[str] = Field(
        None,
        description="Ticker symbol to chart (e.g. AAPL). Optional if company is provided.",
    )
    company: Optional[str] = Field(
        None,
        description="Company name to search via TwelveData if ticker is unknown.",
    )
    timeframe: str = Field(
        "1D",
        description="One of 1D (1 day), 1W (1 week), or 1M (1 month) describing the analysis window.",
    )

    @field_validator("ticker")
    @classmethod
    def at_least_one_identifier(cls, v, info):
        company = info.data.get("company")
        if not v and not company:
            raise ValueError("Provide either a ticker symbol or a company name.")
        return v


class StockChartTool(BaseTool):
    name: str = "stock_chart_tool"
    description: str = (
        "Generates candlestick charts, moving averages, and a markdown report for a "
        "ticker using TwelveData market data. Saves outputs under stock_reports/."
    )
    args_schema: Type[BaseModel] = StockChartToolInput
    TIMEFRAMES: ClassVar[Dict[str, str]] = {
        "1D": "1day",
        "1W": "1week",
        "1M": "1month",
    }

    _output_dir: Path = PrivateAttr()

    def __init__(self) -> None:
        super().__init__()
        output_dir = os.getenv("STOCK_CHART_OUTPUT_DIR", "stock_reports")
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _run(
        self,
        ticker: Optional[str] = None,
        company: Optional[str] = None,
        timeframe: str = "1D",
    ) -> str:
        try:
            api_key = os.getenv("TWELVEDATA_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "TWELVEDATA_API_KEY is not set. Obtain one from https://twelvedata.com "
                    "and add it to your environment."
                )

            if not ticker and not company:
                raise ValueError("Either 'ticker' or 'company' must be provided.")

            interval = self.TIMEFRAMES.get(timeframe.upper())
            if not interval:
                valid = ", ".join(self.TIMEFRAMES.keys())
                raise ValueError(f"Unsupported timeframe '{timeframe}'. Choose from {valid}.")

            symbol = ticker.upper() if ticker else self._search_symbol(company, api_key)
            candles = self._get_candles(symbol, interval, api_key)
            
            if not candles or len(candles) == 0:
                raise RuntimeError(f"No data returned for {symbol} ({timeframe}).")
            
            df = self._prepare_dataframe(candles)
            
            if len(df) == 0:
                raise RuntimeError(f"Failed to process data for {symbol}.")
            
            fig = self._create_chart(df, symbol, timeframe)
            chart_path, chart_filename = self._save_chart(fig, symbol, timeframe)
            report_path, summary = self._create_markdown_report(
                df, symbol, timeframe, chart_filename
            )

            return "\n".join(
                [
                    f"Stock analysis generated for {symbol} ({timeframe}).",
                    f"Chart file: {chart_path.name}",
                    f"Report file: {report_path.name}",
                    summary,
                ]
            )
        except Exception as e:
            return f"Error generating stock chart: {str(e)}"

    def _search_symbol(self, company: Optional[str], api_key: str) -> str:
        if not company:
            raise ValueError("A ticker or company name must be provided.")
        try:
            params = {"symbol": company, "apikey": api_key}
            data = _request_json("https://api.twelvedata.com/symbol_search", params)
            
            if data.get("status") == "error":
                error_msg = data.get("message", "Unknown error")
                raise RuntimeError(f"TwelveData API error: {error_msg}")
            
            matches = data.get("data") or []
            if not matches:
                raise RuntimeError(f"Could not find a ticker for '{company}'. Try using the ticker symbol directly.")
            return matches[0]["symbol"]
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to search for symbol '{company}': {str(e)}")

    def _get_candles(self, symbol: str, interval: str, api_key: str) -> list[Dict[str, str]]:
        try:
            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": 500,
                "apikey": api_key,
            }
            data = _request_json("https://api.twelvedata.com/time_series", params)
            
            if data.get("status") == "error":
                error_msg = data.get("message", "Unknown error")
                raise RuntimeError(f"TwelveData API error for {symbol}: {error_msg}")
            
            candles = data.get("values")
            if not candles:
                raise RuntimeError(
                    f"No candlestick data returned for {symbol} ({interval}). "
                    f"The symbol may not be available or the interval may be invalid."
                )
            return candles
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to fetch candlestick data for {symbol}: {str(e)}")

    @staticmethod
    def _prepare_dataframe(values: list[Dict[str, str]]) -> pd.DataFrame:
        df = pd.DataFrame(values)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Handle datetime column - TwelveData uses 'datetime' or 'time'
        if "datetime" not in df.columns and "time" in df.columns:
            df["datetime"] = df["time"]
        elif "datetime" not in df.columns:
            # Create datetime index if missing
            df["datetime"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="D")
        
        # Convert datetime to string for display if needed
        if df["datetime"].dtype != "object":
            df["datetime"] = df["datetime"].astype(str)
        
        return df.iloc[::-1].reset_index(drop=True)

    @staticmethod
    def _create_chart(df: pd.DataFrame, symbol: str, timeframe: str):
        # Calculate moving averages only if we have enough data points
        ma20 = df["close"].rolling(window=min(20, len(df))).mean() if len(df) > 0 else pd.Series()
        ma50 = df["close"].rolling(window=min(50, len(df))).mean() if len(df) >= 5 else pd.Series()
        ma200 = df["close"].rolling(window=min(200, len(df))).mean() if len(df) >= 10 else pd.Series()

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} - {timeframe} Chart", "Volume"),
        )

        fig.add_trace(
            go.Candlestick(
                x=df["datetime"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=symbol,
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=1,
            col=1,
        )

        # Only add moving averages if they have valid data
        if len(ma20) > 0 and not ma20.isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=ma20,
                    mode="lines",
                    name="MA 20",
                    line=dict(color="#ff9800", width=1),
                ),
                row=1,
                col=1,
            )
        if len(ma50) > 0 and not ma50.isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=ma50,
                    mode="lines",
                    name="MA 50",
                    line=dict(color="#2196f3", width=1),
                ),
                row=1,
                col=1,
            )
        if len(ma200) > 0 and not ma200.isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=ma200,
                    mode="lines",
                    name="MA 200",
                    line=dict(color="#9c27b0", width=1),
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Bar(
                x=df["datetime"],
                y=df["volume"],
                name="Volume",
                marker_color="#78909c",
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title=f"{symbol} Stock Analysis - {timeframe}",
            xaxis_rangeslider_visible=False,
            height=800,
            template="plotly_dark",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        return fig

    def _save_chart(self, fig, symbol: str, timeframe: str) -> tuple[Path, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{timeframe}_{timestamp}.png"
        filepath = self._output_dir / filename
        try:
            fig.write_image(str(filepath), width=1400, height=800, scale=2)
        except Exception:
            html_name = filename.replace(".png", ".html")
            filepath = self._output_dir / html_name
            fig.write_html(str(filepath))
            filename = html_name
        return filepath, filename

    def _create_markdown_report(
        self, df: pd.DataFrame, symbol: str, timeframe: str, chart_filename: str
    ) -> tuple[Path, str]:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        latest = df.iloc[-1]
        first = df.iloc[0]
        change = latest["close"] - first["close"]
        change_pct = (change / first["close"]) * 100 if first["close"] else 0
        high_price = df["high"].max()
        low_price = df["low"].min()
        avg_volume = df["volume"].mean()

        chart_md = (
            f"![{symbol} Chart]({chart_filename})"
            if chart_filename.endswith(".png")
            else f"[View Interactive Chart]({chart_filename})"
        )

        ma20_val = df["close"].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
        ma50_val = df["close"].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
        ma200_val = (
            df["close"].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
        )

        md_lines = [
            f"# Stock Analysis Report: {symbol}",
            f"**Generated:** {timestamp}  ",
            f"**Timeframe:** {timeframe}  ",
            f"**Data Points:** {len(df)}",
            "",
            "---",
            "",
            "## Chart",
            "",
            chart_md,
            "",
            "---",
            "",
            "## Summary Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **First Close** | ${first['close']:.2f} |",
            f"| **Latest Close** | ${latest['close']:.2f} |",
            f"| **Change** | ${change:.2f} ({change_pct:+.2f}%) |",
            f"| **Period High** | ${high_price:.2f} |",
            f"| **Period Low** | ${low_price:.2f} |",
            f"| **Average Volume** | {avg_volume:,.0f} |",
            "",
            "---",
            "",
            "## Recent Data (Last 5 rows)",
            "",
            "| Date | Open | High | Low | Close | Volume |",
            "|------|------|------|-----|-------|--------|",
        ]
        for _, row in df.tail(5).iterrows():
            md_lines.append(
                f"| {row['datetime']} | ${row['open']:.2f} | ${row['high']:.2f} | "
                f"${row['low']:.2f} | ${row['close']:.2f} | {row['volume']:,.0f} |"
            )

        ma20_str = f"${ma20_val:.2f}" if pd.notna(ma20_val) else "N/A"
        ma50_str = f"${ma50_val:.2f}" if pd.notna(ma50_val) else "N/A"
        ma200_str = f"${ma200_val:.2f}" if pd.notna(ma200_val) else "N/A"

        md_lines.extend(
            [
                "",
                "---",
                "",
                "## Technical Indicators",
                "",
                f"- **MA 20:** {ma20_str}",
                f"- **MA 50:** {ma50_str}",
                f"- **MA 200:** {ma200_str}",
                "",
                "---",
                "",
                "*Report generated by Stock Chart Tool*",
            ]
        )

        md_filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        md_path = self._output_dir / md_filename
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        summary = (
            f"Latest close ${latest['close']:.2f}, "
            f"{change:+.2f} ({change_pct:+.2f}%) versus start of period. "
            f"Chart: {chart_filename}, report: {md_filename}."
        )
        return md_path, summary


__all__ = [
    "SerpApiTool",
    "WikipediaTool",
    "YahooFinanceTool",
    "GoogleTrendsTool",
    "NewsApiTool",
    "StockChartTool",
]
