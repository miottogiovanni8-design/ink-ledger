import httpx

from trading_desk.data.news import fetch_alphavantage_sentiment, fetch_finnhub_general_news, fetch_finnhub_headlines


def test_fetch_finnhub_general_news_extracts_headline_field():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "finnhub.io" in str(request.url)
        assert request.url.path.endswith("/news")
        assert request.url.params["category"] == "general"
        return httpx.Response(
            200,
            json=[
                {"headline": "Fed holds rates steady", "url": "https://example.com/fed", "datetime": 1700000000},
                {"headline": "S&P 500 hits new high", "url": "https://example.com/spx", "datetime": 1700000100},
                {"datetime": 1700000200},  # missing headline, should be skipped
            ],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    headlines = fetch_finnhub_general_news("fake-key", client)

    assert headlines == [
        {"headline": "Fed holds rates steady", "url": "https://example.com/fed"},
        {"headline": "S&P 500 hits new high", "url": "https://example.com/spx"},
    ]


def test_fetch_finnhub_headlines_extracts_headline_field():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "finnhub.io" in str(request.url)
        assert request.url.params["symbol"] == "AAPL"
        return httpx.Response(
            200,
            json=[
                {"headline": "Apple beats earnings estimates", "url": "https://example.com/aapl-1", "datetime": 1700000000},
                {"headline": "Apple announces new product", "url": "https://example.com/aapl-2", "datetime": 1700000100},
                {"datetime": 1700000200},  # missing headline, should be skipped
            ],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    headlines = fetch_finnhub_headlines("AAPL", "fake-key", client)

    assert headlines == [
        {"headline": "Apple beats earnings estimates", "url": "https://example.com/aapl-1"},
        {"headline": "Apple announces new product", "url": "https://example.com/aapl-2"},
    ]


def test_fetch_alphavantage_sentiment_finds_matching_ticker():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["tickers"] == "AAPL"
        return httpx.Response(
            200,
            json={
                "feed": [
                    {
                        "ticker_sentiment": [
                            {
                                "ticker": "AAPL",
                                "relevance_score": "0.85",
                                "ticker_sentiment_score": "0.32",
                                "ticker_sentiment_label": "Somewhat-Bullish",
                            },
                            {"ticker": "MSFT", "relevance_score": "0.1", "ticker_sentiment_score": "0.0"},
                        ]
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sentiment = fetch_alphavantage_sentiment("AAPL", "fake-key", client)

    assert sentiment["relevance_score"] == 0.85
    assert sentiment["sentiment_score"] == 0.32
    assert sentiment["sentiment_label"] == "Somewhat-Bullish"


def test_fetch_alphavantage_sentiment_defaults_when_ticker_absent():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"feed": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sentiment = fetch_alphavantage_sentiment("AAPL", "fake-key", client)

    assert sentiment == {"relevance_score": 0.0, "sentiment_score": 0.0, "sentiment_label": "Neutral"}
