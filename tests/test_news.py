import httpx

from trading_desk.data.news import fetch_alphavantage_sentiment, fetch_finnhub_headlines


def test_fetch_finnhub_headlines_extracts_headline_field():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "finnhub.io" in str(request.url)
        assert request.url.params["symbol"] == "AAPL"
        return httpx.Response(
            200,
            json=[
                {"headline": "Apple beats earnings estimates", "datetime": 1700000000},
                {"headline": "Apple announces new product", "datetime": 1700000100},
                {"datetime": 1700000200},  # missing headline, should be skipped
            ],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    headlines = fetch_finnhub_headlines("AAPL", "fake-key", client)

    assert headlines == ["Apple beats earnings estimates", "Apple announces new product"]


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
