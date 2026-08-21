import httpx
import pandas as pd

from trading_desk.data.fundamentals import dollar_volume_proxy_weights, fetch_market_cap, fetch_market_caps


def test_fetch_market_cap_converts_millions_to_dollars():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["symbol"] == "AAPL"
        return httpx.Response(200, json={"marketCapitalization": 3_500_000.0})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cap = fetch_market_cap("AAPL", "fake-key", client)
    assert cap == 3_500_000.0 * 1_000_000


def test_fetch_market_cap_defaults_to_zero_when_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cap = fetch_market_cap("XYZ", "fake-key", client)
    assert cap == 0.0


def test_fetch_market_caps_covers_every_symbol():
    def handler(request: httpx.Request) -> httpx.Response:
        symbol = request.url.params["symbol"]
        return httpx.Response(200, json={"marketCapitalization": 1000.0 if symbol == "AAPL" else 500.0})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    caps = fetch_market_caps(["AAPL", "MSFT"], "fake-key", client)
    assert caps == {"AAPL": 1_000_000_000.0, "MSFT": 500_000_000.0}


def test_dollar_volume_proxy_weights():
    price_panel = pd.DataFrame({"XLK": [100.0] * 25, "XLF": [40.0] * 25})
    volume_panel = pd.DataFrame({"XLK": [1_000_000] * 25, "XLF": [2_000_000] * 25})

    weights = dollar_volume_proxy_weights(price_panel, volume_panel, lookback=20)

    assert weights["XLK"] == 100.0 * 1_000_000
    assert weights["XLF"] == 40.0 * 2_000_000
