from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_EQUITY_WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
DEFAULT_CRYPTO_WATCHLIST = ["BTC/USD", "ETH/USD"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True

    anthropic_api_key: str = ""

    finnhub_api_key: str = ""
    alphavantage_api_key: str = ""

    resend_api_key: str = ""
    email_from: str = ""
    email_to: str = ""

    daily_budget_eur: float = 100.0
    risk_pct_per_trade: float = 0.05
    max_position_pct_of_budget: float = 0.15
    max_concurrent_positions: int = 5
    daily_loss_circuit_breaker_pct: float = 1.0
    max_drawdown_circuit_breaker_pct: float = 0.18
    stop_loss_atr_multiplier: float = 1.5
    take_profit_rr_ratio: float = 1.75

    db_path: str = "data/trading_desk.sqlite"
    snapshot_path: str = "data/dashboard_snapshot.json"

    equity_watchlist: List[str] = DEFAULT_EQUITY_WATCHLIST
    crypto_watchlist: List[str] = DEFAULT_CRYPTO_WATCHLIST


settings = Settings()
