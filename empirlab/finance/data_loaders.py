"""Data loaders for A-share (AKShare), FRED macro, and Yahoo Finance."""
from __future__ import annotations
import pandas as pd
from empirlab.utils.data_io import cache_csv


def load_ashare(
    symbol: str,
    start: str = "2015-01-01",
    end: str = None,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """Load A-share daily OHLCV via AKShare with disk caching.

    Parameters
    ----------
    symbol : str    e.g. '000001' (Ping An Bank A)
    start  : str    'YYYY-MM-DD'
    end    : str    'YYYY-MM-DD' (default: today)
    adjust : str    'qfq' forward-adj | 'hfq' backward-adj | '' unadjusted

    Returns
    -------
    DataFrame indexed by date with columns:
        open, close, high, low, volume, amount, pct_change, turnover
    """
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("pip install akshare")

    key = f"ashare_{symbol}_{start}_{end}_{adjust}"

    def _fetch():
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=(start or "").replace("-", ""),
            end_date=(end or "").replace("-", "") or None,
            adjust=adjust,
        )
        df.columns = [
            "date", "open", "close", "high", "low",
            "volume", "amount", "amplitude", "pct_change", "change", "turnover",
        ]
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()

    return cache_csv(key, _fetch)


def load_fred(series_id: str, start: str = "2000-01-01") -> pd.Series:
    """Load a FRED macro series via pandas-datareader.

    Parameters
    ----------
    series_id : str   e.g. 'GDP', 'UNRATE', 'FEDFUNDS', 'CPIAUCSL'
    start     : str   'YYYY-MM-DD'
    """
    try:
        import pandas_datareader.data as web
    except ImportError:
        raise ImportError("pip install pandas-datareader")

    key = f"fred_{series_id}_{start}"

    def _fetch():
        return web.DataReader(series_id, "fred", start=start)

    return cache_csv(key, _fetch)[series_id]


def load_yfinance(
    tickers: str | list,
    start: str = "2015-01-01",
    end: str = None,
) -> pd.DataFrame:
    """Load adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : str or list   e.g. 'SPY' or ['SPY', 'QQQ', 'GLD']
    start   : str
    end     : str (default: today)

    Returns
    -------
    DataFrame of adjusted close prices, columns = tickers.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    if isinstance(tickers, str):
        tickers = [tickers]

    key = f"yf_{'_'.join(tickers)}_{start}_{end}"

    def _fetch():
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
        if len(tickers) == 1:
            return raw[["Close"]].rename(columns={"Close": tickers[0]})
        return raw["Close"]

    return cache_csv(key, _fetch)
