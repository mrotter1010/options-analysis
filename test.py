import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List, Union

def filter_large_caps(tickers: List[str],
                      min_market_cap: float = 3e9,
                      pause: float = 1.0,
                      max_retries: int=5
                      ) -> List[str]:
    """
    Returns the subset of `tickers` whose current marketCap ≥ min_market_cap.

    Args:
      tickers: List of ticker symbols (e.g. ['AAPL','MSFT',...])
      min_market_cap: threshold in USD (default 3e9)
      pause: seconds to wait between info calls to be polite

    Returns:
      List of tickers passing the filter.
    """
    large = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            if info.get("marketCap", 0) >= min_market_cap:
                large.append(t)
        except Exception:
            # skip symbols that error out
            pass
        finally:
            # small sleep so you don’t get rate-limited
            from time import sleep; sleep(pause)
    return large


def fetch_closing_prices(tickers: List[str],
                         start: Union[str, datetime],
                         end:   Union[str, datetime]
                         ) -> pd.DataFrame:
    """
    Downloads daily 'Close' prices for `tickers` from Yahoo Finance.

    Args:
      tickers: List of ticker symbols.
      start:   'YYYY-MM-DD' or datetime start date (inclusive).
      end:     'YYYY-MM-DD' or datetime end date (exclusive).

    Returns:
      DataFrame with:
        • index = Date,
        • columns = symbols,
        • values = closing price.
    """
    # yfinance will align into a (Date × symbols) table
    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        threads=True,      # faster if you have many tickers
        group_by="ticker"  # keeps columnm MultiIndex for single ticker
    )
    # if only one ticker, data['Close'] is a Series
    if isinstance(data, pd.Series) or data.columns.nlevels == 1:
        return data["Close"].to_frame()

    # multi-ticker: data columns like ('AAPL','Open'),('AAPL','Close'),...
    closes = pd.concat(
        {t: data[t]["Close"] for t in tickers if t in data},
        axis=1
    )
    closes.index = pd.to_datetime(closes.index)
    return closes


# ─── USAGE EXAMPLE ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Define your universe (e.g. S&P 500 tickers)
    import pandas as pd
    sp500 = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]["Symbol"].tolist()

    # 2. Filter for ≥ $3 billion market cap
    large_sp500 = filter_large_caps(sp500, min_market_cap=3e9)

    # 3. Download closes from 2015-01-01 through today
    df_closes = fetch_closing_prices(
        large_sp500,
        start="2015-01-01",
        end=pd.Timestamp.today().strftime("%Y-%m-%d")
    )

    # 4. Inspect or save
    print(df_closes.tail())
    df_closes.to_csv("yahoo_historical_closes.csv")
