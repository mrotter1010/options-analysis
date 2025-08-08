#!/usr/bin/env python3
import os, requests, pandas as pd, pathlib
from schwab import auth
from tqdm import tqdm
from dotenv import load_dotenv
import nest_asyncio
from authlib.integrations.base_client.errors import OAuthError
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

nest_asyncio.apply()

# ————— Load environment variables —————
env_path = pathlib.Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY     = os.getenv("SCHWAB_API_KEY")
APP_SECRET  = os.getenv("SCHWAB_APP_SECRET")
REDIRECT_URI= os.getenv("SCHWAB_REDIRECT_URI")
TOKEN_PATH  = os.getenv("SCHWAB_TOKEN_PATH", "token.json")

# ————— Helper to auto-refresh token —————
def get_client(api_key, app_secret, redirect_uri, token_path):
    try:
        return auth.easy_client(api_key, app_secret, redirect_uri, token_path)
    except ValueError as ve:
        if "token format has changed" in str(ve):
            print("Detected old token format—deleting and retrying.")
            try:
                os.remove(token_path)
            except OSError:
                pass
            return auth.easy_client(api_key, app_secret, redirect_uri, token_path)
        raise
    except OAuthError as oe:
        if "AuthorizationCode has expired" in str(oe):
            print("⚠️  Auth code expired—deleting token.json and retrying.")
            try:
                os.remove(token_path)
            except OSError:
                pass
            return auth.easy_client(api_key, app_secret, redirect_uri, token_path)
        raise

def fetch_daily_closes(client, symbol, start_datetime, end_datetime):
    resp = client.get_price_history_every_day(
        symbol,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        need_extended_hours_data=False,
        need_previous_close=False
    )
    resp.raise_for_status()
    data = resp.json().get("candles", [])
    if not data:
        raise RuntimeError(f"No data for {symbol}")
    df = pd.DataFrame(data)
    # convert epoch ms to datetime and index by it
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df.set_index("datetime", inplace=True)
    # rename the close column to symbol
    return df["close"].rename(symbol)

def fetch_all_closes(client, symbols, start_datetime, end_datetime, max_workers=10):
    """
    Pulls daily close prices for all symbols in parallel,
    returns a DataFrame indexed by date, columns=tickers.
    """
    def _one(sym):
        resp = client.get_price_history_every_day(
            sym,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            need_extended_hours_data=False,
            need_previous_close=False
        )
        resp.raise_for_status()
        data = resp.json().get("candles", [])
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df["close"].rename(sym)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = { ex.submit(_one, s): s for s in symbols }
        series = []
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching prices"):
            sym = futures[fut]
            try:
                series.append(fut.result())
            except Exception as e:
                print(f"⚠️  {sym} failed: {e}")
    return pd.concat(series, axis=1, join="outer").sort_index()

def analyze_ticker(client, closes: pd.Series):
    """
    For one ticker series, find all 4-day drops >10%, then for each:
      - fetch 30-day put chain,
      - if implied vol > 0.60, record a trial,
      - check stock price 30 days later vs strike for success.
    Returns (trials, successes).
    """
    trials = successes = 0
    s = closes.dropna()
    # % change over previous 4 days
    pct4 = s.pct_change(4)
    drop_dates = pct4[pct4 <= -0.10].index

    for dt in drop_dates:
        price_on_dt = s.loc[dt]
        exp_target = (dt + timedelta(days=30)).date()

        # fetch PUT chain between dt and dt+30d
        resp = client.get_option_chain(
            closes.name,
            contract_type=client.Options.ContractType.PUT,
            from_date=dt.date(),
            to_date=exp_target
        )
        chain = resp.json()
        put_map = chain.get("putExpDateMap", {})
        if not put_map:
            continue

        # pick expiration nearest to 30 days out
        def parse_key(k):
            return datetime.fromisoformat(k.split(":")[0]).date()
        exp_keys = list(put_map)
        best_key = min(exp_keys, key=lambda k: abs((parse_key(k) - exp_target).days))
        strikes_map = put_map[best_key]

        # pick ATM strike nearest to price
        strikes = [float(k) for k in strikes_map]
        strike = min(strikes, key=lambda x: abs(x - price_on_dt))
        opt = strikes_map[str(strike)][0]

        iv = opt.get("volatility")  # Schwab returns implied vol here :contentReference[oaicite:0]{index=0}
        if iv is None or iv <= 0.60:
            continue

        # this is a valid trial
        trials += 1

        # find first available close on/after expiration
        future_prices = s[s.index.date >= parse_key(best_key)]
        if future_prices.empty:
            continue
        price_later = future_prices.iloc[0]
        if price_later > strike:
            successes += 1

    return trials, successes

def main():
    client   = get_client(API_KEY, APP_SECRET, REDIRECT_URI, TOKEN_PATH)
    print("✅ Authenticated")

    # 30 large-cap tickers
    large_caps = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK.B", "JPM", "JNJ",
        "V", "PG", "UNH", "MA", "HD", "DIS", "BAC", "XOM", "PFE", "VZ",
        "CVX", "INTC", "NFLX", "T", "MRK", "ABT", "KO", "WMT", "CSCO", "ORCL"
    ]

    # 30 mid-cap tickers (approximate)
    mid_caps = [
        "APTV", "ARNC", "BBY", "BXP", "CAG", "CF", "CINF", "CTAS", "DHI", "DLTR",
        "EBAY", "FAST", "FL", "GRMN", "GPS", "HOG", "JBT", "KMX", "LEG", "LYB",
        "NLSN", "RPM", "SIG", "SNA", "SWN", "TECK", "TTWO", "VFC", "WAB", "WYNN"
    ]

    # 30 small-cap tickers (approximate)
    small_caps = [
        "AKRO", "ALRM", "AMCX", "ANIK", "BANX", "BOOM", "BDSI", "BOX", "CBRL", "CLFD",
        "DENN", "DNUT", "FIZZ", "GOGO", "HAE", "JBSS", "LOGI", "MTDR", "NATI", "OPK",
        "QDEL", "RUTH", "SIMO", "SRPT", "TDOC", "TREE", "TRUP", "VUZI", "WIX", "YETI"
    ]

    # your 90-ticker list here
    tickers = large_caps + mid_caps + small_caps

    # get combined closes
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=365*2)
    closes_df = fetch_all_closes(client, tickers, start_dt, end_dt)

    # Identify which tickers actually made it into the DataFrame
    available = [t for t in tickers if t in closes_df.columns]
    missing   = set(tickers) - set(available)
    if missing:
        print(f"⚠️  No price data for: {', '.join(sorted(missing))}")

    # run analysis per ticker
    # results = []
    # for symbol in tqdm(available, desc="Analyzing tickers"):
    #     series = closes_df[symbol]
    #     t, s = analyze_ticker(client, series)
    #     rate = s / t if t else float("nan")
    #     results.append({
    #         "ticker": symbol,
    #         "trials": t,
    #         "successes": s,
    #         "success_rate": rate
    #     })

    # # final DataFrame
    # results_df = pd.DataFrame(results).set_index("ticker")
    # print(results_df)

    # # save for downstream work
    # results_df.to_csv("put_sell_backtest_results.csv")
    # print("✅ Results saved to put_sell_backtest_results.csv")

    results = []
    for symbol in tqdm(available, desc="Counting 4-day drops"):
        series = closes_df[symbol].dropna()
        if len(series) < 5:
            print(f"– skipping {symbol}, only {len(series)} data points")
            continue

        # compute 4-day % change
        pct4 = series.pct_change(4)

        # count drops of 10% or more
        drop_count = int((pct4 <= -0.10).sum())

        results.append({
            "ticker": symbol,
            "drop_signals": drop_count
        })

    # build and inspect the results table
    results_df = pd.DataFrame(results).set_index("ticker")
    print(results_df)

    # optionally save
    results_df.to_csv("drop_signals.csv")
    print("✅ Saved drop-signal counts to drop_signals.csv")

if __name__ == "__main__":
    main()