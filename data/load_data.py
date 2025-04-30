import pandas as pd
import numpy as np
import time
from pathlib import Path
import multiprocessing as mp
from joblib import Parallel, delayed
import config

def fast_read(path, date_col_name, n_jobs=mp.cpu_count()):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8")
    col = df[date_col_name].to_numpy()
    idx_chunks = np.array_split(np.arange(len(col)), n_jobs)

    def _parse(idx_slice):
        return pd.Series(
            pd.to_datetime(col[idx_slice], errors="coerce", cache=True),
            index=idx_slice
        )

    parsed_parts = Parallel(n_jobs=n_jobs)(delayed(_parse)(slc) for slc in idx_chunks)
    df["date"] = pd.concat(parsed_parts).sort_index()
    return df.drop(columns=[date_col_name])

def load_financial_data():
    btc_df = fast_read(config.BASE_DIR / "Bitcoin Historical Data3.csv", "Date")
    bnb_df = fast_read(config.BASE_DIR / "BNB-USD.csv", "Date")
    return btc_df, bnb_df

def load_twitter_data(path: Path, source: str) -> pd.DataFrame:
    t0 = time.time()
    hdr = pd.read_csv(path, nrows=0)
    date_col = next(c for c in hdr.columns if any(tok in c.lower() for tok in ("date", "time", "created")))
    text_col = next(c for c in hdr.columns if any(tok in c.lower() for tok in ("text", "content", "tweet")))
    usecols = [date_col, text_col]

    def _tidy(df):
        df = df.rename(columns={text_col: "text"})
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["source"] = source
        return df.drop(columns=[date_col]).dropna(subset=["date", "text"])

    try:
        df = pd.read_csv(path, usecols=usecols, engine="pyarrow", dtype={text_col: "string"}, parse_dates=[date_col], memory_map=True)
    except Exception:
        try:
            df = pd.read_csv(path, usecols=usecols, engine="c", on_bad_lines="skip", low_memory=False, dtype={text_col: "string"}, parse_dates=[date_col])
        except Exception:
            chunks = pd.read_csv(path, usecols=usecols, engine="python", chunksize=250_000, on_bad_lines="skip", dtype={text_col: "string"})
            df = pd.concat([chunk.assign(date=pd.to_datetime(chunk[date_col], errors="coerce")) for chunk in chunks], ignore_index=True)

    return _tidy(df)

def load_twitter_datasets():
    tweets_2021 = load_twitter_data(config.BASE_DIR / "archive (1)/bitcoin-tweets-2021.csv", "tweets_2021")
    tweets_2022 = load_twitter_data(config.BASE_DIR / "archive (1)/bitcoin-tweets-2022.csv", "tweets_2022")
    crypto_tweets = load_twitter_data(config.BASE_DIR / "crypto_10k_tweets_(2021_2022Nov).csv/crypto_10k_tweets_(2021_2022Nov).csv", "crypto_tweets")
    return pd.concat([tweets_2021, tweets_2022, crypto_tweets], ignore_index=True)
