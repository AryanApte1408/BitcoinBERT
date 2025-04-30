import pandas as pd
import config

def preprocess_tweets(tweets_df, btc_df):
    tweets_df["date"] = pd.to_datetime(tweets_df["date"], errors="coerce").dt.tz_localize(None)
    tweets_df = tweets_df.dropna(subset=["date", "text"])
    tweets_df = tweets_df.sort_values("date").reset_index(drop=True)

    start, end = pd.to_datetime(btc_df["date"].min()), pd.to_datetime(btc_df["date"].max())
    tweets_df = tweets_df[(tweets_df["date"] >= start) & (tweets_df["date"] <= end)]

    mask_crypto = tweets_df["source"] == "crypto_tweets"
    tweets_df.loc[mask_crypto, "is_crypto"] = tweets_df.loc[mask_crypto, "text"].str.contains("bitcoin", case=False, na=False)
    tweets_df = tweets_df[~mask_crypto | tweets_df["is_crypto"]]

    return tweets_df.reset_index(drop=True)
