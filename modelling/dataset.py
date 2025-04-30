import pandas as pd
import numpy as np
import config
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import BertTokenizerFast

def prepare_datasets(tweets_df, btc_df):
    tweets_df = tweets_df.copy()
    tweets_df["date"] = pd.to_datetime(tweets_df["date"], errors="coerce")

    price_col = next(c for c in btc_df.columns if c.lower() in ("close", "price"))
    btc_price = btc_df.loc[:, ["date", price_col]].rename(columns={price_col: "close"})
    btc_price["date"] = pd.to_datetime(btc_price["date"], errors="coerce")
    btc_price["close"] = pd.to_numeric(btc_price["close"].astype(str).str.replace(",", ""), errors="coerce")
    btc_price = btc_price.dropna()

    btc_price["next_close"] = btc_price["close"].shift(-1)
    btc_price["return"] = (btc_price["next_close"] - btc_price["close"]) / btc_price["close"]
    btc_price["target"] = np.where(
        btc_price["return"] > config.THRESHOLD_UP, "buy",
        np.where(btc_price["return"] < config.THRESHOLD_DOWN, "sell", "hold")
    )

    labeled = pd.merge(tweets_df, btc_price.loc[:, ["date", "target"]], on="date", how="inner")

    def split(df, test_months=2):
        max_day = df["date"].max()
        test_from = max_day - pd.DateOffset(months=test_months)
        train_val = df[df["date"] < test_from]
        test = df[df["date"] >= test_from]
        return train_val, test

    train_val, test = split(labeled)
    if test.empty:
        train_val, test = split(labeled, test_months=1)

    val = train_val.sample(frac=0.1, random_state=config.RANDOM_SEED)
    train = train_val.drop(val.index)

    for df in (train, val, test):
        df["label_id"] = df["target"].map(config.LABEL_MAP)

    raw_ds = DatasetDict({
        "train": Dataset.from_pandas(train[["text", "label_id"]]),
        "validation": Dataset.from_pandas(val[["text", "label_id"]]),
        "test": Dataset.from_pandas(test[["text", "label_id"]]),
    }).cast_column("label_id", ClassLabel(names=config.LABELS))

    tokenizer = BertTokenizerFast.from_pretrained(config.BERT_MODEL)

    def tok_fn(b):
        enc = tokenizer(b["text"], truncation=True, padding="max_length", max_length=128)
        enc["labels"] = b["label_id"]
        return enc

    tok_ds = raw_ds.map(tok_fn, batched=True, remove_columns=["text", "label_id"])
    for split in tok_ds:
        if len(tok_ds[split]):
            tok_ds[split].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return tok_ds
