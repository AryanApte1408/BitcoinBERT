from data.load_data import load_financial_data, load_twitter_datasets
from data.preprocess_data import preprocess_tweets
from embed.embedder import embed_tweets
from modeling.dataset import prepare_datasets
from modeling.train import train_model
from modeling.evaluation import evaluate_model
from simulation.portfolio_sim import run_backtests
from prompts.evaluate_prompts import evaluate_on_prompts

def main():
    btc_df, bnb_df = load_financial_data()
    tweets_df = load_twitter_datasets()
    tweets_df = preprocess_tweets(tweets_df, btc_df)

    embed_tweets(tweets_df)

    datasets = prepare_datasets(tweets_df, btc_df)

    model = train_model(datasets["train"], datasets["validation"])

    evaluate_model(model, None, datasets["test"])

    run_backtests(model, None, datasets["test"])

    evaluate_on_prompts(model, None)

if __name__ == "__main__":
    main()
