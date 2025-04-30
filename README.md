# BitcoinBERT: Sentiment-Driven Portfolio Simulation from Crypto Tweets

## 🧠 Overview
This project builds a fine-tuned BERT model using Bitcoin-related tweets to predict short-term price movements.  
It embeds tweets, labels them using next-day price returns, fine-tunes `bert-base-uncased`, and simulates trading based on the predictions.

## 📁 Project Structure
```
BitcoinBERT/
├── config.py
├── main.py
├── data/
│   ├── load_data.py
│   └── preprocess_data.py
├── embed/
│   └── embedder.py
├── modeling/
│   ├── dataset.py
│   ├── train.py
│   └── evaluation.py
├── simulation/
│   └── portfolio_sim.py
├── prompts/
│   └── evaluate_prompts.py
```

## ⚙️ Requirements
- Python 3.8+
- torch, transformers, datasets
- pandas, numpy, scikit-learn
- chromadb, sentence-transformers
- tqdm, joblib

## 🚀 Running the Project
1. Clone/download the folder.
2. Update paths in `config.py` to point to your dataset.
3. Run:
```bash
python main.py
```

## 📌 Notes
- Only 60000 tweets used due to compute limits.
- Future work includes full-dataset training + live Twitter ingestion.
