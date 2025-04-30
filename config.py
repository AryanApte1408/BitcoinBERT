from pathlib import Path

# Paths
BASE_DIR = Path(r"D:\\text_mining_project")
DB_PATH = "chroma_db"
COLLECTION_NAME = "bitcoin_tweets"

# Embedding
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 512

# Training
BERT_MODEL = "bert-base-uncased"
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 3

# Financial thresholds
THRESHOLD_UP = 0.001
THRESHOLD_DOWN = -0.001

# Data Sampling
TARGET_TOTAL_TWEETS = 60000
RANDOM_SEED = 42

# Labels
LABELS = ["sell", "hold", "buy"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}
