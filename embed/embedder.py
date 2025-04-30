from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import config
from tqdm.auto import tqdm

def embed_tweets(df):
    client = PersistentClient(path=config.DB_PATH, settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection(config.COLLECTION_NAME)
    embedder = SentenceTransformer(config.EMBED_MODEL, device="cuda")

    existing_ids = set(coll.get(limit=coll.count(), include=[])["ids"]) if coll.count() else set()
    df = df.loc[~df.index.astype(str).isin(existing_ids)]

    for start in tqdm(range(0, len(df), config.EMBED_BATCH_SIZE), desc="Embedding"):
        chunk = df.iloc[start:start + config.EMBED_BATCH_SIZE]
        ids = chunk.index.astype(str).tolist()
        texts = chunk["text"].tolist()
        metas = [{"date": d.isoformat(), "source": s} for d, s in zip(chunk["date"], chunk["source"])]
        embs = embedder.encode(texts, batch_size=config.EMBED_BATCH_SIZE, show_progress_bar=False)
        coll.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)
