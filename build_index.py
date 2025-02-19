import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


df = pd.read_csv("data/medquad.csv")
qa_pairs = [
    f"Question: {row['question']} Answer: {row['answer']}"
    for _, row in df.dropna(subset=['question', 'answer']).iterrows()
]

embedder = SentenceTransformer("all-mpnet-base-v2")
embeddings = embedder.encode(qa_pairs, convert_to_numpy=True)


d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
faiss.normalize_L2(embeddings)
index.add(embeddings)

faiss.write_index(index, "faiss_index.index")
np.save("qa_pairs.npy", np.array(qa_pairs))

print(f"FAISS index built with {index.ntotal} QA pairs.")
