from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class Embeddings:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[np.ndarray]:
        return self.model.encode(documents, convert_to_numpy=True).tolist()
