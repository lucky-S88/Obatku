import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class HealthConsultationMatcher:
    def __init__(self, embedding_path="condition_embeddings.npy", data_path="condition_data.pkl"):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Load precomputed embeddings dengan allow_pickle=True
        self.condition_embeddings = np.load(embedding_path, allow_pickle=True)

        # Load condition metadata
        with open(data_path, "rb") as f:
            self.condition_list = pickle.load(f)

    def find_matching_drugs(self, user_input: str, top_k: int = 1):
        input_embedding = self.model.encode([user_input], convert_to_numpy=True)
        similarities = cosine_similarity(input_embedding, self.condition_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            condition_data = self.condition_list[idx]
            results.append(condition_data)
        return results
