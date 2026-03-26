"""
RAG-based Recommendation System for SumerTrip
Simple: embed user query string -> find nearest events/trips
Lazy loading to prevent startup timeout
"""

from typing import List, Dict, Optional
from config import config

# Lazy loaded
_embedding_model = None
_faiss = None
_recommendation_rag = None


def _get_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedding_model


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss

        _faiss = faiss
    return _faiss


class EventRecommendationRAG:
    """Simple RAG: embed query string, find nearest matches"""

    def __init__(self):
        self.dimension = 384
        self.index = None
        self.all_items = []
        self._model_loaded = False

    def _get_model(self):
        return _get_model()

    def _create_item_text(self, item: Dict, item_type: str) -> str:
        parts = [
            item.get("title", ""),
            item.get("titleEn", ""),
            item.get("location", ""),
            item.get("category", ""),
            item.get("description", ""),
        ]
        if "highlights" in item:
            parts.extend(item["highlights"])
        return " ".join(parts)

    def load_data(self, trips: List[Dict], events: List[Dict]) -> int:
        """Load trips and events into vector index"""
        faiss = _get_faiss()
        model = self._get_model()

        self.all_items = []

        for trip in trips:
            self.all_items.append(
                {
                    "type": "trip",
                    "data": trip,
                    "text": self._create_item_text(trip, "trip"),
                }
            )

        for event in events:
            self.all_items.append(
                {
                    "type": "event",
                    "data": event,
                    "text": self._create_item_text(event, "event"),
                }
            )

        if not self.all_items:
            return 0

        texts = [item["text"] for item in self.all_items]
        embeddings = model.encode(texts, convert_to_numpy=True)

        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype("float32"))
        self._model_loaded = True

        return len(self.all_items)

    def recommend(
        self, query: str, top_k: int = 5, item_type: Optional[str] = None
    ) -> List[Dict]:
        """Find nearest matches for query string"""
        if not self.index or not self.all_items:
            return []

        faiss = _get_faiss()
        model = self._get_model()

        query_embedding = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        search_k = top_k * 3 if item_type else top_k
        scores, indices = self.index.search(
            query_embedding.astype("float32"), min(search_k, len(self.all_items))
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = self.all_items[idx]

            if item_type and item["type"] != item_type:
                continue

            results.append(
                {
                    "id": item["data"].get("id"),
                    "type": item["type"],
                    "score": float(score),
                    "data": item["data"],
                }
            )

            if len(results) >= top_k:
                break

        return results


# Global instance - NOT loaded at startup
recommendation_rag = EventRecommendationRAG()


def get_recommendations(
    query: str, top_k: int = 5, item_type: Optional[str] = None
) -> List[Dict]:
    """Get recommendations for query string"""
    return recommendation_rag.recommend(query, top_k, item_type)


def update_data(trips: List[Dict], events: List[Dict]) -> int:
    """Update RAG data"""
    return recommendation_rag.load_data(trips, events)
