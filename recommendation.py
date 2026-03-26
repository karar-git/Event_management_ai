"""
RAG-based Recommendation System for SumerTrip
Simple: embed user query string -> find nearest events/trips
"""

from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
from config import config


class EventRecommendationRAG:
    """Simple RAG: embed query string, find nearest matches"""

    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.dimension = 384
        self.index = None
        self.all_items = []

    def _create_item_text(self, item: Dict, item_type: str) -> str:
        """Create searchable text from item"""
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

        # Build FAISS index
        texts = [item["text"] for item in self.all_items]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)

        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype("float32"))

        return len(self.all_items)

    def recommend(
        self, query: str, top_k: int = 5, item_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Find nearest matches for query string

        Args:
            query: User description/preferences as string
            top_k: Number of results
            item_type: Filter 'trip' or 'event'

        Returns:
            List of {id, type, score, data}
        """
        if not self.index or not self.all_items:
            return []

        # Embed query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Search
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


# Default data
DEFAULT_TRIPS = [
    {
        "id": 1,
        "title": "زيقورة أور العظيمة",
        "titleEn": "Great Ziggurat of Ur",
        "location": "ذي قار",
        "price": 120,
        "rating": 4.9,
        "category": "Heritage",
        "duration": "يومان",
        "description": "استكشف زيقورة أور العظيمة، المعبد الضخم",
        "highlights": ["زيقورة أور", "مقابر الملكة", "المتحف الأثري"],
    },
    {
        "id": 2,
        "title": "بابل الأسطورية",
        "titleEn": "Legendary Babylon",
        "location": "بابل",
        "price": 95,
        "rating": 4.8,
        "category": "Heritage",
        "duration": "يوم كامل",
        "description": "بوابة عشتار وطريق المواكب",
        "highlights": ["بوابة عشتار", "قصر نبوخذنصر"],
    },
    {
        "id": 3,
        "title": "قلعة أربيل",
        "titleEn": "Erbil Citadel",
        "location": "أربيل",
        "price": 55,
        "rating": 4.8,
        "category": "Trips",
        "duration": "نصف يوم",
        "description": "أقدم مدينة مأهولة",
        "highlights": ["قلعة أربيل", "البازار"],
    },
    {
        "id": 4,
        "title": "أهوار العراق",
        "titleEn": "Mesopotamian Marshes",
        "location": "ذي قار",
        "price": 85,
        "rating": 4.9,
        "category": "Camping",
        "duration": "يومان",
        "description": "الأهوار الرافدينية يونسكو",
        "highlights": ["المشحوف", "بيوت القصب", "طيور"],
    },
    {
        "id": 5,
        "title": "نينوى",
        "titleEn": "Nineveh",
        "location": "الموصل",
        "price": 110,
        "rating": 4.7,
        "category": "Heritage",
        "duration": "يوم كامل",
        "description": "الإمبراطورية الآشورية",
        "highlights": ["أسوار نينوى", "قصر سنحاريب"],
    },
    {
        "id": 6,
        "title": "بغداد",
        "titleEn": "Baghdad Tour",
        "location": "بغداد",
        "price": 40,
        "rating": 4.6,
        "category": "Trips",
        "duration": "يوم كامل",
        "description": "جادة المتنبي ومتحف العراق",
        "highlights": ["المتنبي", "متحف العراق"],
    },
    {
        "id": 7,
        "title": "النجف الأشرف",
        "titleEn": "Najaf",
        "location": "النجف",
        "price": 45,
        "rating": 4.9,
        "category": "Trips",
        "duration": "يوم كامل",
        "description": "مرقد الإمام علي",
        "highlights": ["مرقد الإمام علي", "وادي السلام"],
    },
    {
        "id": 8,
        "title": "جبال كردستان",
        "titleEn": "Kurdistan Mountains",
        "location": "كردستان",
        "price": 150,
        "rating": 4.8,
        "category": "Camping",
        "duration": "3 أيام",
        "description": "جبال وشلالات كردستان",
        "highlights": ["شلالات", "تخييم جبلي"],
    },
]

DEFAULT_EVENTS = [
    {
        "id": 1,
        "title": "مهرجان بابل",
        "titleEn": "Babylon Festival",
        "location": "بابل",
        "price": 60,
        "rating": 4.9,
        "category": "Festival",
        "date": "2026-04-15",
        "description": "مهرجان الفنون والثقافة",
    },
    {
        "id": 2,
        "title": "معرض الكتاب",
        "titleEn": "Book Fair",
        "location": "بغداد",
        "price": 10,
        "rating": 4.8,
        "category": "Culture",
        "date": "2026-03-30",
        "description": "معرض بغداد الدولي للكتاب",
    },
    {
        "id": 3,
        "title": "مهرجان المربد",
        "titleEn": "Al-Mirbad",
        "location": "البصرة",
        "price": 0,
        "rating": 4.9,
        "category": "Culture",
        "date": "2026-04-01",
        "description": "مهرجان الشعر",
    },
    {
        "id": 4,
        "title": "ليالي أربيل",
        "titleEn": "Erbil Nights",
        "location": "أربيل",
        "price": 25,
        "rating": 4.7,
        "category": "Festival",
        "date": "2026-05-10",
        "description": "ليالي فنية موسيقية",
    },
    {
        "id": 5,
        "title": "يوم التراث",
        "titleEn": "Heritage Day",
        "location": "النجف",
        "price": 0,
        "rating": 4.8,
        "category": "Culture",
        "date": "2026-04-20",
        "description": "احتفالية التراث العراقي",
    },
]


# Global instance
recommendation_rag = EventRecommendationRAG()
recommendation_rag.load_data(DEFAULT_TRIPS, DEFAULT_EVENTS)


def get_recommendations(
    query: str, top_k: int = 5, item_type: Optional[str] = None
) -> List[Dict]:
    """Get recommendations for query string"""
    return recommendation_rag.recommend(query, top_k, item_type)


def update_data(trips: List[Dict], events: List[Dict]) -> int:
    """Update RAG data"""
    return recommendation_rag.load_data(trips, events)
