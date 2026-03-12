from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


class HuggingFaceEmbeddings(Embeddings):
    """HuggingFace embedding wrapper for LangChain."""

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        """Vectorize multiple texts."""
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        """Vectorize a single text (query)."""
        return self.model.encode(text).tolist()


embedding_model = HuggingFaceEmbeddings(model)
