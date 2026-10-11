"""Small, exact guardrail index without native model or Annoy worker threads."""

import asyncio
import logging

import numpy as np
from google import genai
from google.genai import types
from nemoguardrails.embeddings.index import EmbeddingsIndex
from nemoguardrails.rails.llm.config import EmbeddingsCacheConfig


logger = logging.getLogger(__name__)


class GeminiGuardrailIndex(EmbeddingsIndex):
    def __init__(self, api_key, model):
        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=20000)
        )
        self.model = model
        self.items = []
        self.matrix = None

    @property
    def embedding_size(self):
        return 0 if self.matrix is None else self.matrix.shape[1]

    @property
    def cache_config(self):
        return EmbeddingsCacheConfig(enabled=False)

    async def _get_embeddings(self, texts):
        # Separate Content objects are essential: Embedding 2 otherwise combines
        # a list of strings into one vector instead of returning one per item.
        contents = [
            types.Content(parts=[types.Part.from_text(text=text)]) for text in texts
        ]
        result = await asyncio.to_thread(
            self.client.models.embed_content, model=self.model, contents=contents
        )
        vectors = [embedding.values for embedding in (result.embeddings or [])]
        if len(vectors) != len(texts):
            raise ValueError("Gemini returned an unexpected number of embeddings")
        return vectors

    async def add_item(self, item):
        await self.add_items([item])

    async def add_items(self, items):
        self.items.extend(items)
        self.matrix = None

    async def build(self):
        if not self.items:
            return
        logger.info("Building Gemini guardrail index (%d examples)", len(self.items))
        matrix = np.asarray(
            await self._get_embeddings([item.text for item in self.items]),
            dtype=np.float32,
        )
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("Gemini returned invalid guardrail embeddings")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.maximum(norms, 1e-12)
        logger.info("Gemini guardrail index ready")

    async def search(self, text, max_results=20, threshold=None):
        if self.matrix is None:
            await self.build()
        if self.matrix is None:
            return []
        query = np.asarray((await self._get_embeddings([text]))[0], dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        similarities = self.matrix @ query
        # Preserve NeMo's Annoy angular-distance threshold semantics.
        scores = 1 - np.sqrt(np.maximum(0, 2 - 2 * similarities)) / 2
        indices = np.argsort(-similarities, kind="stable")[:max_results]
        return [
            self.items[int(i)] for i in indices
            if threshold is None or threshold == float("inf") or scores[i] >= threshold
        ]
