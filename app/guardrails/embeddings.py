"""Keep local guardrail inference within the hosted service's memory budget."""

import asyncio

from nemoguardrails.embeddings.providers.fastembed import FastEmbedEmbeddingModel


class LowMemoryFastEmbed(FastEmbedEmbeddingModel):
    engine_name = "LowMemoryFastEmbed"

    def __init__(self, embedding_model: str, **kwargs):
        kwargs.setdefault("threads", 1)
        kwargs.setdefault("enable_cpu_mem_arena", False)
        super().__init__(embedding_model, **kwargs)

    def encode(self, documents: list[str]) -> list[list[float]]:
        # The default batch of 256 can exhaust RAM while indexing intents.
        return [vector.tolist() for vector in self.model.embed(documents, batch_size=1)]

    async def encode_async(self, documents: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.encode, documents)
