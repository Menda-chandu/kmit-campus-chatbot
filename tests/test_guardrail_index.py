import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from nemoguardrails.embeddings.index import IndexItem

from app.guardrails.index import GeminiGuardrailIndex


class GuardrailIndexTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client_patch = patch("app.guardrails.index.genai.Client")
        self.client = self.client_patch.start().return_value
        self.addCleanup(self.client_patch.stop)
        self.index = GeminiGuardrailIndex("test-key", "test-model")
        self.vectors = {"hello": [1, 0], "bye": [0, 1], "zero": [0, 0]}

        def embed(**kwargs):
            return SimpleNamespace(embeddings=[
                SimpleNamespace(values=self.vectors[content.parts[0].text])
                for content in kwargs["contents"]
            ])

        self.client.models.embed_content.side_effect = embed

    async def test_independent_contents_ranking_and_threshold(self):
        hello, bye = IndexItem("hello"), IndexItem("bye")
        await self.index.add_items([bye, hello])
        await self.index.build()
        self.assertEqual(self.index.embedding_size, 2)
        self.assertEqual(await self.index.search("hello", 1), [hello])
        self.assertEqual(await self.index.search("hello", threshold=0.8), [hello])
        self.assertEqual(await self.index.search("hello", threshold=float("inf")), [hello, bye])
        call = self.client.models.embed_content.call_args
        self.assertEqual(call.kwargs["model"], "test-model")
        self.assertEqual(call.kwargs["contents"][0].parts[0].text, "hello")

    async def test_empty_zero_vectors_and_rebuild(self):
        self.assertEqual(await self.index.search("hello"), [])
        await self.index.add_item(IndexItem("zero"))
        await self.index.build()
        self.assertTrue(np.isfinite(self.index.matrix).all())
        await self.index.add_item(IndexItem("hello"))
        self.assertIsNone(self.index.matrix)
        result = await self.index.search("hello", 1)
        self.assertEqual(result[0].text, "hello")

    async def test_aggregated_response_rejected(self):
        self.client.models.embed_content.side_effect = None
        self.client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[1, 0])]
        )
        with self.assertRaisesRegex(ValueError, "number of embeddings"):
            await self.index._get_embeddings(["hello", "bye"])

    async def test_nemo_greeting_with_custom_index(self):
        from langchain_core.language_models.fake import FakeListLLM
        from app.guardrails import rails

        def embed(**kwargs):
            return SimpleNamespace(embeddings=[
                SimpleNamespace(values=[1, 0]) for _ in kwargs["contents"]
            ])

        self.client.models.embed_content.side_effect = embed
        with patch.object(rails, "ChatGroq", return_value=FakeListLLM(
            responses=["express greeting"]
        )), patch.object(rails.settings, "GEMINI_API_KEY", "test-key"):
            rails.initialize_rails()
            # generate() owns its event loop, so keep it outside this test's loop.
            import asyncio
            fired, response = await asyncio.to_thread(rails.guard, "hello")
        self.assertTrue(fired)
        self.assertIn("Hello! I'm the KMIT Campus Chatbot", response)


if __name__ == "__main__":
    unittest.main()
