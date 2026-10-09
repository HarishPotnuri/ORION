import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from orion.memory.embedding import OnnxTextEmbeddingProvider


class FakeSession:
    def __init__(self, inputs):
        self.inputs = inputs
        self.calls = []

    def get_inputs(self):
        return self.inputs

    def run(self, _outputs, values):
        self.calls.append(values)
        return [[[0.1, 0.2], [0.3, 0.4]]]


class EmbeddingProviderTests(unittest.TestCase):
    def test_lazily_runs_configured_direct_text_onnx_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "bge.onnx"
            model_path.touch()
            session = FakeSession([SimpleNamespace(name="text", type="tensor(string)")])
            provider = OnnxTextEmbeddingProvider(
                model_path, session_factory=lambda _path: session
            )

            result = provider.embed(["hello", "world"])

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(session.calls, [{"text": ["hello", "world"]}])

    def test_requires_text_input_contract_and_fails_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "bge.onnx"
            model_path.touch()
            session = FakeSession([SimpleNamespace(name="input_ids", type="tensor(int64)")])
            provider = OnnxTextEmbeddingProvider(
                model_path, session_factory=lambda _path: session
            )

            with self.assertRaisesRegex(RuntimeError, "tokenizer tensors"):
                provider.embed(["hello"])

    def test_missing_embedding_model_fails_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = OnnxTextEmbeddingProvider(Path(directory) / "missing.onnx")
            with self.assertRaisesRegex(FileNotFoundError, "Model file not found"):
                provider.embed(["hello"])


if __name__ == "__main__":
    unittest.main()
