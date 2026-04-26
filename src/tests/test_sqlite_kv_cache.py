import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.sqlite_kv_cache import SQLiteKVCache

try:
    import numpy as np
except Exception:  # pragma: no cover - dependency may be absent in lightweight envs
    np = None


class SQLiteKVCacheTest(unittest.TestCase):
    def test_set_get_and_get_many(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with SQLiteKVCache(os.path.join(tmpdir, "cache.sqlite3")) as cache:
                cache.set("prompt", "a", pickle.dumps({"value": 1}))
                cache.set_many("prompt", {
                    "b": pickle.dumps({"value": 2}),
                    "c": pickle.dumps({"value": 3}),
                })

                self.assertEqual(pickle.loads(cache.get("prompt", "a")), {"value": 1})
                values = cache.get_many("prompt", ["a", "b", "missing"])
                self.assertEqual(pickle.loads(values["a"]), {"value": 1})
                self.assertEqual(pickle.loads(values["b"]), {"value": 2})
                self.assertIsNone(values["missing"])

    @unittest.skipIf(np is None, "numpy is not installed")
    def test_rag_cache_numpy_and_retrieval_roundtrip(self):
        from rag.cache import CacheManager

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(tmpdir)
            try:
                vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                cache.set_embedding("vec-key", vec)
                np.testing.assert_allclose(cache.get_embedding("vec-key"), vec)

                retrieval = {"ids": ["1", "2"], "sims": [0.9, 0.8]}
                cache.set_retrieval("ret-key", retrieval)
                self.assertEqual(cache.get_retrieval("ret-key"), retrieval)

                selection = {"shots": [{"doc_id": "1"}]}
                cache.set_selection("sel-key", selection)
                self.assertEqual(cache.get_selection("sel-key"), selection)
            finally:
                cache.close()


if __name__ == "__main__":
    unittest.main()
