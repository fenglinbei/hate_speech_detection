import tempfile
import unittest
from pathlib import Path

from data.build_data import BuildCacheManager, build_prompt, token_length, truncate_prompt_from_tail


class CharacterTokenizer:
    name_or_path = "character-tokenizer"

    def encode(self, text, add_special_tokens=True):
        token_ids = list(text)
        if add_special_tokens:
            return ["<bos>"] + token_ids + ["<eos>"]
        return token_ids

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        if skip_special_tokens:
            token_ids = [token_id for token_id in token_ids if token_id not in {"<bos>", "<eos>"}]
        return "".join(token_ids)


class PromptTailTruncationTest(unittest.TestCase):
    def test_tail_truncation_keeps_prefix_within_token_budget(self):
        tokenizer = CharacterTokenizer()

        truncated, original_len, truncated_len, was_truncated = truncate_prompt_from_tail(
            tokenizer,
            "abcdefghij",
            max_length=7,
        )

        self.assertTrue(was_truncated)
        self.assertEqual(original_len, 12)
        self.assertEqual(truncated, "abcde")
        self.assertEqual(truncated_len, 7)
        self.assertLessEqual(token_length(tokenizer, truncated), 7)

    def test_auto_length_disabled_truncates_rendered_prompt_tail(self):
        class Config:
            task_type = "structured"
            prompt_template = "prefix:{text}:suffix"
            example_template = ""
            system_prompt = ""
            use_srag = False
            use_lex = False
            use_global_demos = False
            enable_build_cache = False
            enable_retrieval_cache = False
            build_cache_dir = ""
            cache_backend = "sqlite"
            retrieval_batch_size = 1
            auto_length = False
            max_length = 10
            srag_top_k = 0
            srag_threshold = 0
            weights = None
            weights_reverse = False
            similarity_alpha = 1
            ramdom_strategy = "none"
            random_ratio = 0.0
            random_temperature = 1.0
            candidate_multiplier = 1
            mmr = False

        raw_data = {
            "id": "sample-1",
            "content": "abcdefghij",
            "quadruples": [
                {
                    "target": "x",
                    "argument": "y",
                    "targeted_group": "non-hate",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = BuildCacheManager(cache_dir=str(Path(tmp_dir) / "cache"), enabled=False)
            messages = build_prompt(
                datas=[raw_data],
                config=Config(),
                tokenizer=CharacterTokenizer(),
                build_cache=cache,
            )

        self.assertEqual(messages[0]["input"], "prefix:a")
        self.assertLessEqual(token_length(CharacterTokenizer(), messages[0]["input"]), Config.max_length)


if __name__ == "__main__":
    unittest.main()
