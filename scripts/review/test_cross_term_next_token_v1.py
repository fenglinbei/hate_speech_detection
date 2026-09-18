#!/usr/bin/env python3
"""CPU acceptance of positions, checkpoint integrity and numerical stage gates."""
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics import cross_term_next_token_v1 as contract
from diagnostics.cross_term_next_token_gpu_v1 import forward_logits, save_vector


class CrossTermTests(unittest.TestCase):
    def test_mask_positions_and_right_padding(self):
        ids = [4, 8, 10, 17]
        for side in ['none', 'left_to_next_strict_multiple_of_16', 'right_to_next_strict_multiple_of_16']:
            p = contract.padded_input(ids, side)
            self.assertEqual([v for v, m in zip(p['input_ids'], p['attention_mask']) if m], ids)
            self.assertEqual([v for v, m in zip(p['position_ids'], p['attention_mask']) if m], [0, 1, 2, 3])
            self.assertEqual(p['input_ids'][p['last_valid_index']], 17)
            if side.startswith('right'):
                self.assertNotEqual(p['last_valid_index'], p['tensor_tokens'] - 1)
        self.assertEqual(contract.padded_input(list(range(16)), 'left_to_next_strict_multiple_of_16')['tensor_tokens'], 32)

    def test_tiny_native_qwen3_forward_and_padding(self):
        import numpy as np
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.set_num_threads(2); torch.manual_seed(731)
        config = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=64, attention_dropout=0.0, pad_token_id=151643)
        config._attn_implementation = 'eager'
        model = Qwen3ForCausalLM(config).float().eval()
        ids = [11, 25, 36, 49, 57]
        original = None
        for side in ['none', 'left_to_next_strict_multiple_of_16', 'right_to_next_strict_multiple_of_16']:
            p, actual = forward_logits(model, ids, side, 'cpu')
            tensor = {k: torch.tensor([p[k]]) for k in ['input_ids', 'attention_mask', 'position_ids']}
            with torch.inference_mode():
                full = model(**tensor, use_cache=False, return_dict=True).logits[0].numpy()
            np.testing.assert_allclose(actual, full[p['last_valid_index']], rtol=0, atol=1e-6)
            if original is None:
                original = actual
            else:
                np.testing.assert_allclose(actual, original, rtol=0, atol=1e-6)
            if side.startswith('right'):
                self.assertGreater(float(np.max(np.abs(actual - full[-1]))), 0.001)

    def test_truncation_and_unknown_padding_rejected(self):
        for ids, side in [([], 'none'), ([1] * 40961, 'none'), ([1, 2], 'mystery')]:
            with self.assertRaises(ValueError):
                contract.padded_input(ids, side)

    def engineering_fixture(self):
        base = {'m': 0.5, 'z_no': 0.5, 'z_yes': 0.0, 'log_p_no': -1.0,
                'log_p_yes': -1.5, 'legal_mass': 0.5, 'log_legal_mass': -0.7, 'pair_support_no': 0.6}
        return {p: {str(i): copy.deepcopy(base) for i in range(120)} for p in contract.PASS_NAMES[:-1]}

    def test_qualification_bound_and_complete_inventory(self):
        data = self.engineering_fixture()
        config = contract.read(contract.PREP / 'qualification-plan.json')
        self.assertEqual(contract.qualification_values(data, config)['margin_error_bound'], 0.000001)
        data['engineering-left-padding']['17']['m'] += 0.000125
        data['engineering-left-padding']['17']['log_p_no'] += 0.000125
        self.assertAlmostEqual(contract.qualification_values(data, config)['margin_error_bound'], 0.00025)
        del data['engineering-repeat']['1']
        with self.assertRaises(ValueError): contract.qualification_values(data, config)

    def test_engineering_failures_cannot_enter_science(self):
        config = contract.read(contract.PREP / 'qualification-plan.json')
        for pass_id, delta in [('engineering-repeat', 1e-5), ('engineering-reverse-request-order', 1e-5),
                               ('engineering-left-padding', 0.002), ('engineering-right-padding', 0.002)]:
            data = self.engineering_fixture()
            data[pass_id]['0']['m'] += delta
            data[pass_id]['0']['log_p_no'] += delta
            with self.assertRaises(ValueError): contract.qualification_values(data, config)
        data = self.engineering_fixture(); data['engineering-reference']['0']['log_p_no'] += 0.01
        with self.assertRaises(ValueError): contract.qualification_values(data, config)

    def test_checkpoint_reconstruction_and_mutation_detection(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp); request = {'request_id': 'test', 'condition_id': 'test-condition', 'prompt_sha256': 'synthetic',
                'input_ids': [7, 9], 'input_ids_sha256': 'synthetic-ids'}
            spec = {'pass_id': 'engineering-reference', 'padding': 'none'}
            contract.atomic_json(run / 'binding.json', {'run_id': 'synthetic-run'})
            binding = contract.sha(run / 'binding.json')
            record_path, vector_path = contract.record_paths(run, spec['pass_id'], request['request_id'])
            vector = np.zeros(151936, dtype=np.float32); vector[42192] = 0.25
            save_vector(vector_path, vector)
            score = contract.math_module().readout(vector)
            record = dict(request, pass_id=spec['pass_id'], binding_sha256=binding,
                physical_score_id='synthetic-run:engineering-reference:test',
                candidate_forward_calls=1, candidates_share_forward=True, vocab_size=151936,
                prepared_input=contract.padded_input(request['input_ids'], 'none'),
                raw_logits=contract.file_info(vector_path), readout=score)
            record.pop('input_ids')
            contract.atomic_json(record_path, record)
            before = record_path.read_bytes(), vector_path.read_bytes()
            actual = contract.check_record(run, spec, request, binding)
            self.assertEqual(actual['readout'], score)
            self.assertEqual(before, (record_path.read_bytes(), vector_path.read_bytes()))
            modified = copy.deepcopy(record); modified['readout']['m'] += 0.5
            contract.atomic_json(record_path, modified)
            with self.assertRaises(ValueError): contract.check_record(run, spec, request, binding)
            contract.atomic_json(record_path, record)
            vector[42192] = -1.0
            save_vector(vector_path, vector)
            with self.assertRaises(ValueError): contract.check_record(run, spec, request, binding)

    def test_committed_receipt_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'receipt.json'
            contract.atomic_json(path, {'first': True}, replace=False)
            with self.assertRaises(ValueError): contract.atomic_json(path, {'first': False}, replace=False)
            self.assertEqual(contract.read(path), {'first': True})

    def test_scoring_input_and_worker_reference_isolation(self):
        rows = [json.loads(l) for l in (contract.PREP / 'model-inputs.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 120)
        disallowed = {'query_reference', 'reference', 'attack_severity', 'rule_fit', 'sense_fit', 'original_gold'}
        for row in rows:
            self.assertFalse(set(row) & disallowed)
        worker = (ROOT / 'src/diagnostics/cross_term_next_token_gpu_v1.py').read_text()
        self.assertNotIn("plan['analysis_plan']", worker)
        self.assertNotIn('reference_status(', worker)


if __name__ == '__main__':
    unittest.main(verbosity=2)
