#!/usr/bin/env python3
"""Meaningful CPU checks. No research checkpoint load and no CUDA initialization."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from decimal import Decimal, localcontext
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from diagnostics import hehe_sense_context_inputs_v1 as c
from diagnostics import case_attention_capture_v1 as a
from diagnostics import hehe_sense_context_runtime_v1 as rt

RULES = {'margin_repeat_order_hook_cap': 0.0, 'margin_padding_prefix_cap': 0.001,
         'attention_repeat_order_cap': 0.0, 'attention_element_cap': 0.0001,
         'attention_row_l1_cap': 0.001, 'attention_row_sum_cap': 0.000002,
         'margin_bound_floor': 0.000001, 'attention_mass_bound_floor': 0.0000001}


def request(index=0, lexical=True, demos=True):
    ids = list(range(6, 14))
    roles = {'pre_answer': [7], 'lexicon_end': [2] if lexical else [], 'demos_end': [4] if demos else [],
             'query_end': [6], 'query_all': [5, 6], 'query_focal': [5]}
    return {'request_id': 'synthetic-' + str(index), 'input_ids': ids, 'input_ids_sha256': c.digest(ids),
            'prompt_sha256': 'explicit-synthetic-prompt', 'prompt_tokens': len(ids), 'candidate_tokens': {'有': 3, '无': 4},
            'roles': roles, 'prefix_proofs': [{'role': k} for k in ('lexicon_end', 'demos_end') if roles[k]],
            'spans': [{'id': 'first', 'kind': 'lexicon_definition', 'label': 'first', 'token_positions': [0, 1, 2, 3]},
                      {'id': 'second', 'kind': 'demo_text', 'label': 'second', 'token_positions': [4, 5, 6, 7]}]}


def synthetic_forward(model, req, pad_token, padding='none', capture=True, prefix_role=None):
    roles = req['roles']
    n = len(req['input_ids'])
    if prefix_role:
        n = roles[prefix_role][0] + 1
        roles = {k: roles[k] if k == prefix_role else [] for k in c.ROLES}
    vector = np.full(16, -1.0, dtype=np.float32)
    vector[3], vector[4] = 1.0, 3.0
    if len(req['input_ids']) > req['prompt_tokens']:
        vector[2] = 5.0
    values = None
    if capture:
        values = np.zeros((2, 4, len(c.ROLES), n), dtype=np.float64)
        for ri, role in enumerate(c.ROLES):
            for pos in roles[role]:
                values[:, :, ri, :pos + 1] += 1 / ((pos + 1) * len(roles[role]))
    return vector, values, roles


class CPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        with patch.object(torch.cuda, 'is_available', return_value=False):
            from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(7)
        torch.set_num_threads(2)
        cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, attention_dropout=0,
                         bos_token_id=1, eos_token_id=2, pad_token_id=0)
        cfg._attn_implementation = 'eager'
        cls.torch = torch
        cls.model = Qwen3ForCausalLM(cfg).float().eval()

    def test_native_hook_logits_rows_and_cleanup(self):
        req = request()
        base, _, _ = a.forward(self.model, req, 0, capture=False)
        observed, maps, roles = a.forward(self.model, req, 0)
        np.testing.assert_array_equal(base, observed)
        self.assertEqual(maps.shape, (2, 4, 6, 8))
        a.validate_attention(maps, req)
        with self.torch.inference_mode():
            native = self.model.model(input_ids=self.torch.tensor([req['input_ids']]), use_cache=False,
                                      output_attentions=True, return_dict=True).attentions
        for li, raw in enumerate(native):
            for ri, role in enumerate(c.ROLES):
                expected = raw[0, :, roles[role], :].cpu().numpy().mean(axis=1, dtype=np.float64)
                np.testing.assert_array_equal(maps[li, :, ri], expected)
        self.assertTrue(all(not block.self_attn._forward_hooks for block in self.model.model.layers))
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_padding_and_true_prefix(self):
        req = request()
        v, maps, _ = a.forward(self.model, req, 0)
        for mode in ('left', 'right'):
            padded, pm, _ = a.forward(self.model, req, 0, padding=mode)
            self.assertLess(float(np.abs(v - padded).max()), 1e-6)
            self.assertLess(a.difference(maps, pm)['max_row_l1'], 1e-6)
        for role in ('lexicon_end', 'demos_end'):
            _, prefix, roles = a.forward(self.model, req, 0, prefix_role=role)
            a.validate_attention(prefix, req, roles)
            i = c.ROLES.index(role)
            self.assertLess(a.difference(maps[:, :, i:i + 1, :prefix.shape[-1]], prefix[:, :, i:i + 1])['max_row_l1'], 1e-6)

    def test_nonvisible_not_measured_zero(self):
        req = request()
        _, maps, _ = synthetic_forward(None, req, 0)
        summary = a.aggregate(maps, req)
        row = next(s for s in summary if s['role'] == 'lexicon_end' and s['span_id'] == 'second')
        self.assertFalse(row['visible'])
        self.assertIsNone(row['mass'])
        broken = maps.copy()
        broken[:, :, 1, -1] = 1e-9
        with self.assertRaisesRegex(ValueError, 'Future'):
            a.validate_attention(broken, req)

    def test_future_query_changes_do_not_rewrite_reference_prefix(self):
        req = request()
        _, before, _ = a.forward(self.model, req, 0)
        changed = dict(req, input_ids=req['input_ids'][:5] + [21, 22, 23])
        _, after, _ = a.forward(self.model, changed, 0)
        for role in ('lexicon_end', 'demos_end'):
            ri = c.ROLES.index(role)
            np.testing.assert_array_equal(before[:, :, ri], after[:, :, ri])
        absent = request(lexical=False, demos=False)
        _, maps, _ = a.forward(self.model, absent, 0)
        a.validate_attention(maps, absent)
        self.assertEqual(np.count_nonzero(maps[:, :, 1:3]), 0)

    def test_decimal_aggregation_and_density(self):
        req = request()
        _, maps, _ = synthetic_forward(None, req, 0)
        summary = a.aggregate(maps, req)
        with localcontext() as ctx:
            ctx.prec = 60
            for row in summary:
                if not row['visible']:
                    continue
                ri = c.ROLES.index(row['role'])
                keys = next(s for s in req['spans'] if s['id'] == row['span_id'])['token_positions']
                for li in range(2):
                    for h in range(4):
                        exact = sum((Decimal.from_float(float(maps[li, h, ri, k])) for k in keys), Decimal(0))
                        self.assertLessEqual(abs(float(exact) - row['mass'][li][h]), 1e-15)
                        exact_density = exact / Decimal(str(row['mean_visible_tokens']))
                        self.assertLessEqual(abs(float(exact_density) - row['density'][li][h]), 1e-15)
        qrow = next(s for s in summary if s['role'] == 'query_all' and s['span_id'] == 'second')
        self.assertEqual(qrow['mean_visible_tokens'], 2.5)

    def test_strict_gates_no_relaxation(self):
        good = {'kind': 'repeat', 'margin_difference': 0.0, 'attention': {'max_element': 0.0, 'max_row_l1': 0.0}}
        self.assertEqual(a.qualify([good], RULES)['margin_error_bound'], 1e-6)
        for row in [dict(good, margin_difference=1e-12), dict(good, attention={'max_element': 1e-15, 'max_row_l1': 1e-15}),
                    {'kind': 'left', 'margin_difference': 0.001001, 'attention': None},
                    {'kind': 'prefix', 'margin_difference': None, 'attention': {'max_element': 0.000100001, 'max_row_l1': 0.000100001}}]:
            with self.assertRaises(ValueError):
                a.qualify([row], RULES)

    def test_synthetic_worker_pause_resume_seal_corruption_and_terminal(self):
        # Execute the actual controller and worker on synthetic telemetry, in a
        # temporary directory. Native model capture is independently tested above.
        with tempfile.TemporaryDirectory(prefix='case-attention-fixture-') as td, ExitStack() as stack:
            tmp = Path(td)
            prepared = tmp / 'prepared'
            prepared.mkdir()
            c.write(prepared / 'manifest.json', {'explicitly_synthetic': True})
            requests = []
            arms = [c.CONDITIONS[0]] * 9
            for i, arm in enumerate(arms):
                qid = f'Q0{i//3+1}'
                req = request(i, arm[2] != 'none', arm[1])
                req.update(request_id=f'synthetic-{i}-{qid}-{arm[0]}', query_id=qid, condition=arm[0], dictionary_id=f'D0{i%3+1}')
                requests.append(req)
            c.write(prepared / 'comparisons.json', {'comparisons':[{'comparison_id': 'synthetic-difference', 'kind':'variant_minus_baseline', 'query_id':'Q01', 'condition':'L', 'terms':[{'request_id':requests[0]['request_id'],'coefficient':1},{'request_id':requests[2]['request_id'],'coefficient':-1}]}]})
            c.write(prepared / 'analysis-references.json', {'references': [
                {'query_id': qid, 'reference': '有' if qid == 'Q02' else '无', 'synthetic': True} for qid in ('Q01', 'Q02', 'Q03')]})
            c.write(prepared / 'definition-components.json', {'records':[{'request_id':r['request_id'], 'parent_span_id':'first', 'components':[{'component':'original','char_start':0,'char_end':2,'owned_token_positions':[0,1]}], 'separator_or_cross_boundary_tokens':[]} for r in requests]})
            plan = {'runtime_versions': runtime_versions(), 'acceptance': RULES}
            profile = {'vocab_size': 16, 'layers': 2, 'heads': 4, 'pad_token_id': 0, 'candidate_tokens': {'有': 3, '无': 4},
                       'eos_token_ids': [2], 'metadata_sources': [], 'weight_sources': []}
            allocation = [{'index': 0, 'uuid': 'synthetic-cpu-only', 'name': 'synthetic', 'total_mib': 48000}]
            inventory = {'devices': [dict(allocation[0], used_mib=0, utilization=0)], 'compute_processes': [], 'synthetic': True}
            bound = tmp / 'bound.json'
            c.write(bound, {'prepared_manifest': c.info(prepared / 'manifest.json'), 'allocation': allocation,
                            'runtime_versions': runtime_versions(), 'authorization_note': 'CPU SYNTHETIC FIXTURE ONLY'})
            run = tmp / 'run-synthetic'
            stack.enter_context(patch.object(c, 'WORK', tmp))
            stack.enter_context(patch.object(c, 'validate', return_value=(plan, profile, requests)))
            stack.enter_context(patch.object(rt, 'gpu_inventory', return_value=inventory))
            stack.enter_context(patch.object(rt.signal, 'signal'))
            stack.enter_context(patch('diagnostics.cross_model_applicability_models_v1.load_checkpoint', return_value=(object(), {'synthetic': True})))
            calls = []

            def counted(*args, **kwargs):
                result = synthetic_forward(*args, **kwargs)
                calls.append((args[1]['request_id'], kwargs.get('prefix_role')))
                if len(calls) == 2:
                    (run / 'STOP').touch()
                return result

            stack.enter_context(patch.object(a, 'forward', side_effect=counted))

            class Process:
                pid = 987654321
                def __init__(self, cmd, **kwargs):
                    val = lambda name: cmd[cmd.index(name) + 1]
                    rt.worker(val('--prepared'), val('--bound'), val('--run'), val('--invocation'), val('--phase'))
                def wait(self):
                    return 0

            stack.enter_context(patch.object(rt.subprocess, 'Popen', Process))
            paused = rt.supervise(prepared, bound, run)
            self.assertEqual(paused['status'], 'paused')
            self.assertEqual(len(calls), 2)
            first_bytes = rt.record_path(run, 'baseline', requests[0]['request_id']).read_bytes()
            (run / 'STOP').unlink()
            qualified = rt.supervise(prepared, bound, run, resume=True)
            self.assertEqual(qualified['status'], 'qualified')
            self.assertEqual(len(calls), 54 + 9 + 9)
            self.assertEqual(first_bytes, rt.record_path(run, 'baseline', requests[0]['request_id']).read_bytes())
            checked = rt.check_run(prepared, run)
            self.assertFalse(checked['query_reference_join_performed'])
            self.assertEqual(checked['qualification']['maxima']['prefix']['comparisons'], 9)
            from diagnostics.hehe_sense_context_report_v1 import analyze
            with self.assertRaisesRegex(ValueError, 'Full run'):
                analyze(prepared, run, tmp / 'premature-analysis')
            self.assertFalse((tmp / 'premature-analysis').exists())
            complete = rt.supervise(prepared, bound, run, phase='full')
            self.assertEqual(complete['status'], 'complete')
            self.assertEqual(len(calls), 81)
            rt.check_run(prepared, run)
            result = analyze(prepared, run, tmp / 'synthetic-analysis')
            self.assertEqual((result['conditions'], result['comparisons']), (9, 1))
            self.assertTrue(c.read(tmp / 'synthetic-analysis/audit.json')['query_references_joined_after_release'])
            self.assertTrue(all(row['delta_m'] == 0 for row in c.read(tmp / 'synthetic-analysis/comparisons.json')))
            final_scores = c.read(tmp / 'synthetic-analysis/scores.json')
            self.assertTrue(all(row['reference_aligned_margin'] == (-row['m'] if row['query_id']=='Q02' else row['m']) for row in final_scores))
            with self.assertRaisesRegex(ValueError, 'terminal'):
                rt.supervise(prepared, bound, run, phase='full')
            item = c.read(rt.record_path(run, 'production', requests[0]['request_id']))
            path = Path(item['attention']['path'])
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'Pinned bytes'):
                rt.check_run(prepared, run)
        self.assertFalse(self.torch.cuda.is_initialized())


def runtime_versions():
    from diagnostics.cross_model_applicability_execution_v1 import runtime_versions as versions
    return versions()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CPUChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    import torch
    receipt = {'status': 'pass' if result.wasSuccessful() else 'fail', 'tests': result.testsRun,
               'failures': len(result.failures), 'errors': len(result.errors), 'CUDA_initialized': torch.cuda.is_initialized(),
               'research_pretrained_weights_loaded': False, 'native_model': 'tiny random CPU Qwen3, 2 layers/4 heads',
               'lifecycle_fixture': 'synthetic telemetry only, /tmp; actual worker/controller; 81 forwards including reuse; post-release 9-score/1-comparison report',
               'GPU_qualification': False, 'implementation_snapshot': [c.info(p) for p in c.CODE]}
    if args.output:
        c.write(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
