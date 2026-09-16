#!/usr/bin/env python3
"""CPU contracts, causal toy forwards, real tokenizer geometry and synthetic gates."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import require, sha
from diagnostics.general_model_numeric_analysis import candidate_scores
from diagnostics.evidence_label_calibration import readouts
from diagnostics.q01_mechanism_inputs import (digest, select_contexts, locate, build_requests, schedule, budget, LAYERS)
from diagnostics.q01_mechanism_hooks import HookRuntime, hidden_output, replace_output, resolve_module
from diagnostics.q01_mechanism_package import PARENT, read_json, read_lines
from diagnostics.q01_mechanism_scoring import Expression, ReadoutBuilder, nominate, compare_derived
from diagnostics.q01_mechanism_execution import accept_pass, check_geometry


class TinyTokenizer:
    eos_token_id, pad_token_id = 31, 0
    def encode(self, text, add_special_tokens=False): return [ord(c) - 96 for c in text]


def tiny_context(name, text, lex, *, probe=None, encoding='original', surface='H'):
    ids = TinyTokenizer().encode(text)
    c = {'record_id': name, 'condition': name, 'query_id': 'FD-3169-Q01' if probe is None else 'shared-background',
         'task': 'hate', 'prompt_text': text, 'prompt_tokens': len(ids), 'prompt_token_ids': ids,
         'prompt_sha256': sha(text.encode()), 'prompt_token_ids_sha256': digest(ids),
         'encoding': encoding, 'probe_id': probe,
         'bindings': [{'query_id': 'FD-3169-Q01', 'demo_family': 1, 'demo_surface': surface,
                       'lexicon_arm': lex, 'condition_id': name}]}
    c['context_sha256'] = digest(c)
    return c


def tiny_positions(context):
    roles = {'lexicon_slot': [4, 5, 6], 'lexicon_end': [6], 'before_lexicon': [3],
             'demo_answer_end': [9], 'query_hehe': [] if context['probe_id'] else [11],
             'unrelated_demo_end': [14], 'pre_answer': [context['prompt_tokens'] - 1]}
    return {'roles': {k: {'token_positions': v, 'applicable': bool(v)} for k, v in roles.items()}}


def tiny_runner():
    import torch
    from torch import nn
    torch.manual_seed(913)
    torch.set_num_threads(1)
    class Attention(nn.Module):
        def __init__(self):
            super().__init__(); self.linear = nn.Linear(8, 8, bias=False)
        def forward(self, hidden):
            n = torch.arange(1, hidden.shape[1] + 1, device=hidden.device)[None, :, None]
            return self.linear(hidden.cumsum(1) / n), None
    class Block(nn.Module):
        def __init__(self):
            super().__init__(); self.self_attn = Attention(); self.mlp = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        def forward(self, hidden):
            hidden = hidden + 0.1 * self.self_attn(hidden)[0]
            return hidden + 0.1 * self.mlp(hidden)
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__(); self.embed_tokens = nn.Embedding(32, 8); self.layers = nn.ModuleList([Block() for _ in range(36)])
        def forward(self, *, input_ids, attention_mask, use_cache=False):
            hidden = self.embed_tokens(input_ids)
            for block in self.layers: hidden = block(hidden)
            return SimpleNamespace(last_hidden_state=hidden)
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.model = Backbone(); self.lm_head = nn.Linear(8, 32, bias=False)
            self.config = SimpleNamespace(hidden_size=8)
    runner = SimpleNamespace(model=Model().float().eval(), tokenizer=TinyTokenizer(), device='cpu',
                             eos_ids={31}, padding_extra=0, numeric_runtime={'padding_policy': 'dynamic'},
                             identity={'fixture': 'tiny-causal-CPU-no-checkpoint'}, torch=torch)
    return runner


def tiny_catalog():
    result = []
    for i, (label, answer) in enumerate((('hate', 'xy'), ('non-hate', 'xyzx'))):
        ids = TinyTokenizer().encode(answer)
        result.append({'candidate_id': label, 'ordinal': i, 'canonical_answer': answer,
                       'answer_token_ids': ids, 'answer_tokens': len(ids),
                       'answer_token_ids_sha256': digest(ids), 'answer_sha256': sha(answer.encode())})
    return {'hate': result}


def options(**extra):
    return {'reference': True, 'prefix': False, 'permuted': False, 'replica_shift': 0, 'padding_extra': 0, **extra}


class HookTests(unittest.TestCase):
    def setUp(self):
        self.contexts = [tiny_context('O', 'abcdefghijklmnopqr', 'O'),
                         tiny_context('N', 'abcdjjj hijklmnopqr'.replace(' ', ''), 'N1')]
        self.runner = tiny_runner()
        self.positions = {c['record_id']: tiny_positions(c) for c in self.contexts}
        self.runtime = HookRuntime(self.runner, self.contexts, self.positions, 'cpu-test', expected_hidden=8)
        self.catalog = tiny_catalog()

    def request(self, **changes):
        r = {'request_id': 'test', 'kind': 'patch', 'recipient': 'O', 'donor': 'N',
             'module': 'block', 'layer': 15, 'role': 'pre_answer', 'positions': [17], 'encoding': 'original', 'probe_id': None}
        r.update(changes); return r

    def margin(self, row): return readouts(row)['margin/answer_sum']

    def test_actual_prompt_cache_reused_for_two_candidates_and_prefix(self):
        request = self.request()
        full = self.runtime.score(request, self.catalog, options())
        prefix = self.runtime.score(request, self.catalog, options(reference=False, prefix=True))
        self.assertEqual(self.runtime.forward_capture_count, 1)
        self.assertEqual(full['hook_receipt']['donor_vector_sha256'], prefix['hook_receipt']['donor_vector_sha256'])
        self.assertAlmostEqual(self.margin(full), self.margin(prefix), places=5)
        self.assertEqual(self.runtime.capture_receipts[0]['prompt_tokens'], 18)
        self.assertTrue(all(s['identity']['source_is_prompt_only'] for s in self.runtime.capture_receipts[0]['states']))

    def test_embedding_reconstruction_is_end_to_end_for_all_answer_tokens(self):
        actual = self.runtime.score(self.request(kind='embedding_reconstruction', module='embedding', layer=None,
                                    role='lexicon_slot', positions=[4, 5, 6]), self.catalog, options())
        donor = self.runtime.score(self.request(kind='baseline', recipient='N'), self.catalog, options())
        self.assertEqual(readouts(actual), readouts(donor))

    def test_self_patch_and_read_only_capture(self):
        baseline = self.runtime.score(self.request(kind='baseline'), self.catalog, options())
        for layer in LAYERS:
            value = self.runtime.score(self.request(kind='self_patch', donor='O', layer=layer), self.catalog, options())
            self.assertAlmostEqual(self.margin(value), self.margin(baseline), places=5)
        observed = self.runtime.score(self.request(kind='capture_only'), self.catalog, options())
        self.assertEqual(readouts(observed), readouts(baseline))
        self.assertFalse(observed['hook_receipt']['donor_cache_written_by_observer'])

    def test_causal_prefix_and_terminal_nonpropagation(self):
        baseline = self.runtime.score(self.request(kind='baseline'), self.catalog, options())
        for kwargs in ({'role': 'before_lexicon', 'positions': [3]},
                       {'layer': 35, 'role': 'query_hehe', 'positions': [11]}):
            actual = self.runtime.score(self.request(**kwargs), self.catalog, options())
            self.assertEqual(readouts(actual), readouts(baseline))

    def test_padding_and_candidate_order(self):
        base = self.runtime.score(self.request(), self.catalog, options())
        for opt in (options(padding_extra=64), options(permuted=True)):
            row = self.runtime.score(self.request(), self.catalog, opt)
            self.assertAlmostEqual(self.margin(base), self.margin(row), places=5)

    def test_rejects_answer_positions_and_cross_probe_or_encoding(self):
        with self.assertRaises(ValueError): self.runtime.donor(self.request(positions=[18]))
        for field, value in (('encoding', 'ab_forward'), ('probe_id', 'empty')):
            old = self.contexts[1][field]; self.contexts[1][field] = value
            with self.assertRaises(ValueError): self.runtime.donor(self.request())
            self.contexts[1][field] = old

    def test_corrupted_cache_is_detected(self):
        import torch
        d = self.runtime.donor(self.request())
        with torch.inference_mode(): d.vectors[0, 0] += 1
        with self.assertRaises(ValueError): self.runtime.donor(self.request())

    def test_exception_removes_all_hooks_and_restores_padding(self):
        self.runtime.capture('N')
        with patch.object(self.runner.model.lm_head, 'forward', side_effect=RuntimeError('injected forward failure')):
            with self.assertRaises(RuntimeError): self.runtime.score(self.request(), self.catalog, options(padding_extra=64))
        self.assertFalse(self.runtime.active); self.assertEqual(self.runner.padding_extra, 0)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.runner.model.modules()))

    def test_guard_rejects_candidate_bearing_capture(self):
        import torch
        with self.assertRaises(ValueError), self.runtime.guard(self.contexts[0], prompt_only=True):
            ids = torch.tensor([self.contexts[0]['prompt_token_ids'] + [24]])
            self.runner.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)

    def test_real_installed_qwen_module_shapes_with_tiny_random_cpu_weights(self):
        import torch
        with patch.object(torch.cuda, 'is_available', return_value=False):
            from transformers import Qwen3Config, Qwen3ForCausalLM
            cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                              num_attention_heads=2, num_key_value_heads=2, head_dim=8)
            cfg._attn_implementation = 'eager'
            model = Qwen3ForCausalLM(cfg).float().eval()
        shapes = {}
        handles = []
        for role, layer in (('embedding', None), ('block', 0), ('attention', 0), ('mlp', 0)):
            def observer(module, inputs, output, role=role):
                value = hidden_output(output); shapes[role] = tuple(value.shape)
                return replace_output(output, value.clone())
            handles.append(resolve_module(model, role, layer).register_forward_hook(observer))
        try:
            with torch.inference_mode(): model.model(input_ids=torch.tensor([[2, 3, 4]]), attention_mask=torch.ones(1, 3, dtype=torch.long), use_cache=False)
        finally:
            for h in handles: h.remove()
        self.assertEqual(shapes, {r: (1, 3, 16) for r in ('embedding', 'block', 'attention', 'mlp')})

    def test_worker_partial_checkpoint_and_resume_do_not_repeat_completed_requests(self):
        from diagnostics.q01_mechanism_execution import _worker
        import diagnostics.q01_mechanism_hooks as hook_module
        contexts, positions, runner = self.contexts, self.positions, self.runner
        requests = [self.request(request_id='r1', kind='baseline', donor=None), self.request(request_id='r2')]
        plan = {'code_sha256': {}, 'runtime_parent_plan': {}, 'runtime': {},
                'plan_id': 'cpu-test', 'catalog': self.catalog, 'eos_token_id': 31}
        hardware = {'uuid': 'CPU-test-only', 'physical_gpu_index': 0}
        class Connection:
            def __init__(self, jobs): self.jobs = iter(jobs); self.sent = []
            def recv(self): return next(self.jobs)
            def send(self, value): self.sent.append(value)
            def close(self): pass
        with tempfile.TemporaryDirectory(prefix='q01-checkpoint-') as temp:
            shard = Path(temp) / 'shards/0'
            for stop, wanted in ((0, 0), (None, 2), (None, 2)):
                connection = Connection([{'kind': 'pass', 'requests': requests, 'options': options(),
                    'directory': str(shard), 'stop_epoch': stop}, {'kind': 'stop'}])
                runtime_constructor = lambda r, cs, ps, pid: HookRuntime(r, cs, ps, pid, expected_hidden=8)
                with patch('diagnostics.general_model_numeric_kernel_v2.NumericRunner', return_value=runner), \
                     patch('diagnostics.evidence_label_calibration_execution.hardware_identity', return_value=hardware), \
                     patch('diagnostics.q01_mechanism_execution.validate_numeric_identity'), \
                     patch.object(hook_module, 'HookRuntime', side_effect=runtime_constructor):
                    _worker(connection, plan, contexts, positions, 0, hardware, str(ROOT))
                self.assertEqual(connection.sent[-1]['completed'], wanted)
                self.assertEqual(len(read_lines(shard / 'scores.jsonl')), wanted)
                if wanted and stop is None:
                    contents = (shard / 'scores.jsonl').read_bytes()
                    if 'previous' in locals(): self.assertEqual(contents, previous)
                    previous = contents
            from diagnostics.q01_mechanism_execution import verify_capture_audit
            from diagnostics.general_model_evidence_evaluation import file_sha
            artifacts = {str(p.relative_to(temp)): file_sha(p) for p in shard.iterdir()}
            verify_capture_audit(plan, contexts, requests, read_lines(shard / 'scores.jsonl'),
                                 Path(temp), {'artifacts': artifacts}, {}, {}, hidden_size=8)
            changed = deepcopy(read_lines(shard / 'scores.jsonl'))
            changed[1]['hook_receipt']['donor_vector_sha256'] = 'wrong'
            with self.assertRaises(ValueError):
                verify_capture_audit(plan, contexts, requests, changed, Path(temp), {'artifacts': artifacts}, {}, {}, hidden_size=8)

    def test_analysis_gate_precedes_reference_parsing(self):
        from diagnostics.q01_mechanism_execution import analyze
        with tempfile.TemporaryDirectory(prefix='q01-analysis-order-') as temp:
            with patch('diagnostics.q01_mechanism_execution.load_frozen', return_value=({}, [], {}, [])), \
                 patch('diagnostics.q01_mechanism_execution.check_run', side_effect=ValueError('unsealed raw scores')), \
                 patch('diagnostics.q01_mechanism_execution.read_json') as reader:
                with self.assertRaises(ValueError): analyze(temp, temp, Path(temp) / 'result')
                reader.assert_not_called()


def synthetic_scores(contexts, requests, catalog, opt):
    by_id = {c['record_id']: c for c in contexts}
    def value(cid):
        c = by_id[cid]; b = next(b for b in c['bindings'] if b['query_id'] == 'FD-3169-Q01')
        n = {'O': 0, 'N1': 1, 'N2': 1.1}[b['lexicon_arm']]
        lex_effect = 4.0 if c['probe_id'] is None else 0.8
        interaction = -0.5 if b['demo_surface'] == 'H' else 0
        return -3 + 0.2 * b['demo_family'] + n * (lex_effect + interaction)
    rows = []
    for r in requests:
        c = by_id[r['recipient']]
        m = value(r['recipient'])
        if r['kind'] == 'embedding_reconstruction': m = value(r['donor'])
        elif r['category'] not in ('baseline', 'engineering'):
            m += 0.6 * (value(r['donor']) - m)
        candidates = []
        for a in catalog[c['task']]:
            total = -10.0 + (m if a['candidate_id'] == 'non-hate' else 0)
            tokens = [total / a['answer_tokens']] * a['answer_tokens']
            scores = candidate_scores(tokens, -0.2)
            k = {**deepcopy(a), **scores, 'token_logprobs': tokens, 'scores': scores,
                 'finite_target_logits_checked': True, 'token_boundary_checked': True,
                 'prompt_tokens': c['prompt_tokens'], 'prompt_token_ids_sha256': c['prompt_token_ids_sha256'],
                 'eos_token_id': 151645, 'batch_size': 1, 'use_cache': False, 'causal_shift': 1,
                 'model_logits_dtype': 'torch.float32', 'reference_checked': opt['reference'],
                 'padded_sequence_tokens': None if opt['prefix'] else c['prompt_tokens'] + a['answer_tokens'] + 1 + opt['padding_extra'],
                 'scoring_implementation': 'uncached-prefix-only' if opt['prefix'] else 'full-sequence-selected-projection'}
            if opt['reference']: k['reference_scores'] = {**scores, 'token_logprobs': tokens}
            candidates.append(k)
        hook = None
        if r['kind'] == 'capture_only': hook = {'observation_counts': {'fixture': 2}, 'donor_cache_written_by_observer': False}
        elif r['kind'] != 'baseline':
            prefixes = {tuple(a['answer_token_ids'][:i]) for a in catalog[c['task']] for i in range(a['answer_tokens'] + 1)}
            hook = {'candidate_independent': True, 'source_is_prompt_only': True, 'donor_positions': r['positions'],
                    'hook_calls': len(prefixes) if opt['prefix'] else 2}
        rows.append({'request_id': r['request_id'], **{k: c[k] for k in
            ('record_id', 'query_id', 'task', 'condition', 'context_sha256', 'prompt_sha256')},
            'candidates': candidates, 'hook_receipt': hook, 'scoring_options': opt,
            'query_reference_loaded': False, 'formal_test_or_reserve_access': False,
            'physical_gpu_index': opt['replica_shift'], 'physical_gpu_uuid': 'synthetic-' + str(opt['replica_shift'])})
    return rows


class FrameAndScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.parent = read_json(PARENT / 'plan.json')
        cls.contexts, cls.index = select_contexts(read_lines(PARENT / 'contexts.jsonl'))
        model = read_json(Path(cls.parent['runtime_parent_plan']['package_path']) / 'models.json')[0]
        tok = AutoTokenizer.from_pretrained(str(ROOT / model['tokenizer_inventory']['logical_repo_path']),
                                          local_files_only=True, trust_remote_code=False, use_fast=True)
        cls.positions = {c['record_id']: locate(c, tok) for c in cls.contexts}
        cls.requests, cls.proofs = build_requests(cls.contexts, cls.index, cls.positions)
        cls.schedule = schedule(cls.requests, cls.contexts, cls.parent['catalog'])
        cls.science_requests = [r for r in cls.requests if r['category'] != 'engineering']
        cls.science_rows = synthetic_scores(cls.contexts, cls.science_requests, cls.parent['catalog'], options())
        cls.comparisons = ReadoutBuilder(cls.contexts, cls.requests, cls.science_rows, cls.parent['numeric_policy']['epsilon']).comparisons()

    def test_frame_positions_candidates_and_budget(self):
        b = budget(self.requests, self.schedule)
        self.assertEqual(b['scheduled_candidate_evaluations'], 144192)
        self.assertEqual(b['scoring_forward_calls'], 214768)
        self.assertEqual(len(self.proofs), 96)
        self.assertEqual(sum(not p['roles']['query_hehe']['applicable'] for p in self.positions.values()), 60)
        self.assertEqual(len({p['candidate_start'] for p in self.positions.values() if p['probe_id'] is not None}), 4)
        self.assertEqual(sum(r['category'] == 'primary' for r in self.requests), 3456)

    def test_independent_decimal_reproduction_of_all_four_historical_targets(self):
        from decimal import Decimal, localcontext
        historical = {r['record_id']: r for r in read_lines(PARENT.parent / 'run-stage-1-01/functional-b1-r0/scores.jsonl')}
        expected = {(1, 'N1'): (13.52717399597168, -0.5993232727050781, 4.295114282209971, -0.23042076182974824),
                    (1, 'N2'): (13.69474983215332, -0.4219017028808594, 4.402528091224232, -0.17089996277600417),
                    (2, 'N1'): (13.016891479492188, -0.585052490234375, 4.194666071121794, -0.15038863400767477),
                    (2, 'N2'): (13.209766387939453, -0.44567108154296875, 4.304215586163021, -0.0964004572583459)}
        def margin(family, lex, surface, probe, mean):
            row = historical[self.index[family, lex, surface, 'original', probe]['record_id']]
            scores = [sum(Decimal.from_float(x) for x in c['token_logprobs']) /
                      (Decimal(c['answer_tokens']) if mean else Decimal(1)) for c in row['candidates']]
            return scores[1] - scores[0]
        def ncc(family, lex, surface):
            ps = [1 / (1 + (-margin(family, lex, surface, p, True)).exp()) for p in ('empty', 'space', 'na', 'mask', 'lorem')]
            mean = sum(ps) / 5
            return margin(family, lex, surface, None, True) - (mean / (1 - mean)).ln()
        with localcontext() as context:
            context.prec = 60
            for (family, lex), values in expected.items():
                d = [margin(family, lex, s, None, False) - margin(family, 'O', s, None, False) for s in ('H', 'A')]
                n = [ncc(family, lex, s) - ncc(family, 'O', s) for s in ('H', 'A')]
                actual = ((d[0] + d[1]) / 2, d[0] - d[1], (n[0] + n[1]) / 2, n[0] - n[1])
                for a, b in zip(actual, values): self.assertLess(abs(a - Decimal.from_float(b)), Decimal('1e-12'))

    def test_bidirectional_effect_orientation_and_shared_coefficient_cancellation(self):
        r = next(r for r in self.comparisons if r['view'] == 'original/answer_sum' and r['direction'] == 'R')
        k = next(k for k in self.comparisons if k['cell_id'] == r['cell_id'] and k['view'] == r['view'] and k['direction'] == 'K')
        self.assertAlmostEqual(r['metrics']['C']['effect']['value'], k['metrics']['C']['effect']['value'])
        self.assertLess(k['arms']['H']['actual_shift']['value'], 0)
        self.assertLess(r['metrics']['I']['effect']['value'], 0)
        eps = self.parent['numeric_policy']['epsilon']
        self.assertEqual(r['metrics']['C']['residual']['numeric_bound'], 2 * eps)
        self.assertEqual(r['metrics']['I']['residual']['numeric_bound'], 4 * eps)

    def test_fixed_ncc_has_same_effect_and_target_as_mean(self):
        by_key = {(r['cell_id'], r['direction'], r['view']): r for r in self.comparisons}
        for r in self.comparisons:
            if r['view'] != 'ncc_fixed_recipient': continue
            raw = by_key[r['cell_id'], r['direction'], 'original/answer_mean']
            for metric in ('C', 'I'):
                for field in ('target', 'effect', 'residual'):
                    self.assertEqual(r['metrics'][metric][field], raw['metrics'][metric][field])

    def test_query_span_has_no_recalibrated_or_fabricated_background_view(self):
        q = [r for r in self.comparisons if r['role'] == 'query_hehe']
        self.assertTrue(q)
        self.assertFalse(any(r['view'].startswith(('ncc_recalibrated', 'ncc_single/', 'ncc_loo/')) for r in q))
        for cell in {r['cell_id'] for r in self.comparisons if r['role'] != 'query_hehe'}:
            views = {r['view'] for r in self.comparisons if r['cell_id'] == cell}
            self.assertEqual(len([v for v in views if v.startswith('ncc_single/')]), 5)
            self.assertEqual(len([v for v in views if v.startswith('ncc_loo/')]), 5)

    def test_nomination_requires_four_groups_two_directions_and_deduplicates(self):
        result = nominate(self.comparisons)
        self.assertLessEqual(len(result['selected_units']), 2)
        self.assertTrue(result['selected_units'])
        self.assertFalse(result['automatic_refinement_started'])
        broken = deepcopy(self.comparisons)
        for r in broken:
            if r['category'] == 'primary' and r['view'] == 'original/answer_sum' and r['group'] == 'F2:O-N2':
                for m in r['metrics'].values(): m['eligible_direction'] = False
        self.assertTrue(nominate(broken)['stop_this_scan_range'])

    def test_zero_target_has_no_recovery_ratio(self):
        zero_rows = deepcopy(self.science_rows)
        for row in zero_rows:
            for c in row['candidates']:
                c['token_logprobs'] = [-1.0] * c['answer_tokens']
                c.update(candidate_scores(c['token_logprobs'], c['eos_logprob']))
                c['scores'] = candidate_scores(c['token_logprobs'], c['eos_logprob'])
        result = ReadoutBuilder(self.contexts, self.requests, zero_rows, 0.001).comparisons()
        self.assertTrue(all(r['metrics']['C']['effect_over_target'] is None for r in result))
        self.assertTrue(nominate(result)['stop_this_scan_range'])

    def test_all_twelve_numerical_passes_on_synthetic_frame(self):
        plan = {'catalog': self.parent['catalog'], 'numeric_policy': self.parent['numeric_policy'], 'eos_token_id': 151645}
        by_id = {r['request_id']: r for r in self.requests}
        history = [r for r in self.science_rows if by_id[r['request_id']]['kind'] == 'baseline']
        references = {}; bridge = None
        for spec in self.schedule:
            rs = [by_id[rid] for rid in spec['request_ids']]
            rows = synthetic_scores(self.contexts, rs, self.parent['catalog'], spec['options'])
            result = accept_pass(plan, self.contexts, self.requests, rows, spec, history,
                                 references.get(spec['phase']), bridge if spec['phase'] == 'science' else None)
            self.assertEqual(result['status'], 'passed')
            if spec['mode'] == 'reference':
                references[spec['phase']] = rows
                if spec['phase'] == 'engineering': bridge = {r['record_id']: r for r in rows if by_id[r['request_id']]['kind'] == 'baseline'}

    def test_changed_scores_fail_gate_without_adjusting_epsilon(self):
        spec = next(p for p in self.schedule if p['pass_id'] == 'science-repeat')
        plan = {'catalog': self.parent['catalog'], 'numeric_policy': self.parent['numeric_policy'], 'eos_token_id': 151645}
        rows = synthetic_scores(self.contexts, self.science_requests, self.parent['catalog'], spec['options'])
        old = deepcopy(rows)
        c = rows[-1]['candidates'][1]
        c['token_logprobs'][0] -= 0.1; c.update(candidate_scores(c['token_logprobs'], c['eos_logprob'])); c['scores'] = candidate_scores(c['token_logprobs'], c['eos_logprob'])
        history = [r for r in old if next(x for x in self.science_requests if x['request_id'] == r['request_id'])['kind'] == 'baseline']
        with self.assertRaises(ValueError): accept_pass(plan, self.contexts, self.requests, rows, spec, history, old)


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.config = {'first_check_at': '2026-09-15T08:30:00+08:00',
            'checkpoint_at': '2026-09-15T16:45:00+08:00', 'release_deadline': '2026-09-15T16:55:00+08:00',
            'poll_interval_seconds': 1800, 'device_indices': [0, 1, 2, 3], 'plan': '/tmp/plan', 'run': '/tmp/run'}

    def test_no_gpu_check_before_window_and_half_hour_next_slot(self):
        from scripts.review.schedule_q01_gpu_window import epoch, decision
        start = epoch(self.config['first_check_at'])
        self.assertEqual(decision(self.config, start - 1)['action'], 'wait')
        self.assertEqual(decision(self.config, start + 2)['next_check_epoch'], start + 1800)
        self.assertEqual(decision(self.config, start + 1803)['next_check_epoch'], start + 3600)
        self.assertEqual(decision(self.config, epoch(self.config['checkpoint_at']))['action'], 'window_closed')

    def test_window_launch_requires_exact_four_cards_and_checkpoint_time(self):
        from scripts.review.schedule_q01_gpu_window import command
        args = command(self.config)
        self.assertEqual(args[args.index('--gpus') + 1:args.index('--stop-at')], ['0', '1', '2', '3'])
        self.assertEqual(args[-1], self.config['checkpoint_at'])
        with self.assertRaises(ValueError): command({**self.config, 'device_indices': [1, 2]})

    def test_preflight_rejects_loaded_but_quiet_gpu(self):
        from diagnostics.q01_mechanism_execution import preflight
        sample = '0,GPU-a,Model,20000,80000,0\n1,GPU-b,Model,0,80000,0\n'
        with patch('subprocess.run', return_value=SimpleNamespace(stdout=sample)):
            with self.assertRaises(ValueError): preflight([0, 1])


if __name__ == '__main__': unittest.main(verbosity=2)
