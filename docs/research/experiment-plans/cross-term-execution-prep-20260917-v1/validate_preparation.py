"""Input isolation and independent CPU arithmetic audit; never imports a model runtime."""
import hashlib
import importlib.util
import json
import math
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FROZEN = HERE.parent / 'cross-term-joint-v1/frozen-01'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(p):
    return json.loads(p.read_text())


def verify(e):
    b = (ROOT / e['path']).read_bytes()
    require(hashlib.sha256(b).hexdigest() == e['sha256'], f'hash mismatch: {e["path"]}')
    if 'bytes' in e:
        require(len(b) == e['bytes'], 'byte count mismatch')


def arithmetic_audit(design):
    spec = importlib.util.spec_from_file_location('cross_term_next_token_math', HERE / 'next_token_math.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ref = {'kind': 'synthetic_CPU_only', 'not_a_GPU_qualification': True}
    vocab_size = 151936
    scores = {}
    max_scalar_error = Decimal(0)
    max_expression_error = Decimal(0)
    max_bound_error = Decimal(0)
    with localcontext() as ctx:
        ctx.prec = 60
        for i, c in enumerate(design['conditions']):
            yes = ((i * 7) % 43 - 21) / 16
            no = ((i * 11) % 47 - 23) / 16
            bound = (1 + i % 3) / 1024
            logits = [-10.0] * vocab_size
            logits[module.YES_ID], logits[module.NO_ID] = yes, no
            result = module.readout(logits, bound, ref)
            dy, dn = Decimal.from_float(yes), Decimal.from_float(no)
            normalizer = ((Decimal(vocab_size) - 2) * Decimal(-10).exp() + dy.exp() + dn.exp()).ln()
            expected = {'m': dn - dy, 'log_p_yes': dy - normalizer, 'log_p_no': dn - normalizer,
                'pair_support_no': Decimal(1) / (Decimal(1) + (dy - dn).exp()),
                'log_legal_mass': (dy.exp() + dn.exp()).ln() - normalizer,
                'legal_mass': (dy.exp() + dn.exp()) / normalizer.exp()}
            for key, value in expected.items():
                error = abs(Decimal.from_float(result[key]) - value)
                max_scalar_error = max(max_scalar_error, error)
                require(error < Decimal('1e-12'), f'Decimal scalar mismatch: {key}')
            require(abs(result['m'] - (result['log_p_no'] - result['log_p_yes'])) < 1e-10, 'margin identity mismatch')
            require(result['raw_prediction'] == ('无' if dn > dy else '有' if dn < dy else None), 'prediction differs')
            result.update(physical_score_id=f'SYNTHETIC-{i:03d}', prompt_sha256=c['prompt_sha256'])
            scores[c['condition_id']] = result
        for c in design['comparisons']:
            result = module.linear_effect(c['terms'], scores)
            expected = sum((Decimal(t['coefficient']) * Decimal.from_float(scores[t['condition_id']]['m']) for t in c['terms']), Decimal(0))
            bound = sum((abs(Decimal(t['coefficient'])) * Decimal.from_float(scores[t['condition_id']]['margin_error_bound']) for t in c['terms']), Decimal(0))
            error = abs(Decimal.from_float(result['value']) - expected)
            bound_error = abs(Decimal.from_float(result['bound']) - bound)
            max_expression_error = max(max_expression_error, error)
            max_bound_error = max(max_bound_error, bound_error)
            require(error < Decimal('1e-12') and bound_error < Decimal('1e-12'), 'expression/bound differs')
            state = 'positive' if expected > bound else 'negative' if expected < -bound else 'numerical_unresolved'
            require(result['resolution'] == state, 'expression resolution differs')

    logits = [-10.0] * vocab_size
    logits[module.YES_ID] = logits[module.NO_ID] = 0.0
    tied = module.readout(logits, 0.0, ref)
    require(tied['exact_tie'] and tied['raw_prediction'] is None, 'tie defaulted to a class')
    require(module.reference_status(tied, '无')['raw_correct'] is False, 'tie deleted from correctness denominator')
    require(tied['resolution'] == 'numerical_unresolved', 'bound equality resolved')
    unqualified = module.readout(logits)
    require(unqualified['resolution'] == 'unqualified' and module.reference_status(unqualified, '无')['conservative_correct'] is None, 'missing bound treated as zero')
    logits[module.NO_ID] = 0.25
    borderline = module.readout(logits, 0.25, ref)
    require(borderline['resolution'] == 'numerical_unresolved', 'exact bound edge resolved')
    logits[module.NO_ID] = 2.0
    correct = module.readout(logits, 0.25, ref)
    logits[module.NO_ID] = -2.0
    wrong = module.readout(logits, 0.25, ref)
    require(module.transition(wrong, correct, '无') == 'repair', 'repair criterion differs')
    require(module.transition(correct, wrong, '无') == 'damage', 'damage criterion differs')
    require(module.transition(borderline, correct, '无') == 'unresolved_transition', 'unresolved treated as repair')
    logits = [-10000.0] * vocab_size
    logits[module.YES_ID], logits[module.NO_ID] = 0.0, -10000.0
    extreme = module.readout(logits)
    require(extreme['pair_support_no'] == 0 and math.isfinite(extreme['log_legal_mass']), 'extreme logits overflow')
    logits = [0.0] * vocab_size
    logits[module.YES_ID], logits[module.NO_ID] = -20.0, -10.0
    weak_legal = module.readout(logits)
    require(weak_legal['pair_support_no'] > 0.99 and weak_legal['legal_mass'] < 1e-6, 'pair support confused with legal output mass')
    physical = {'physical_score_id': 'same-physical-request', 'prompt_sha256': 'synthetic',
                'm': 0.75, 'margin_error_bound': None, 'qualification_ref': None}
    cancellation = module.linear_effect([{'condition_id': 'a', 'coefficient': 1}, {'condition_id': 'alias_a', 'coefficient': -1}], {'a': physical, 'alias_a': physical})
    require(cancellation['value'] == cancellation['bound'] == 0 and cancellation['physical_terms'] == [], 'physical alias did not cancel before bound propagation')
    rejected = []
    for name, operation in [
        ('nonfinite_logit', lambda: module.readout([float('nan')] * vocab_size)),
        ('bound_without_receipt', lambda: module.readout(logits, 0.01)),
        ('negative_bound', lambda: module.readout(logits, -0.01, ref)),
        ('conflicting_physical_alias', lambda: module.linear_effect([{'condition_id': 'a', 'coefficient': 1}, {'condition_id': 'b', 'coefficient': -1}], {'a': physical, 'b': dict(physical, m=1.0)})),
    ]:
        try:
            operation()
        except ValueError:
            rejected.append(name)
        else:
            raise ValueError(f'invalid input was accepted: {name}')
    return {'synthetic_readouts': 120, 'decimal_precision': 60, 'scalar_checks': 720,
            'registered_expression_value_and_bound_checks': 252,
            'max_scalar_error': str(max_scalar_error), 'max_expression_error': str(max_expression_error),
            'max_bound_error': str(max_bound_error), 'semantic_edge_checks': 10, 'invalid_inputs_rejected': rejected,
            'synthetic_only': True, 'GPU_qualification': False}


def check():
    manifest = read(HERE / 'manifest.json')
    for e in manifest['artifacts'] + manifest['sources']:
        verify(e)
    plan = read(HERE / 'plan.json')
    require(not plan['GPU_job_started'] and not plan['model_forward_performed'], 'execution claimed')
    require(all(v is None for v in plan['runtime_binding'].values()), 'unverified runtime binding')
    for e in [plan['scientific_input_freeze'], plan['model_input_file'], plan['analysis_plan'], plan['qualification_plan'], plan['model']['model_config']]:
        verify(e)
    frozen_manifest = read(FROZEN / 'manifest.json')
    for e in frozen_manifest['artifacts'] + frozen_manifest['sources']:
        verify(e)
    design = read(FROZEN / 'design.json')
    previews = read(FROZEN / 'prompt-previews.json')['records']
    inputs = [json.loads(line) for line in (HERE / 'model-inputs.jsonl').read_text().splitlines()]
    allowed = {'request_id', 'condition_id', 'messages', 'chat_prompt', 'prompt_sha256', 'input_ids',
               'input_ids_sha256', 'prompt_tokens', 'last_input_token_index', 'next_token_position', 'candidate_tokens'}
    require(len(inputs) == len({r['request_id'] for r in inputs}) == 120, 'request inventory differs')
    for i, (r, p) in enumerate(zip(inputs, previews), 1):
        require(set(r) == allowed and r['request_id'] == f'CTEXEC-{i:03d}', 'extra scoring metadata or unstable identity')
        require(all(r[key] == p[key] for key in allowed - {'request_id'}), 'model-facing input changed')
    analysis = read(HERE / 'analysis-plan.json')
    require(analysis['comparisons'] == design['comparisons'] and len(analysis['references']) == 12, 'analysis frame differs')
    require(analysis['scorer_must_not_read_this_file'] and not analysis['aggregation']['rule_fit_stratification'], 'analysis scope differs')
    require(all(r['original_gold'] is None and r['original_correct'] is None for r in analysis['references']), 'invented original Gold')
    qplan = read(HERE / 'qualification-plan.json')
    require(len(qplan['passes']) == 5 and sum(p['prompt_forwards'] for p in qplan['passes']) == 600, 'engineering budget differs')
    require(plan['primary_budget']['total_prompt_forwards'] == 720 and plan['primary_budget']['candidate_values'] == 1440, 'candidate/forward budget confused')
    require(qplan['acceptance']['margin_error_bound_before_qualification'] is None, 'bound invented before GPU qualification')
    baseline = [c['condition_id'] for c in design['conditions'] if c['demo_arm'] == 'none' and c['lexicon_arm'] == 'absent']
    require(plan['generation_diagnostic']['condition_ids'] == baseline and len(baseline) == 12, 'generation subset changed')
    padded = [(len(r['input_ids']) // 16 + 1) * 16 for r in inputs]
    require(max(padded) == qplan['padding_contract']['max_padded_tokens'] == 672, 'padding envelope differs')
    require(all(n > len(r['input_ids']) for n, r in zip(padded, inputs)), 'padding variant not actually padded')
    return {'status': 'pass', 'model_visible_byte_and_token_bridges': 120,
            'scoring_input_extra_metadata_fields': 0, 'analysis_only_references': 12,
            'comparison_scope_counts': dict(Counter(c['scope'] for c in analysis['comparisons'])),
            'engineering_forward_budget': 600, 'science_forward_budget': 120,
            'generation_diagnostic_prompts': 12, 'arithmetic': arithmetic_audit(design),
            'weights_loaded': False, 'model_forward_performed': False, 'numerical_GPU_acceptance': False,
            'remaining_runtime_binding_explicit': True}


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, indent=2))
