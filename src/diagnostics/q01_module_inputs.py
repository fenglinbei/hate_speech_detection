"""Separately frozen six-module refinement of the selected Q01 pre-answer site."""
from collections import Counter

from diagnostics.general_model_evidence_evaluation import require
from diagnostics.q01_mechanism_inputs import (
    digest, q01_bindings, pair_proof, VARIANTS, schedule, DESCRIPTOR, locate, select_contexts)

LAYERS = (33, 34, 35)
MODULES = ('attention', 'mlp')
UNITS = tuple((layer, module) for layer in LAYERS for module in MODULES)
TEST_ROLES = ('pre_answer', 'unrelated_demo_end')


def capture_specs(position):
    specs = []
    for layer, module in UNITS:
        roles = (*TEST_ROLES, 'before_lexicon')
        if layer == 35:
            roles += ('lexicon_end', 'query_hehe')
        for role in roles:
            r = position['roles'][role]
            if r['applicable']:
                specs.append({'module': module, 'layer': layer, 'role': role,
                              'positions': r['token_positions']})
    for layer in (34, 35):
        specs.append({'module': 'block', 'layer': layer, 'role': 'pre_answer',
                      'positions': position['roles']['pre_answer']['token_positions']})
    specs.append({'module': 'embedding', 'layer': None, 'role': 'lexicon_slot',
                  'positions': position['roles']['lexicon_slot']['token_positions']})
    return specs


def build_requests(contexts, index, positions):
    requests, proofs = [], []

    def add(kind, recipient, donor=None, *, layer=None, module=None, role=None,
            category, group=None, direction=None, expected=None, changed=None):
        b = q01_bindings(recipient)[0]
        r = {'kind': kind, 'category': category, 'recipient': recipient['record_id'],
             'donor': donor['record_id'] if donor else None, 'module': module,
             'layer': layer, 'role': role,
             'positions': changed if changed is not None else
                 (positions[recipient['record_id']]['roles'][role]['token_positions'] if role else []),
             'encoding': recipient['encoding'], 'probe_id': recipient['probe_id'],
             'family': b['demo_family'], 'surface': b['demo_surface'], 'group': group,
             'direction': direction, 'expected_baseline': expected['record_id'] if expected else None}
        r['request_id'] = 'QMP-' + digest(r)
        requests.append(r)

    for c in contexts:
        add('baseline', c, category='baseline')
        add('capture_only', c, category='engineering', expected=c)
        for layer, module in UNITS:
            for role in TEST_ROLES:
                add('self_patch', c, c, layer=layer, module=module, role=role,
                    category='engineering', expected=c)
        for layer in (34, 35):
            add('self_patch', c, c, layer=layer, module='block', role='pre_answer',
                category='engineering', expected=c)
        add('self_patch', c, c, module='embedding', role='lexicon_slot',
            category='engineering', expected=c)

    for family in (1, 2):
        for pair in (('O', 'N1'), ('O', 'N2'), ('N1', 'N2')):
            neutral = pair[0] == 'N1'
            group = f'F{family}:{pair[0]}-{pair[1]}'
            for surface in ('H', 'A'):
                for enc, probe in VARIANTS:
                    left, right = (index[family, lex, surface, enc, probe] for lex in pair)
                    proof = pair_proof(left, right, positions)
                    proofs.append(proof)
                    for recipient, donor, direction in ((left, right, 'R'), (right, left, 'K')):
                        for layer, module in UNITS:
                            for role in TEST_ROLES:
                                category = ('neutral_control' if role == 'pre_answer' else 'neutral_site_control') if neutral else (
                                    'primary' if role == 'pre_answer' else 'site_control')
                                add('patch', recipient, donor, layer=layer, module=module, role=role,
                                    category=category, group=group, direction=direction)
                        add('patch', recipient, donor, layer=34, module='block', role='pre_answer',
                            category='neutral_benchmark' if neutral else 'block_benchmark', group=group, direction=direction)
                        if neutral:
                            continue
                        add('patch', recipient, donor, layer=35, module='block', role='pre_answer',
                            category='output_diagnostic', group=group, direction=direction)
                        add('embedding_reconstruction', recipient, donor, module='embedding', role='lexicon_slot',
                            category='engineering', group=group, direction=direction,
                            expected=donor, changed=proof['changed_positions'])
                        for layer, module in UNITS:
                            add('identical_prefix', recipient, donor, layer=layer, module=module,
                                role='before_lexicon', category='engineering', group=group,
                                direction=direction, expected=recipient)
                        # No later attention follows these final-layer module outputs.
                        # MLP, residual additions, normalization and lm_head are token-local.
                        # This guarantee DOES NOT apply to layers 33 or 34.
                        for module in MODULES:
                            for role in ('lexicon_end',) + (() if probe is not None else ('query_hehe',)):
                                add('terminal_nonpropagation', recipient, donor, layer=35, module=module,
                                    role=role, category='engineering', group=group, direction=direction,
                                    expected=recipient)
    require(len(requests) == len({r['request_id'] for r in requests}), 'duplicate module request')
    require(sum(r['category'] == 'primary' for r in requests) == 768, 'six-module primary frame differs')
    require(len(proofs) == 96, 'matched-pair proof frame differs')
    return requests, proofs


def budget(requests, passes):
    categories = Counter(r['category'] for r in requests)
    return {'source_prompts': 96, 'primary_units': 6, 'primary_prompt_requests': categories['primary'],
            'primary_candidates_one_pass': 2 * categories['primary'],
            'requests_by_category': dict(sorted(categories.items())), 'unique_requests': len(requests),
            'passes': len(passes), 'scheduled_candidate_evaluations': sum(p['candidate_evaluations'] for p in passes),
            'scoring_forward_calls': sum(p['scoring_forward_calls'] for p in passes),
            'engineering_candidate_evaluations': sum(p['candidate_evaluations'] for p in passes if p['phase'] == 'engineering'),
            'science_candidate_evaluations': sum(p['candidate_evaluations'] for p in passes if p['phase'] == 'science'),
            'donor_capture_forward_upper_bound_per_gpu': len(passes) * 96,
            'donor_and_capture_only_sharing': 'one prompt-only capture per source, pass and physical producer; resume adds cold captures',
            'refinement_included': True, 'new_independent_queries': 0}
