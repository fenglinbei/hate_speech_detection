"""Q01-only activation-patching frame. CPU preparation; no model imports."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re

from diagnostics.general_model_evidence_evaluation import canonical, require, sha

QUERY = 'FD-3169-Q01'
LAYERS = (0, 5, 10, 15, 20, 25, 30, 34)
MAIN_ROLES = ('lexicon_end', 'demo_answer_end', 'query_hehe', 'pre_answer')
ROLES = MAIN_ROLES + ('unrelated_demo_end',)
VARIANTS = (('original', None), ('ab_forward', None), ('ab_reverse', None),
            *[('original', p) for p in ('empty', 'space', 'na', 'mask', 'lorem')])
DESCRIPTOR = ('record_id', 'query_id', 'task', 'condition', 'context_sha256',
              'prompt_sha256', 'prompt_tokens', 'prompt_token_ids_sha256')


def digest(value):
    return sha(canonical(value).encode())


def q01_bindings(context):
    return [b for b in context['bindings']
            if b['query_id'] == QUERY and b['lexicon_arm'] in ('O', 'N1', 'N2')]


def select_contexts(contexts):
    selected = [deepcopy(c) for c in contexts if q01_bindings(c)]
    index = {}
    for c in selected:
        require(len(q01_bindings(c)) == 1, 'ambiguous Q01 alias')
        b = q01_bindings(c)[0]
        key = (b['demo_family'], b['lexicon_arm'], b['demo_surface'], c['encoding'], c['probe_id'])
        require(key not in index, 'duplicated Q01 context')
        index[key] = c
    expected = {(f, lex, surface, enc, probe) for f in (1, 2) for lex in ('O', 'N1', 'N2')
                for surface in ('H', 'A') for enc, probe in VARIANTS}
    require(index.keys() == expected and len(selected) == 96, 'Q01 frame is not exactly 96 prompts')
    return selected, index


def locate(context, tokenizer):
    text = context['prompt_text']
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    require(ids == context['prompt_token_ids'], 'tokenizer replay differs')
    require(tokenizer.apply_chat_template(context['messages'], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) == text, 'chat template replay differs')

    def span(start, stop):
        result = [i for i, (a, b) in enumerate(offsets) if a < stop and b > start]
        require(result and result == list(range(result[0], result[-1] + 1)), 'non-contiguous/empty role')
        return result

    def row(positions, *, semantic_chars=None, absent=None):
        if absent:
            return {'applicable': False, 'reason': absent, 'token_positions': [],
                    'position_ids': [], 'token_ids': [], 'token_char_spans': [], 'semantic_char_span': None}
        require(all(type(i) is int and 0 <= i < len(ids) for i in positions), 'role exceeds prompt')
        return {'applicable': True, 'token_positions': positions, 'position_ids': positions,
                'token_ids': [ids[i] for i in positions],
                'token_char_spans': [list(offsets[i]) for i in positions],
                'semantic_char_span': semantic_chars,
                'token_text': [text[offsets[i][0]:offsets[i][1]] for i in positions]}

    lexicons = list(re.finditer(r'词条：', text))
    require(len(lexicons) == 6, 'dictionary slot inventory differs')
    lex_start, lex_stop = lexicons[2].start(), lexicons[3].start()
    lex = span(lex_start, lex_stop)
    require(len(lex) == 33, 'matched lexicon slot length differs')

    def demo_end(number):
        start = text.index(f'示例 {number}\n')
        match = re.search(r'输出：("[^"\n]+")', text[start:])
        require(match is not None, 'demo answer missing')
        a, b = start + match.start(1), start + match.end(1)
        return row(span(a, b)[-1:], semantic_chars=[a, b])

    marker = '待判断文本（JSON 字符串）：\n'
    require(text.count(marker) == 1, 'query marker is ambiguous')
    query_start = text.index(marker) + len(marker)
    if context['probe_id'] is None:
        start = text.index('嘿嘿', query_start)
        query_role = row(span(start, start + 2), semantic_chars=[start, start + 2])
    else:
        query_role = row([], absent='query_hehe is absent in this exact background probe')
    roles = {'lexicon_slot': row(lex, semantic_chars=[lex_start, lex_stop]),
             'lexicon_end': row(lex[-1:], semantic_chars=[lex_start, lex_stop]),
             'before_lexicon': row([lex[0] - 1]), 'demo_answer_end': demo_end(3),
             'unrelated_demo_end': demo_end(9), 'query_hehe': query_role,
             'pre_answer': row([len(ids) - 1])}
    if context['probe_id'] is None:
        expected = ([218, 415, 730, 745, 673, 185] if context['encoding'] == 'original'
                    else [234, 427, 732, 747, 676, 201])
        actual = [roles[r]['token_positions'][0] for r in
                  (*MAIN_ROLES, 'unrelated_demo_end', 'before_lexicon')]
        require(actual == expected, f'previously audited real-Q01 positions differ: {actual}')
    return {**{k: context[k] for k in DESCRIPTOR}, 'encoding': context['encoding'],
            'probe_id': context['probe_id'], 'binding': q01_bindings(context)[0],
            'index_convention': 'zero-based; character spans half-open; token positions explicit',
            'candidate_start': len(ids), 'first_prediction_position': len(ids) - 1,
            'roles': roles}


def pair_proof(left, right, positions):
    require(left['encoding'] == right['encoding'] and left['probe_id'] == right['probe_id'], 'cross encoding/probe pair')
    a, b = q01_bindings(left)[0], q01_bindings(right)[0]
    require(all(a[k] == b[k] for k in ('query_id', 'demo_family', 'demo_surface')), 'cross family/surface pair')
    require(left['prompt_tokens'] == right['prompt_tokens'], 'unequal prompt lengths')
    changed = [i for i, (x, y) in enumerate(zip(left['prompt_token_ids'], right['prompt_token_ids'], strict=True)) if x != y]
    slot = positions[left['record_id']]['roles']['lexicon_slot']['token_positions']
    require(changed and set(changed) <= set(slot), 'pair differs outside lexicon slot')
    for role in (*ROLES, 'before_lexicon', 'lexicon_slot'):
        require(positions[left['record_id']]['roles'][role]['token_positions'] ==
                positions[right['record_id']]['roles'][role]['token_positions'], 'external role geometry differs')
    rebuilt = list(left['prompt_token_ids'])
    for i in changed: rebuilt[i] = right['prompt_token_ids'][i]
    require(rebuilt == right['prompt_token_ids'], 'embedding reconstruction does not recreate donor tokens')
    return {'left': left['record_id'], 'right': right['record_id'], 'changed_positions': changed,
            'outside_slot_identical': True, 'whole_prompt_reconstructed': True,
            'internal_semantic_alignment_claimed': False}


def build_requests(contexts, index, positions):
    """Every physical request is explicit. Repeated scientific aliases are not samples."""
    requests, proofs = [], []

    def add(kind, recipient, donor=None, *, layer=None, role=None, module='block', category,
            group=None, direction=None, expected=None, token_positions=None):
        role_positions = token_positions
        if role_positions is None and role:
            role_positions = positions[recipient['record_id']]['roles'][role]['token_positions']
        r = {'kind': kind, 'category': category, 'recipient': recipient['record_id'],
             'donor': donor['record_id'] if donor else None, 'module': module if role else None,
             'layer': layer, 'role': role, 'positions': role_positions or [],
             'encoding': recipient['encoding'], 'probe_id': recipient['probe_id'],
             'family': q01_bindings(recipient)[0]['demo_family'],
             'surface': q01_bindings(recipient)[0]['demo_surface'],
             'group': group, 'direction': direction,
             'expected_baseline': expected['record_id'] if expected else None}
        r['request_id'] = 'QMP-' + digest(r)
        requests.append(r)

    for c in contexts:
        add('baseline', c, category='baseline')
        add('capture_only', c, category='engineering', expected=c)
        for layer in LAYERS:
            for role in ROLES:
                if not positions[c['record_id']]['roles'][role]['applicable']: continue
                add('self_patch', c, c, layer=layer, role=role, category='engineering', expected=c)
        add('self_patch', c, c, layer=35, role='pre_answer', category='engineering', expected=c)
        add('self_patch', c, c, role='lexicon_slot', module='embedding', category='engineering', expected=c)

    for family in (1, 2):
        for pair in (('O', 'N1'), ('O', 'N2'), ('N1', 'N2')):
            neutral = pair[0] == 'N1'
            group = f'F{family}:{pair[0]}-{pair[1]}'
            for surface in ('H', 'A'):
                for enc, probe in VARIANTS:
                    left, right = (index[family, lex, surface, enc, probe] for lex in pair)
                    proof = pair_proof(left, right, positions); proofs.append(proof)
                    for recipient, donor, direction in ((left, right, 'R'), (right, left, 'K')):
                        for layer in LAYERS:
                            for role in ROLES:
                                if not positions[recipient['record_id']]['roles'][role]['applicable']: continue
                                category = 'neutral_control' if neutral else ('site_control' if role == 'unrelated_demo_end' else 'primary')
                                add('patch', recipient, donor, layer=layer, role=role, category=category,
                                    group=group, direction=direction)
                        if neutral: continue
                        add('patch', recipient, donor, layer=35, role='pre_answer', category='output_diagnostic',
                            group=group, direction=direction)
                        add('embedding_reconstruction', recipient, donor, role='lexicon_slot', module='embedding',
                            category='engineering', group=group, direction=direction, expected=donor,
                            token_positions=proof['changed_positions'])
                        for layer in LAYERS:
                            add('identical_prefix', recipient, donor, layer=layer, role='before_lexicon',
                                category='engineering', group=group, direction=direction, expected=recipient)
                        terminal_roles = ('lexicon_end',) + (() if probe else ('query_hehe',))
                        for role in terminal_roles:
                            add('terminal_nonpropagation', recipient, donor, layer=35, role=role,
                                category='engineering', group=group, direction=direction, expected=recipient)
    require(len({r['request_id'] for r in requests}) == len(requests), 'duplicate physical request')
    require(sum(r['category'] == 'primary' for r in requests) == 3456, 'primary frame count differs')
    require(len(proofs) == 96, 'pair proof coverage differs')
    return requests, proofs


def capture_specs(position):
    result = []
    for layer in LAYERS:
        for role in (*ROLES, 'before_lexicon'):
            if position['roles'][role]['applicable']:
                result.append({'module': 'block', 'layer': layer, 'role': role,
                               'positions': position['roles'][role]['token_positions']})
    for role in ('lexicon_end', 'pre_answer', 'query_hehe'):
        if position['roles'][role]['applicable']:
            result.append({'module': 'block', 'layer': 35, 'role': role,
                           'positions': position['roles'][role]['token_positions']})
    result.append({'module': 'embedding', 'layer': None, 'role': 'lexicon_slot',
                   'positions': position['roles']['lexicon_slot']['token_positions']})
    return result


def schedule(requests, contexts, catalog):
    passes = []
    for phase in ('engineering', 'science'):
        selected = [r for r in requests if r['category'] == 'baseline' or
                    ((r['category'] == 'engineering') == (phase == 'engineering'))]
        for mode in ('reference', 'repeat', 'padding', 'prefix', 'member_order', 'replica'):
            passes.append({'pass_id': f'{phase}-{mode}', 'phase': phase, 'mode': mode,
                           'request_ids': [r['request_id'] for r in selected],
                           'candidate_evaluations': 2 * len(selected),
                           'options': {'reference': mode == 'reference', 'padding_extra': 64 if mode == 'padding' else 0,
                                       'prefix': mode == 'prefix', 'permuted': mode == 'member_order',
                                       'replica_shift': 1 if mode == 'replica' else 0},
                           'gate_before_next_pass': True})
    # Prefix kernels share the empty answer prefix; candidate counts are not forward counts.
    by_id = {c['record_id']: c for c in contexts}; by_request = {r['request_id']: r for r in requests}
    for p in passes:
        count = 0
        for rid in p['request_ids']:
            c = by_id[by_request[rid]['recipient']]
            prefixes = {tuple(a['answer_token_ids'][:i]) for a in catalog[c['task']]
                        for i in range(a['answer_tokens'] + 1)}
            count += len(prefixes) if p['options']['prefix'] else 2
        p['scoring_forward_calls'] = count
        # A cold cache per pass/producer is required, including repeats and replica checks.
        p['capture_forward_bound_per_worker'] = len(contexts)
    return passes


def budget(requests, passes):
    return {'source_prompts': 96, 'primary_prompt_requests': 3456, 'primary_candidates_one_pass': 6912,
            'requests_by_category': dict(sorted(Counter(r['category'] for r in requests).items())),
            'unique_requests': len(requests), 'passes': len(passes),
            'scheduled_candidate_evaluations': sum(p['candidate_evaluations'] for p in passes),
            'scoring_forward_calls': sum(p['scoring_forward_calls'] for p in passes),
            'donor_capture_forward_upper_bound_per_gpu': len(passes) * 96,
            'donor_and_capture_only_sharing': 'one prompt-only capture per source, pass and physical producer',
            'refinement_included': False}
