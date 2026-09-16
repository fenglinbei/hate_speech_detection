"""CPU-only paired evaluation against immutable original and reviewed references.

No model, tokenizer, server, or runtime loader is imported. Raw score metadata is
scanned across the sealed file; candidate payloads are decoded only for discovery.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import tempfile

from diagnostics.general_model_numeric_analysis import (
    GROUP_LABELS, SCORE_MODES, _gold_ordinal, _validated_candidates, candidate_catalog,
)

TASKS = ('hate', 'group')
CONDITIONS = ('C0', 'CLnew', 'CD', 'CLDnew', 'CLnewNoCat', 'CLDnewNoCat')
CORE = ('C0', 'CLnewNoCat', 'CD', 'CLDnewNoCat')
PAIRS = {
    'remove_with_D': ('CLDnew', 'CLDnewNoCat'),
    'remove_without_D': ('CLnew', 'CLnewNoCat'),
    'S_given_D': ('CD', 'CLDnewNoCat'),
    'S_vs_0': ('C0', 'CLnewNoCat'),
    'D_vs_0': ('C0', 'CD'),
    'D_given_S': ('CLnewNoCat', 'CLDnewNoCat'),
}
REFS = ('original', 'reviewed')
EXPERIMENT = Path('exps/causal_context/general_model_evidence_applicability_v1')
META_ID = re.compile(rb'"query_id"\s*:\s*"([^"\\]+)"')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def jsonl(rows):
    return ''.join(canonical(row) + '\n' for row in rows).encode()


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def loads(raw):
    def bad_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=bad_constant)


def read_json(path):
    return loads(Path(path).read_bytes())


def read_lines(path):
    return [loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def local_path(root, value):
    rel = Path(value)
    require(not rel.is_absolute() and '..' not in rel.parts, 'path must be repository relative: ' + value)
    path = (root / rel).resolve()
    require(path.is_relative_to(root), 'source escapes repository')
    return path


class Sources:
    def __init__(self, root):
        self.root = root
        self.hashes = {}

    def verify(self, path, expected):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root), 'source outside repository')
        require(file_sha(path) == expected, 'source hash mismatch: ' + str(path))
        self.hashes[str(path.relative_to(self.root))] = expected
        return path

    def ref(self, ref):
        return self.verify(local_path(self.root, ref['path']), ref['sha256'])

    def unchanged(self):
        for path, expected in self.hashes.items():
            require(file_sha(self.root / path) == expected, 'source changed during evaluation: ' + path)


def unique(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        require(identity not in result, 'duplicate row: ' + str(identity))
        result[identity] = row
    return result


def label(task, value, nullable=False):
    require(task in TASKS, 'unsupported task')
    if value is None and nullable:
        return None
    _gold_ordinal(task, value)
    if task == 'group':
        require(value == [v for v in GROUP_LABELS if v in value], 'noncanonical group order')
    return value


def prediction_label(task, labels):
    if task == 'hate':
        require(isinstance(labels, list) and len(labels) == 1, 'invalid hate prediction')
        return label(task, labels[0])
    return label(task, labels)


def ranking(candidates, epsilon, mode='answer_sum'):
    ordered = sorted(candidates, key=lambda c: (-c['scores'][mode], c['ordinal']))
    best = ordered[0]
    gap = best['scores'][mode] - ordered[1]['scores'][mode]
    return {'labels': best['labels'], 'ordinal': best['ordinal'], 'top_score_gap': gap,
            'tied_top_count': sum(c['scores'][mode] == best['scores'][mode] for c in candidates),
            'within_two_epsilon': gap <= 2 * epsilon}


def margin(task, candidates, gold):
    if candidates is None or gold is None:
        return None
    ordinal = _gold_ordinal(task, gold)
    scores = [c['scores']['answer_sum'] for c in candidates]
    return scores[ordinal] - max(v for i, v in enumerate(scores) if i != ordinal)


def transition(before, after):
    if before is None or after is None:
        return None
    return ('right' if before else 'wrong') + '_to_' + ('right' if after else 'wrong')


def f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def metrics(task, predictions, golds):
    require(len(predictions) == len(golds), 'unpaired metrics')
    if not golds:
        return {'n': 0, 'exact_accuracy': None, 'macro_f1': None, 'micro_f1': None, 'per_label': []}
    space = ('hate', 'non-hate') if task == 'hate' else GROUP_LABELS
    pred = [{v} for v in predictions] if task == 'hate' else [set(v) for v in predictions]
    gold = [{v} for v in golds] if task == 'hate' else [set(v) for v in golds]
    rows = []
    for v in space:
        tp = sum(v in p and v in g for p, g in zip(pred, gold))
        fp = sum(v in p and v not in g for p, g in zip(pred, gold))
        fn = sum(v not in p and v in g for p, g in zip(pred, gold))
        rows.append({'label': v, 'tp': tp, 'fp': fp, 'fn': fn, 'support': tp + fn, 'f1': f1(tp, fp, fn)})
    return {'n': len(gold), 'exact_accuracy': sum(p == g for p, g in zip(pred, gold)) / len(gold),
            'macro_f1': math.fsum(r['f1'] for r in rows) / len(rows),
            'micro_f1': f1(sum(r['tp'] for r in rows), sum(r['fp'] for r in rows), sum(r['fn'] for r in rows)),
            'per_label': rows}


def pattern_flags(mask):
    if mask is None:
        return []
    return [name for name, active in (
        ('stable_correct', mask == '1111'), ('stable_wrong', mask == '0000'),
        ('joint_only', mask == '0001'), ('S_and_D_each_correct', mask[1:3] == '11'),
        ('SD_loses_D_correctness', mask[2:] == '10'), ('SD_gains_over_D', mask[2:] == '01'),
        ('SD_loses_S_correctness', mask[1] == '1' and mask[3] == '0')) if active]


def validate_config(config):
    expected = {'schema_version': 'evidence-dual-reference-evaluation-config/v1',
                'analysis_kind': 'posthoc-descriptive-discovery-subset',
                'tasks': list(TASKS), 'conditions': list(CONDITIONS), 'core_conditions': list(CORE),
                'prediction_score': 'answer_sum', 'tie_rule': 'smallest-canonical-ordinal',
                'bootstrap_replicates': 0, 'zero_division': 0, 'query_count': 32}
    for k, v in expected.items():
        require(config.get(k) == v, 'unsupported evaluation contract: ' + k)
    for k in ('allow_model_forward', 'allow_gpu', 'allow_test_or_reserve_analysis'):
        require(config.get(k) is False, 'CPU-only boundary violated: ' + k)
    require(type(config['epsilon']) in (int, float) and math.isfinite(config['epsilon']) and config['epsilon'] >= 0, 'invalid epsilon')


def load_inputs(root, config_path, allow_missing_scores=False):
    sources = Sources(root)
    config = read_json(config_path)
    validate_config(config)
    sources.verify(config_path, file_sha(config_path))
    sources.ref(config['candidate_math'])
    pointer_path = sources.ref(config['reference_pointer'])
    pointer = read_json(pointer_path)
    require(pointer['status'] == 'frozen', 'reference pointer is not frozen')
    freeze = local_path(root, str(EXPERIMENT / pointer['freeze_path']))
    fm = read_json(sources.verify(freeze / 'manifest.json', pointer['manifest_sha256']))
    require(fm['status'] == 'frozen' and fm['schema_version'] == 'evidence-analysis-reference-freeze/v1', 'invalid reference freeze')
    require(all(fm[k] == pointer[k] for k in ('session_sha256', 'session_revision')), 'reference/session binding mismatch')
    for name, h in fm['artifacts'].items():
        require(Path(name).name == name, 'invalid frozen artifact path')
        sources.verify(freeze / name, h)
    overlays = unique(read_lines(freeze / 'gold_overlay.jsonl'), lambda r: (r['query_id'], r['task']))
    eligible = unique(read_lines(freeze / 'eligibility.jsonl'), lambda r: (r['query_id'], r['task']))
    pm_path = sources.ref(config['paired_manifest'])
    paired = read_json(pm_path)
    pub = read_json(sources.ref(config['paired_export_manifest']))
    require(paired['status'] == 'complete' and pub['source_manifest_sha256'] == config['paired_manifest']['sha256']
            and pub['source_identity'] == paired['identity'], 'paired export identity mismatch')
    paired_root = pm_path.parent
    for name in ('cases/cards_index.json', 'config.frozen.json'):
        sources.verify(paired_root / name, pub['artifacts'][name])
    old_config = read_json(paired_root / 'config.frozen.json')
    require(old_config['epsilon'] == config['epsilon'] and old_config['conditions'] == list(CONDITIONS)
            and old_config['prediction_score'] == 'answer_sum', 'historical scoring contract mismatch')
    index = read_json(paired_root / 'cases/cards_index.json')
    cards = {}
    for row in index:
        q = row['query_id']
        require(q not in cards, 'duplicate discovery ID')
        rel = row['resources_card'].replace('/cards/', '/card_data/').replace('-1-resources.md', '.json')
        require(rel.startswith('cases/card_data/') and '..' not in Path(rel).parts, 'invalid card path')
        c = read_json(sources.verify(paired_root / rel, pub['artifacts'][rel]))
        require(str(c['query']['id']) == q and c['selection']['split'] == 'discovery', 'non-discovery or mismatched card')
        require(set(c['profile']['conditions']) == set(CONDITIONS), 'missing historical conditions')
        require(len(c['contexts']) == 12 and {(v['task'], v['condition']) for v in c['contexts']} == set(itertools.product(TASKS, CONDITIONS)), 'context matrix mismatch')
        for ctx in c['contexts']:
            require(sha(ctx['prompt_text'].encode()) == ctx['prompt_sha256'], 'prompt text mismatch')
        cards[q] = c
    require(len(cards) == config['query_count'], 'discovery count mismatch')
    expected = set(itertools.product(cards, TASKS))
    require(set(overlays) == set(eligible) == expected, 'reference task frame differs from discovery')
    for key, r in overlays.items():
        q, task = key
        require(r['session_sha256'] == fm['session_sha256'] and r['session_revision'] == fm['session_revision'], 'mixed session references')
        require(r['source_manifest_sha256'] == config['paired_manifest']['sha256'], 'reference refers to different experiment')
        require(r['text_sha256'] == sha(cards[q]['query']['content'].encode()), 'review/query text mismatch')
        require(label(task, r['original_label']) == cards[q]['query']['projection'][task], 'original Gold mismatch')
        label(task, r['adjudicated_label'], nullable=True)
        e = eligible[key]
        require(e['session_sha256'] == r['session_sha256'] and e['text_sha256'] == r['text_sha256'], 'eligibility binding mismatch')
        require(type(e['reference_analysis_eligible']) is bool and e['reference_analysis_eligible'] == r['analysis_reference_eligible'], 'eligibility disagreement')
        if e['reference_analysis_eligible']:
            require(r['adjudicated_label'] is not None and e['native_eligibility']['reference_eligible']
                    and e['native_eligibility']['status'] == 'confirmed'
                    and e['native_eligibility']['availability'] == 'resolved'
                    and not e['native_eligibility']['stale']
                    and not e['unavailable_reasons'], 'eligible reference is unresolved')
    for name, old_name in (('raw_scores', 'raw'), ('raw_manifest', 'raw_manifest')):
        require(config[name]['sha256'] == paired['source_receipt'][old_name]['sha256'], 'raw input not bound to historical source')
    raw_path = local_path(root, config['raw_scores']['path'])
    if not raw_path.exists():
        require(allow_missing_scores, 'candidate_scores_missing; pass --allow-missing-scores for discrete-only evaluation')
        raw = None
    else:
        require(raw_path.is_file(), 'raw path is not a file')
        raw = sources.ref(config['raw_scores'])
        rm = read_json(sources.ref(config['raw_manifest']))
        require(rm['status'] == 'complete' and rm['scores_sha256'] == config['raw_scores']['sha256']
                and rm['identity']['plan_id'] == old_config['plan_id'] and rm['blocks'] == 7716
                and rm['candidates'] == 131172, 'raw run is incomplete or incompatible')
    return sources, config, pointer, fm, cards, overlays, eligible, raw, old_config


def selected_raw(path, query_ids, expected_plan_id):
    indexed = {}
    if path is None:
        return indexed
    with path.open('rb') as stream:
        for line in stream:
            ids = META_ID.findall(line)
            require(len(ids) == 1, 'raw row query metadata is ambiguous')
            q = ids[0].decode('utf-8')
            if q not in query_ids:
                continue
            b = loads(line)
            task, condition = b['task'], b['condition']
            key = q, task, condition
            require(task in TASKS and condition in CONDITIONS and key not in indexed, 'duplicate or unexpected selected raw block')
            require(b['record_id'] == ':'.join(key) and b['plan_id'] == expected_plan_id
                    and b['pass_name'] == 'dev-b1' and b['repetition'] == 0, 'raw block source identity mismatch')
            _validated_candidates(task, b['candidates'])
            indexed[key] = b
    require(set(indexed) == set(itertools.product(query_ids, TASKS, CONDITIONS)), 'selected raw matrix incomplete')
    return indexed


def same_number(actual, expected, name):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), 'historical readout mismatch: ' + name)


def evaluate_blocks(cards, overlays, eligible, raw, epsilon):
    result = []
    for q, task, condition in itertools.product(cards, TASKS, CONDITIONS):
        r, e = overlays[q, task], eligible[q, task]
        saved = cards[q]['profile']['conditions'][condition][task]
        pred = saved['prediction']
        ctx = next(v for v in cards[q]['contexts'] if v['task'] == task and v['condition'] == condition)
        block = raw.get((q, task, condition))
        candidates = block['candidates'] if block is not None else None
        modes = saved['score_mode_predictions']
        require(set(modes) == set(SCORE_MODES), 'incomplete historical score modes')
        sensitive = len({canonical(value) for value in modes.values()}) > 1
        require(sensitive == saved['score_mode_sensitive'], 'score-mode sensitivity mismatch')
        if candidates is not None:
            for field in ('context_sha256', 'prompt_sha256'):
                require(block[field] == ctx[field] == saved['context'][field], 'historical prompt binding mismatch')
            for candidate in candidates:
                require(candidate['prompt_token_ids_sha256'] == ctx['prompt_token_ids_sha256']
                        and candidate['prompt_tokens'] == ctx['prompt_tokens'], 'raw token context mismatch')
            computed = ranking(candidates, epsilon)
            require(all(computed[k] == pred[k] for k in ('labels','ordinal','tied_top_count','within_two_epsilon')), 'historical prediction mismatch')
            same_number(computed['top_score_gap'], pred['top_score_gap'], 'top gap')
            computed_modes = {mode: ranking(candidates, epsilon, mode)['labels'] for mode in SCORE_MODES}
            require(computed_modes == modes, 'score-mode predictions mismatch')
            pred = computed
        prediction = prediction_label(task, pred['labels'])
        old, reviewed = r['original_label'], r['adjudicated_label']
        old_correct = prediction == old
        require(old_correct == saved['correct'], 'historical correctness mismatch')
        usable = e['reference_analysis_eligible']
        new_correct = prediction == reviewed if usable else None
        old_margin = margin(task, candidates, old)
        new_margin = margin(task, candidates, reviewed) if usable else None
        if old_margin is not None:
            same_number(old_margin, saved['readouts']['answer_sum/gold/best_nongold_margin'], 'original margin')
        result.append({'query_id': q, 'task': task, 'condition': condition,
                       'prediction': prediction, 'prediction_ordinal': pred['ordinal'],
                       'original_label': old, 'reviewed_label': reviewed, 'reference_eligible': usable,
                       'query_material_label': r['query_material_label'],
                       'review_status': e['native_eligibility']['status'],
                       'review_availability': e['native_eligibility']['availability'],
                       'unavailable_reasons': e['unavailable_reasons'],
                       'original_correct': old_correct, 'reviewed_correct': new_correct,
                       'correctness_relabel_transition': transition(old_correct, new_correct),
                       'original_margin': old_margin, 'revised_margin': new_margin,
                       'original_margin_archived': saved['readouts']['answer_sum/gold/best_nongold_margin'],
                       'continuous_status': 'candidate_scores_missing' if candidates is None else 'available' if usable else 'reference_unavailable',
                       'candidate_count': len(candidates) if candidates else None,
                       'candidate_scores': [c['scores']['answer_sum'] for c in candidates] if candidates else None,
                       'hate_score_direction': (candidates[0]['scores']['answer_sum'] - candidates[1]['scores']['answer_sum']) if candidates and task == 'hate' else None,
                       'top_gap': pred['top_score_gap'], 'tied_top_count': pred['tied_top_count'],
                       'within_two_epsilon': pred['within_two_epsilon'],
                       'score_mode_predictions': modes, 'score_mode_sensitive': saved['score_mode_sensitive'],
                       'prompt_tokens': ctx['prompt_tokens'], 'context_sha256': ctx['context_sha256'],
                       'prompt_sha256': ctx['prompt_sha256'], 'text_sha256': r['text_sha256'],
                       'native_case_policy': r['native_case_policy'], 'query_material_policy': r['query_material_policy'],
                       'original_status': r['original_status'], 'original_bucket': e['original_bucket']})
    return result


def aggregate(blocks, order):
    lookup = unique(blocks, lambda b: (b['query_id'], b['task'], b['condition']))
    require(set(lookup) == set(itertools.product(order, TASKS, CONDITIONS)), 'evaluation matrix incomplete')
    coverage, metric_rows, per_label = [], [], []
    masks, pairs, contrasts = [], [], []
    for task in TASKS:
        frame = [lookup[q, task, CONDITIONS[0]] for q in order]
        eligible_ids = [q for q in order if lookup[q, task, CONDITIONS[0]]['reference_eligible']]
        progress = Counter(b['review_status'] for b in frame)
        require(set(progress) <= {'confirmed', 'draft', 'unreviewed'}, 'unknown review status')
        coverage.append({'task': task, 'queue_n': len(order), 'paired_reference_n': len(eligible_ids),
                         'confirmed_n': progress['confirmed'], 'draft_n': progress['draft'],
                         'unreviewed_n': progress['unreviewed'],
                         'unresolved_n': sum(b['review_availability'] == 'unresolved' for b in frame),
                         'label_changed_n': sum(b['original_label'] != b['reviewed_label'] for b in frame if b['reference_eligible']),
                         'excluded_n': len(order) - len(eligible_ids), 'eligible_query_ids': eligible_ids,
                         'excluded': [{'query_id': q, 'reasons': lookup[q, task, CONDITIONS[0]]['unavailable_reasons']}
                                      for q in order if q not in eligible_ids]})
        for ref, condition in itertools.product(REFS, CONDITIONS):
            bs = [lookup[q, task, condition] for q in eligible_ids]
            m = metrics(task, [b['prediction'] for b in bs], [b[ref + '_label'] for b in bs])
            detail = m.pop('per_label')
            metric_rows.append({'reference': ref, 'task': task, 'condition': condition, **m})
            per_label.extend({'reference': ref, 'task': task, 'condition': condition, **d} for d in detail)
        for q in order:
            old = ''.join(str(int(lookup[q, task, c]['original_correct'])) for c in CORE)
            reviewed = ''.join(str(int(lookup[q, task, c]['reviewed_correct'])) for c in CORE) if q in eligible_ids else None
            b = lookup[q, task, CORE[0]]
            masks.append({'query_id': q, 'task': task, 'reference_eligible': q in eligible_ids,
                          'original_label': b['original_label'], 'reviewed_label': b['reviewed_label'],
                          'query_material_label': b['query_material_label'], 'original_status': b['original_status'],
                          'core_order': list(CORE), 'original_mask': old, 'reviewed_mask': reviewed,
                          'mask_changed': old != reviewed if reviewed is not None else None,
                          'original_flags': pattern_flags(old), 'reviewed_flags': pattern_flags(reviewed),
                          'original_bucket': b['original_bucket'], 'input_control_eligible': False,
                          'explanation_status': 'behavior_description_only'})
            for name, (before, after) in PAIRS.items():
                a, b = lookup[q, task, before], lookup[q, task, after]
                pair = {'query_id': q, 'task': task, 'contrast': name, 'before': before, 'after': after,
                        'reference_eligible': q in eligible_ids,
                        'original_transition': transition(a['original_correct'], b['original_correct']),
                        'reviewed_transition': transition(a['reviewed_correct'], b['reviewed_correct']),
                        'original_margin_delta': b['original_margin'] - a['original_margin'] if a['original_margin'] is not None else None,
                        'reviewed_margin_delta': b['revised_margin'] - a['revised_margin'] if a['revised_margin'] is not None else None,
                        'score_mode_sensitive': a['score_mode_sensitive'] or b['score_mode_sensitive']}
                pair['transition_changed'] = pair['original_transition'] != pair['reviewed_transition'] if q in eligible_ids else None
                pairs.append(pair)
            for ref, field in (('original', 'original_margin'), ('reviewed', 'revised_margin')):
                vals = {c: lookup[q, task, c][field] for c in CORE}
                contrasts.append({'query_id': q, 'task': task, 'reference': ref,
                                  'score_scale': 'reference_best_nongold_margin', 'values': vals,
                                  'SD_minus_S_minus_D_plus_0': math.fsum([vals[CORE[3]], -vals[CORE[1]], -vals[CORE[2]], vals[CORE[0]]]) if all(v is not None for v in vals.values()) else None})
    mask_counts, mask_changes, transition_counts, contrast_metrics, relabel_counts = [], [], [], [], []
    metric_index = {(r['reference'], r['task'], r['condition']): r for r in metric_rows}
    for task in TASKS:
        eligible_masks = [m for m in masks if m['task'] == task and m['reference_eligible']]
        for condition in CONDITIONS:
            bs = [b for b in blocks if b['task'] == task and b['condition'] == condition and b['reference_eligible']]
            counts = Counter(b['correctness_relabel_transition'] for b in bs)
            relabel_counts.append({'task': task, 'condition': condition, 'n': len(bs),
                                   **{t: counts[t] for t in ('wrong_to_wrong', 'wrong_to_right', 'right_to_wrong', 'right_to_right')}})
        for ref, i in itertools.product(REFS, range(16)):
            mask_counts.append({'task': task, 'reference': ref, 'mask': f'{i:04b}', 'n': len(eligible_masks),
                                'count': sum(m[ref + '_mask'] == f'{i:04b}' for m in eligible_masks)})
        for i, j in itertools.product(range(16), repeat=2):
            mask_changes.append({'task': task, 'original_mask': f'{i:04b}', 'reviewed_mask': f'{j:04b}',
                                 'count': sum(m['original_mask'] == f'{i:04b}' and m['reviewed_mask'] == f'{j:04b}' for m in eligible_masks)})
        for name, (before, after) in PAIRS.items():
            subset = [p for p in pairs if p['task'] == task and p['contrast'] == name and p['reference_eligible']]
            for ref in REFS:
                counts = Counter(p[ref + '_transition'] for p in subset)
                transition_counts.append({'task': task, 'contrast': name, 'reference': ref, 'n': len(subset),
                                          **{t: counts[t] for t in ('wrong_to_wrong','wrong_to_right','right_to_wrong','right_to_right')}})
                a, b = metric_index[ref, task, before], metric_index[ref, task, after]
                contrast_metrics.append({'task': task, 'contrast': name, 'reference': ref, 'n': a['n'],
                                         **{metric + '_delta': b[metric] - a[metric] if a[metric] is not None else None for metric in ('exact_accuracy','macro_f1','micro_f1')}})
    return {'coverage': coverage, 'metrics': metric_rows, 'per_label_metrics': per_label,
            'case_masks': masks, 'core_mask_counts': mask_counts, 'core_mask_transitions': mask_changes,
            'case_pairs': pairs, 'transition_counts': transition_counts,
            'reference_change_counts': relabel_counts,
            'metric_contrasts': contrast_metrics, 'margin_interactions': contrasts}


def csv_bytes(rows):
    out = io.StringIO()
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {k: canonical(v) if v is None or isinstance(v, (dict, list)) else v for k, v in row.items()}
            writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@', '\t', '\r')) else v for k, v in values.items()})
    return out.getvalue().encode('utf-8-sig')


def report(tables, raw_available):
    lines = ['# 原／审核双参考评估', '',
             '同一冻结 discovery 子集、同一历史预测，只切换参考标签。本次为 CPU 描述性分析，没有模型 forward。', '',
             '| 任务 | 查询总数 | 已确认 | 草稿 | 未审 | 未决 | 两套参考共同分母 | 排除 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in tables['coverage']:
        lines.append(f"| {r['task']} | {r['queue_n']} | {r['confirmed_n']} | {r['draft_n']} | {r['unreviewed_n']} | {r['unresolved_n']} | {r['paired_reference_n']} | {r['excluded_n']} |")
    lines += ['', '## 主分类点值', '', '| 任务/指标 | 条件 | 原参考 | 审核参考 |', '| --- | --- | ---: | ---: |']
    mi = {(r['task'],r['condition'],r['reference']):r for r in tables['metrics']}
    for task, condition in itertools.product(TASKS, CONDITIONS):
        field = 'macro_f1' if task == 'hate' else 'micro_f1'
        values = [mi[task,condition,ref][field] for ref in REFS]
        fmt = lambda v: f'{100*v:.2f}%' if v is not None else 'null'
        lines.append(f'| {task}/{field} | {condition} | {fmt(values[0])} | {fmt(values[1])} |')
    lines += ['', '完整的 exact accuracy、两类 F1 及逐标签计数见 metrics.csv 与 per_label_metrics.csv。', '',
              '## 切换参考后的正确性变化（预测固定）', '',
              '| 任务 | 条件 | 错→对 | 对→错 | 不变 |', '| --- | --- | ---: | ---: | ---: |']
    for row in tables['reference_change_counts']:
        lines.append(f"| {row['task']} | {row['condition']} | {row['wrong_to_right']} | {row['right_to_wrong']} | {row['wrong_to_wrong'] + row['right_to_right']} |")
    lines += ['', '## 条件转换（同一参考内）', '',
              '| 任务 | 对比 | 参考 | 错→对 | 对→错 | 不变 |', '| --- | --- | --- | ---: | ---: | ---: |']
    for row in tables['transition_counts']:
        if row['contrast'] in ('S_given_D','remove_with_D','remove_without_D'):
            lines.append(f"| {row['task']} | {row['contrast']} | {row['reference']} | {row['wrong_to_right']} | {row['right_to_wrong']} | {row['wrong_to_wrong'] + row['right_to_right']} |")
    lines += ['', '## 位型变化', '', '| 任务 | 纳入数 | 位型变化数 |', '| --- | ---: | ---: |']
    for task in TASKS:
        subset = [m for m in tables['case_masks'] if m['task']==task and m['reference_eligible']]
        lines.append(f"| {task} | {len(subset)} | {sum(m['mask_changed'] for m in subset)} |")
    lines += ['', '四位次序固定为 C0、CLnewNoCat、CD、CLDnewNoCat；所有 16 种位型和 16×16 转换（含零计数）均导出。', '',
              '## 连续读数与解释范围', '',
              '已校验完整候选，按审核参考重新求 best-nongold margin；group 使用全部 32 候选。' if raw_available else '原始分数缺失：仅使用已验证卡片中的历史预测，连续读数为 null，并标记 candidate_scores_missing。', '',
              '计分口径固定为不含 EOS 的 answer_sum；精确并列取最小 canonical ordinal，计分口径敏感和原 epsilon 近并列标记保留。', '',
              '原始 643 条总体表不改写。本子集按模型行为富集，未新增 bootstrap 或显著性检验，不估计总体误标率。', '',
              '案例原标注状态、基础政策与材料补充政策分别保留。标签差异不自动证明源标注错误；资源删除还改变长度/位置。', '',
              'case_masks 的模式仅描述行为变化，不是人工机制解释或输入实验放行。具体干预、假设、替代解释仍需单独冻结。', '']
    return '\n'.join(lines).encode()


def build(root, config_path, allow_missing_scores=False):
    sources, config, pointer, fm, cards, overlay, eligibility, raw_path, old_config = load_inputs(root, config_path, allow_missing_scores)
    raw = selected_raw(raw_path, set(cards), old_config['plan_id'])
    blocks = evaluate_blocks(cards, overlay, eligibility, raw, config['epsilon'])
    tables = aggregate(blocks, list(cards))
    audit = {'schema_version': 'evidence-dual-reference-audit/v1', 'passed': True,
             'selected_queries': len(cards), 'selected_task_conditions': len(blocks),
             'selected_candidates_verified': sum(len(b['candidates']) for b in raw.values()),
             'historical_predictions_reproduced': len(raw),
             'continuous_status': 'available' if raw_path is not None else 'candidate_scores_missing',
             'reference_session_revision': fm['session_revision'], 'reference_manifest_sha256': pointer['manifest_sha256'],
             'raw_payload_scope': 'query metadata scanned; only selected discovery candidates decoded',
             'reserve_content_reviewed': False, 'model_forward_executed': False,
             'new_human_judgments_created': False, 'original_sources_modified': False,
             'allow_gpu': False, 'gpu_status': 'not_required_for_this_evaluation',
             'new_input_controls_status': 'specific_interventions_and_hypotheses_not_frozen',
             'coverage': tables['coverage']}
    files = {'blocks.jsonl': jsonl(blocks), 'audit.json': json_bytes(audit), 'REPORT.md': report(tables, raw_path is not None),
             'evaluation_contract.json': json_bytes(config), 'candidate_catalog.json': json_bytes(candidate_catalog()),
             'summary.json': json_bytes({k:v for k,v in tables.items() if k not in ('case_pairs','margin_interactions')})}
    for name, rows in tables.items():
        files[name + '.csv'] = csv_bytes(rows)
        if name in ('case_masks','case_pairs','margin_interactions'):
            files[name + '.jsonl'] = jsonl(rows)
    sources.unchanged()
    return files, sources.hashes, audit


def write_output(target, files):
    require(not target.exists() and not target.is_symlink(), 'refusing to overwrite existing evaluation')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.dual-reference-', dir=target.parent) as tmp:
        for name, raw in files.items():
            p = Path(tmp) / name
            p.write_bytes(raw)
            p.chmod(0o600)
        # A single atomic publication; concurrent writers may not share an output.
        require(not target.exists(), 'output appeared during publication')
        os.rename(tmp, target)


def verify_output(target):
    m = read_json(target / 'manifest.json')
    require(m['schema_version'] == 'evidence-dual-reference-run/v1' and m['status'] == 'complete', 'not a complete evaluation')
    require({p.name for p in target.iterdir()} == {*m['artifacts'], 'manifest.json'}, 'unexpected or missing evaluation files')
    for name, h in m['artifacts'].items():
        require(Path(name).name == name and file_sha(target / name) == h, 'evaluation artifact hash mismatch')
    return m
