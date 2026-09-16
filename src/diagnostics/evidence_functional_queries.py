"""Deduplicated scoring aliases and query-aware functional contrasts, without Gold."""
from collections import defaultdict
import math

from diagnostics.general_model_evidence_evaluation import require
from diagnostics.evidence_label_calibration import PROBES, SCORE_MODES, readouts, background_margin, validate_block

MAIN_VIEWS = ('original/answer_sum', 'original/answer_mean', 'ncc',
              'ab_forward/answer_sum', 'ab_reverse/answer_sum')
RAW_VIEWS = tuple(e + '/' + m for e in ('original', 'ab_forward', 'ab_reverse') for m in SCORE_MODES)
CAL_VIEWS = ('background', 'ncc') + tuple(k + '/' + p for k in ('single_probe', 'leave_one_out') for p, _ in PROBES)
VIEWS = RAW_VIEWS + CAL_VIEWS + ('ab_symmetric_sum', 'ab_mapping_gap_sum')


def direction(value, bound):
    require(math.isfinite(value) and math.isfinite(bound) and bound >= 0, 'invalid numerical direction')
    return 'unresolved' if abs(value) <= bound else 'positive' if value > 0 else 'negative'


def numeric_values(contexts, raw, catalog=None):
    """Each unique prompt is scored once; aliases never become extra observations."""
    require([c['record_id'] for c in contexts] == [r['record_id'] for r in raw], 'raw input coverage/order differs')
    index, metadata, values, calibration = {}, {}, {}, []
    for c, row in zip(contexts, raw):
        if catalog is not None: validate_block(row, c, catalog)
        score = readouts(row)
        for a in c['bindings']:
            cid = a['condition_id']; key = (cid, c['encoding'], c['probe_id'])
            require(key not in index, 'duplicate logical alias')
            index[key] = (score, c['record_id'])
            metadata[cid] = a
    conditions = sorted(cid for cid, enc, probe in index if enc == 'original' and probe is None)
    for cid in conditions:
        meta = metadata[cid]
        vv = {}
        for encoding in ('original', 'ab_forward', 'ab_reverse'):
            r, _ = index[cid, encoding, None]
            for mode in SCORE_MODES: vv[encoding + '/' + mode] = r['margin/' + mode]
        probes = [index[cid, 'original', p][0]['margin/answer_mean'] for p, _ in PROBES]
        real = vv['original/answer_mean']; bg = background_margin(probes)
        vv.update(background=bg, ncc=real - bg)
        single = {p: real - probes[i] for i, (p, _) in enumerate(PROBES)}
        loo = {p: real - background_margin(probes[:i] + probes[i + 1:]) for i, (p, _) in enumerate(PROBES)}
        vv.update({'single_probe/' + p: v for p, v in single.items()})
        vv.update({'leave_one_out/' + p: v for p, v in loo.items()})
        f, rev = vv['ab_forward/answer_sum'], vv['ab_reverse/answer_sum']
        vv.update(ab_symmetric_sum=(f + rev) / 2, ab_mapping_gap_sum=f - rev)
        require(set(vv) == set(VIEWS) and all(math.isfinite(x) for x in vv.values()), 'missing/nonfinite view')
        values[cid] = vv
        calibration.append({'condition_id': cid, 'query_id': meta['query_id'], 'real_mean_margin': real,
                            'background_margin': bg, 'ncc_margin': real - bg,
                            'probe_margins': dict(zip((p for p, _ in PROBES), probes)),
                            'probe_ncc_margins': single, 'leave_one_out_ncc_margins': loo,
                            'physical_real_record_id': index[cid, 'original', None][1],
                            'physical_probe_record_ids': {p: index[cid, 'original', p][1] for p, _ in PROBES}})
    return values, calibration, metadata, index


def linear_value(comp, view, values, aliases):
    """Combine shared physical terms before error propagation, including NCC priors."""
    terms = comp['terms']; real_terms, background_terms = defaultdict(float), defaultdict(float)
    is_cal = view in CAL_VIEWS
    for t in terms:
        cid, w = t['condition_id'], t['coefficient']
        require(cid in values, 'comparison references an absent condition')
        if is_cal:
            if view != 'background': real_terms[aliases[cid, 'original', None][1]] += w
            if view.startswith('single_probe/'):
                key = (view, aliases[cid, 'original', view.split('/')[1]][1])
            else:
                selected = [p for p, _ in PROBES if not view.startswith('leave_one_out/') or p != view.split('/')[1]]
                key = (view, tuple(aliases[cid, 'original', p][1] for p in selected))
            background_terms[key] += w
        elif view.startswith('ab_') and '/' not in view:
            for enc, factor in [('ab_forward', .5 if view == 'ab_symmetric_sum' else 1),
                                ('ab_reverse', .5 if view == 'ab_symmetric_sum' else -1)]:
                real_terms[aliases[cid, enc, None][1]] += w * factor
        else:
            real_terms[aliases[cid, view.split('/')[0], None][1]] += w
    factor = math.fsum(abs(w) for w in real_terms.values()) + math.fsum(abs(w) for w in background_terms.values())
    value = math.fsum(t['coefficient'] * values[t['condition_id']][view] for t in terms)
    cancellation_residual = None
    if is_cal and not any(background_terms.values()):
        simplified = 0.0 if view == 'background' else math.fsum(
            t['coefficient'] * values[t['condition_id']]['original/answer_mean'] for t in terms)
        cancellation_residual = value - simplified
        require(abs(cancellation_residual) <= 1e-10, 'NCC shared-background algebra failed')
        value = simplified
    return value, factor, cancellation_residual


def condition_factor(view):
    return 2 if view == 'ncc' or view.startswith(('single_probe/', 'leave_one_out/')) or view == 'ab_mapping_gap_sum' else 1


def derived_readouts(contexts, raw, comparisons):
    values, _, _, aliases = numeric_values(contexts, raw)
    result = {}
    for cid, vv in values.items():
        for view in CAL_VIEWS: result[cid + '/' + view] = (vv[view], condition_factor(view))
    for c in comparisons:
        for view in CAL_VIEWS:
            value, factor, _ = linear_value(c, view, values, aliases)
            result[c['contrast_id'] + '/' + view] = (value, factor)
    return result


def compare_derived(contexts, reference, other, comparisons, epsilon):
    a, b = [derived_readouts(contexts, rows, comparisons) for rows in (reference, other)]
    require(a.keys() == b.keys(), 'derived coverage differs')
    rows, largest = [], 0.0
    for name, (left, factor) in a.items():
        right, other_factor = b[name]
        require(factor == other_factor, 'propagated factor changed')
        diff, bound = right - left, epsilon * factor
        passed = abs(diff) <= bound
        ratio = abs(diff) / bound if bound else (0 if diff == 0 else float('inf'))
        largest = max(largest, ratio)
        rows.append({'metric': name, 'difference': diff, 'bound': bound, 'passed': passed,
                     'shared_background_structural_zero': factor == 0})
    return {'passed': all(r['passed'] for r in rows), 'readouts': len(rows),
            'max_bound_fraction': largest, 'differences': rows}


def prediction_fields(margin, bound, reference):
    predicted = 'non-hate' if margin > 0 else 'hate'
    original, reviewed = reference['original_label'], reference['adjudicated_label']
    return {'prediction': predicted, 'exact_tie': margin == 0, 'numerically_unresolved': abs(margin) <= bound,
            'original_reference': original, 'reviewed_reference': reviewed,
            'original_correct': None if original is None else predicted == original,
            'reviewed_correct': None if reviewed is None else predicted == reviewed,
            'reviewed_margin': None if reviewed is None else margin * (1 if reviewed == 'non-hate' else -1)}


def analyze_rows(contexts, raw, references, comparisons, catalog, epsilon):
    values, calibration, metadata, aliases = numeric_values(contexts, raw, catalog)
    refs = {r['query_id']: r for r in references}
    rows = []
    for cid, vv in values.items():
        q = metadata[cid]['query_id']; ref = refs[q]
        for view in VIEWS:
            bound = epsilon * condition_factor(view)
            prediction = prediction_fields(vv[view], bound, ref)
            if view in ('background', 'ab_mapping_gap_sum'):
                for key in ('prediction', 'original_correct', 'reviewed_correct', 'reviewed_margin'):
                    prediction[key] = None
            rows.append({'condition_id': cid, 'query_id': q, 'demo_family': metadata[cid]['demo_family'],
                         'lexicon_arm': metadata[cid]['lexicon_arm'], 'demo_surface': metadata[cid]['demo_surface'],
                         'view': view, 'non_hate_margin': vv[view], 'numeric_bound': bound,
                         **prediction})
    for r in calibration:
        r.update(prediction_fields(r['ncc_margin'], 2 * epsilon, refs[r['query_id']]))
    effects = []
    for c in comparisons:
        for view in VIEWS:
            value, factor, residual = linear_value(c, view, values, aliases)
            effects.append({'contrast_id': c['contrast_id'], 'role': c['role'], 'cross_query': c['cross_query'],
                            'view': view, 'effect': value, 'numeric_bound': epsilon * factor,
                            'direction': direction(value, epsilon * factor), 'statistical_interval': False,
                            'ncc_background_cancels': c['ncc_background_cancels'],
                            'cancellation_residual': residual, 'terms': c['terms']})
    ei = {(r['contrast_id'], r['view']): r for r in effects}
    summaries = []
    for c in comparisons:
        dirs = {v: ei[c['contrast_id'], v]['direction'] for v in MAIN_VIEWS}
        summaries.append({**c, 'main_effects': {v: ei[c['contrast_id'], v]['effect'] for v in MAIN_VIEWS},
            'main_directions': dirs, 'all_five_directions_agree_resolved': len(set(dirs.values())) == 1 and 'unresolved' not in dirs.values(),
            'all_single_probes_agree_with_ncc': dirs['ncc'] != 'unresolved' and all(ei[c['contrast_id'], 'single_probe/' + p]['direction'] == dirs['ncc'] for p, _ in PROBES),
            'all_leave_one_out_agree_with_ncc': dirs['ncc'] != 'unresolved' and all(ei[c['contrast_id'], 'leave_one_out/' + p]['direction'] == dirs['ncc'] for p, _ in PROBES)})
    return rows, calibration, effects, summaries
