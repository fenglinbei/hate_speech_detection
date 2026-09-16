"""Gold-free Q01 C/I readouts, identity-aware bounds and predeclared nomination."""
from __future__ import annotations

from dataclasses import dataclass
import math

from diagnostics.evidence_label_calibration import background_margin, readouts
from diagnostics.general_model_evidence_evaluation import require
from diagnostics.q01_mechanism_inputs import MAIN_ROLES, digest, q01_bindings

MODES = ('answer_sum', 'answer_mean', 'total_with_eos', 'mean_with_eos')
PROBES = ('empty', 'space', 'na', 'mask', 'lorem')


@dataclass
class Expression:
    terms: dict

    def __add__(self, other):
        keys = self.terms.keys() | other.terms.keys()
        return Expression({k: v for k in keys if (v := self.terms.get(k, 0.0) + other.terms.get(k, 0.0)) != 0})

    def __mul__(self, coefficient):
        return Expression({k: coefficient * v for k, v in self.terms.items() if coefficient * v != 0})

    def __sub__(self, other): return self + other * -1

    def report(self, atoms, epsilon):
        return {'value': math.fsum(v * atoms[k] for k, v in sorted(self.terms.items())),
                'numeric_bound': math.fsum(abs(v) for v in self.terms.values()) * epsilon,
                'coefficients': dict(sorted(self.terms.items()))}


class ReadoutBuilder:
    def __init__(self, contexts, requests, rows, epsilon):
        self.contexts = {c['record_id']: c for c in contexts}
        self.requests = {r['request_id']: r for r in requests}
        self.rows = {r['request_id']: r for r in rows}
        require(len(self.rows) == len(rows), 'duplicate physical scoring result')
        self.baselines = {r['recipient']: r['request_id'] for r in requests if r['kind'] == 'baseline'}
        self.context_index = {}
        for c in contexts:
            b = q01_bindings(c)[0]
            self.context_index[b['demo_family'], b['lexicon_arm'], b['demo_surface'], c['encoding'], c['probe_id']] = c['record_id']
        self.patch_index = {}
        for r in requests:
            if r['category'] in ('baseline', 'engineering'): continue
            key = (r['group'], r['module'], r['layer'], r['role'], r['direction'], r['surface'], r['encoding'], r['probe_id'])
            require(key not in self.patch_index, 'duplicate intervention cell')
            self.patch_index[key] = r['request_id']
        self.values = {rid: readouts(row) for rid, row in self.rows.items()}
        self.atoms = {}
        self.atom_sources = {}
        self.epsilon = epsilon

    def margin(self, rid, mode):
        key = f'm:{rid}:{mode}'
        self.atoms[key] = self.values[rid]['margin/' + mode]
        self.atom_sources[key] = {'kind': 'margin', 'request_id': rid, 'mode': mode}
        return Expression({key: 1.0})

    def background(self, rids):
        require(rids and len(rids) == len(set(rids)), 'background probe identities are empty or duplicated')
        if len(rids) == 1: return self.margin(rids[0], 'answer_mean')
        key = 'b:' + digest(sorted(rids))
        self.atoms[key] = background_margin([self.values[rid]['margin/answer_mean'] for rid in rids])
        self.atom_sources[key] = {'kind': 'logit_mean_sigmoid', 'request_ids': sorted(rids), 'mode': 'answer_mean',
                                 'sup_norm_lipschitz_bound': 1.0}
        return Expression({key: 1.0})

    def old_background(self, cid, probes):
        c = self.contexts[cid]; b = q01_bindings(c)[0]
        return self.background([self.baselines[self.context_index[b['demo_family'], b['lexicon_arm'],
                               b['demo_surface'], 'original', probe]] for probe in probes])

    def arm(self, cell, direction, surface, encoding, mode, ncc=None, probes=PROBES):
        rid = self.patch_index[(*cell, direction, surface, encoding, None)]
        r = self.requests[rid]
        recipient, donor = r['recipient'], r['donor']
        p = self.margin(rid, mode)
        br = self.margin(self.baselines[recipient], mode)
        bd = self.margin(self.baselines[donor], mode)
        if ncc == 'fixed':
            prior = self.old_background(recipient, probes)
            p, br, bd = p - prior, br - prior, bd - prior
        elif ncc == 'recalibrated':
            patch_probes = [self.patch_index[(*cell, direction, surface, 'original', probe)] for probe in probes]
            p = p - self.background(patch_probes)
            br = br - self.old_background(recipient, probes)
            bd = bd - self.old_background(donor, probes)
        sign = 1 if direction == 'R' else -1
        return {'recipient': br, 'donor_under_estimand': bd, 'patched': p,
                'actual_shift': p - br, 'effect': (p - br) * sign,
                'target': (bd - br) * sign}

    def report(self, expr): return expr.report(self.atoms, self.epsilon)

    def comparisons(self):
        output = []
        cells = sorted({key[:4] for key in self.patch_index}, key=str)
        for cell in cells:
            example = self.requests[self.patch_index[(*cell, 'R', 'H', 'original', None)]]
            views = [(f'{encoding}/{mode}', encoding, mode, None, PROBES)
                     for encoding in ('original', 'ab_forward', 'ab_reverse') for mode in MODES]
            views.append(('ncc_fixed_recipient', 'original', 'answer_mean', 'fixed', PROBES))
            if cell[3] != 'query_hehe':
                views.append(('ncc_recalibrated', 'original', 'answer_mean', 'recalibrated', PROBES))
                views.extend((f'ncc_single/{p}', 'original', 'answer_mean', 'recalibrated', (p,)) for p in PROBES)
                views.extend((f'ncc_loo/{p}', 'original', 'answer_mean', 'recalibrated', tuple(x for x in PROBES if x != p)) for p in PROBES)
            for view, encoding, mode, ncc, probes in views:
                for direction in ('R', 'K'):
                    arms = {s: self.arm(cell, direction, s, encoding, mode, ncc, probes) for s in ('H', 'A')}
                    metrics = {}
                    for metric, coeff in (('C', (0.5, 0.5)), ('I', (1.0, -1.0))):
                        target = arms['H']['target'] * coeff[0] + arms['A']['target'] * coeff[1]
                        effect = arms['H']['effect'] * coeff[0] + arms['A']['effect'] * coeff[1]
                        residual = target - effect
                        t, e, z = (self.report(x) for x in (target, effect, residual))
                        resolved = abs(t['value']) > t['numeric_bound']
                        aligned = (resolved and abs(e['value']) > e['numeric_bound'] and t['value'] * e['value'] > 0)
                        improvement = abs(t['value']) - abs(z['value'])
                        improvement_bound = t['numeric_bound'] + z['numeric_bound']
                        metrics[metric] = {'target': t, 'effect': e, 'residual': z,
                            'target_resolved': resolved, 'aligned_beyond_bound': aligned,
                            'absolute_residual_improvement': improvement,
                            'improvement_numeric_bound': improvement_bound,
                            'eligible_direction': aligned and improvement > improvement_bound,
                            'effect_over_target': e['value'] / t['value'] if resolved else None,
                            'closeness_gain': 1 - abs(z['value']) / abs(t['value']) if resolved else None}
                    output.append({'cell_id': digest(cell), 'group': cell[0], 'module': cell[1], 'layer': cell[2],
                        'role': cell[3], 'category': example['category'], 'view': view, 'direction': direction,
                        'arms': {s: {k: self.report(v) for k, v in values.items()} for s, values in arms.items()},
                        'metrics': metrics, 'ncc_estimand': ncc, 'probe_ids': list(probes) if ncc else None,
                        'ncc_recalibration_applicable': cell[3] != 'query_hehe',
                        'shared_O_and_donors_are_not_replicates': True})
        return output


def flat_scalars(comparisons):
    result = {}
    for r in comparisons:
        prefix = (r['cell_id'], r['view'], r['direction'])
        for surface, fields in r['arms'].items():
            for field, value in fields.items(): result[(*prefix, 'arm', surface, field)] = value
        for metric, fields in r['metrics'].items():
            for field in ('target', 'effect', 'residual'): result[(*prefix, metric, field)] = fields[field]
    return result


def compare_derived(contexts, requests, reference, observed, limit):
    a = flat_scalars(ReadoutBuilder(contexts, requests, reference, limit).comparisons())
    b = flat_scalars(ReadoutBuilder(contexts, requests, observed, limit).comparisons())
    require(a.keys() == b.keys(), 'derived comparison coverage differs')
    largest, maximum, failures = None, 0.0, []
    for key in a:
        require(a[key]['coefficients'] == b[key]['coefficients'], 'derived estimand identity differs')
        error, bound = abs(a[key]['value'] - b[key]['value']), a[key]['numeric_bound']
        fraction = error / bound if bound else (0.0 if error == 0 else float('inf'))
        if fraction > maximum: maximum, largest = fraction, {'key': key, 'error': error, 'bound': bound}
        if error > bound: failures.append({'key': key, 'error': error, 'bound': bound})
    return {'passed': not failures, 'readouts': len(a), 'max_bound_fraction': maximum,
            'largest': largest, 'failures': failures}


def nominate(comparisons):
    primary = [r for r in comparisons if r['category'] == 'primary' and r['view'] == 'original/answer_sum']
    units = sorted({(r['layer'], r['role']) for r in primary}, key=lambda x: (x[0], MAIN_ROLES.index(x[1])))
    rankings, selected = {}, []
    for metric in ('C', 'I'):
        candidates = []
        for layer, role in units:
            rows = [r for r in primary if (r['layer'], r['role']) == (layer, role)]
            require(len(rows) == 8 and len({(r['group'], r['direction']) for r in rows}) == 8,
                    'nomination must cover all four groups in both directions')
            eligible = all(r['metrics'][metric]['eligible_direction'] for r in rows)
            ranks = [r['metrics'][metric]['closeness_gain'] for r in rows]
            candidates.append({'layer': layer, 'role': role, 'eligible': eligible,
                               'worst_closeness_gain': min(ranks) if all(x is not None for x in ranks) else None})
        ordered = sorted([c for c in candidates if c['eligible']],
                         key=lambda c: (-c['worst_closeness_gain'], c['layer'], MAIN_ROLES.index(c['role'])))
        winner = ordered[0] if ordered else None
        rankings[metric] = {'winner': winner, 'all_units': candidates, 'eligible_ranking': ordered}
        if winner and (winner['layer'], winner['role']) not in [(r['layer'], r['role']) for r in selected]:
            selected.append({'layer': winner['layer'], 'role': winner['role'], 'selected_for': [metric]})
        elif winner:
            next(r for r in selected if (r['layer'], r['role']) == (winner['layer'], winner['role']))['selected_for'].append(metric)
    secondary = []
    for s in selected:
        rows = [r for r in comparisons if r['category'] == 'primary' and (r['layer'], r['role']) == (s['layer'], s['role'])
                and r['view'] != 'original/answer_sum']
        secondary.append({**s, 'direction_exceptions': [
            {'group': r['group'], 'direction': r['direction'], 'view': r['view'], 'metric': metric,
             'target': r['metrics'][metric]['target']['value'], 'effect': r['metrics'][metric]['effect']['value']}
            for r in rows for metric in s['selected_for'] if not r['metrics'][metric]['aligned_beyond_bound']],
            'control_specificity_requires_interpretation': True})
    return {'primary_selection_score': 'original/answer_sum', 'rankings': rankings,
            'selected_units': selected, 'secondary_checks': secondary, 'selection_by_label_flip': False,
            'replace_winner_after_secondary_failure': False, 'automatic_refinement_started': False,
            'refinement_requires_separate_freeze': True, 'maximum_refinement_units': 12,
            'stop_this_scan_range': not selected, 'mechanism_ready': False}
