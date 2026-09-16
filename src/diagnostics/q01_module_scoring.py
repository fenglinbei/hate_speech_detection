"""Reuse frozen C/I arithmetic, but keep layer AND module in all selection keys."""
from diagnostics.general_model_evidence_evaluation import require
from diagnostics.q01_mechanism_scoring import ReadoutBuilder, compare_derived, MODES
from diagnostics.q01_module_inputs import UNITS, MODULES


def nominate(comparisons):
    primary = [r for r in comparisons if r['category'] == 'primary' and r['view'] == 'original/answer_sum']
    require({(r['layer'], r['module']) for r in primary} == set(UNITS), 'module nomination frame differs')
    rankings, selected = {}, []
    for metric in ('C', 'I'):
        candidates = []
        for layer, module in UNITS:
            rows = [r for r in primary if (r['layer'], r['module'], r['role']) == (layer, module, 'pre_answer')]
            require(len(rows) == 8 and len({(r['group'], r['direction']) for r in rows}) == 8,
                    'nomination requires all four groups in both directions per module')
            gains = [r['metrics'][metric]['closeness_gain'] for r in rows]
            candidates.append({'layer': layer, 'module': module, 'role': 'pre_answer',
                'eligible': all(r['metrics'][metric]['eligible_direction'] for r in rows),
                'worst_closeness_gain': min(gains) if all(x is not None for x in gains) else None})
        ordered = sorted([c for c in candidates if c['eligible']],
                         key=lambda c: (-c['worst_closeness_gain'], c['layer'], MODULES.index(c['module'])))
        winner = ordered[0] if ordered else None
        rankings[metric] = {'winner': winner, 'all_units': candidates, 'eligible_ranking': ordered}
        if winner:
            previous = next((s for s in selected if (s['layer'], s['module']) == (winner['layer'], winner['module'])), None)
            if previous:
                previous['selected_for'].append(metric)
            else:
                selected.append({k: winner[k] for k in ('layer', 'module', 'role')} | {'selected_for': [metric]})
    secondary = []
    for s in selected:
        rows = [r for r in comparisons if r['category'] == 'primary' and r['view'] != 'original/answer_sum'
                and (r['layer'], r['module'], r['role']) == (s['layer'], s['module'], s['role'])]
        secondary.append({**s, 'direction_exceptions': [
            {'group': r['group'], 'direction': r['direction'], 'view': r['view'], 'metric': metric,
             'target': r['metrics'][metric]['target']['value'], 'effect': r['metrics'][metric]['effect']['value'],
             'aligned_beyond_bound': r['metrics'][metric]['aligned_beyond_bound'],
             'residual_improvement_beyond_bound': r['metrics'][metric]['eligible_direction']}
            for r in rows for metric in s['selected_for'] if not r['metrics'][metric]['eligible_direction']],
            'control_specificity_requires_interpretation': True})
    return {'primary_selection_score': 'original/answer_sum', 'rankings': rankings,
            'selected_units': selected, 'secondary_checks': secondary, 'selection_by_label_flip': False,
            'replace_winner_after_secondary_failure': False, 'automatic_refinement_started': False,
            'automatic_transfer_started': False, 'stop_this_scan_range': not selected,
            'module_effects_are_not_additive_shares': True, 'mechanism_ready': False}
