"""Native eager attention telemetry. Imports torch only inside forward()."""
from __future__ import annotations

import math
import numpy as np

from diagnostics.case_attention_inputs_v1 import ROLES, require
from diagnostics.cross_model_applicability_models_v1 import prepared_input


def forward(model, request, pad_token, padding='none', capture=True, prefix_role=None):
    """Return native final logits and [layer, head, role, valid-key] maps.

    All keys retain their original probability denominator. Multiple query rows
    are copied one layer at a time and averaged in FP64 on CPU. Model tensors
    and probabilities stay FP32. A missing role is zeros + explicit availability,
    never a scientific zero. No hook changes a module input or output.
    """
    import torch
    ids = request['input_ids']
    roles = request['roles']
    if prefix_role:
        require(prefix_role in ('lexicon_end', 'demos_end') and len(roles[prefix_role]) == 1, 'Bad prefix role')
        ids = ids[:roles[prefix_role][0] + 1]
        roles = {name: roles[name] if name == prefix_role else [] for name in ROLES}
    p = prepared_input(ids, pad_token, padding)
    valid = [i for i, v in enumerate(p['attention_mask']) if v]
    require(model.config._attn_implementation == 'eager' and not model.training, 'Native eager eval required')
    device = model.get_input_embeddings().weight.device
    tensors = {k: torch.tensor([p[k]], dtype=torch.long, device=device)
               for k in ('input_ids', 'attention_mask', 'position_ids')}
    maps, handles = {}, []
    row_positions = sorted(set(i for ps in roles.values() for i in ps))
    row_index = {v: i for i, v in enumerate(row_positions)}

    def hook(layer):
        def collect(module, args, output):
            require(isinstance(output, tuple) and len(output) >= 2, 'Attention output structure changed')
            weights = output[1]
            require(weights is not None and weights.dtype == torch.float32 and weights.ndim == 4, 'Missing native FP32 attention')
            require(weights.shape[0] == 1 and weights.shape[-2:] == (len(p['input_ids']),) * 2,
                    'Unexpected attention axes')
            require(layer not in maps, 'Layer hook ran twice')
            row_ids = torch.tensor([valid[i] for i in row_positions], dtype=torch.long, device=weights.device)
            key_ids = torch.tensor(valid, dtype=torch.long, device=weights.device)
            selected = weights[0].index_select(1, row_ids).index_select(2, key_ids).detach().cpu().numpy()
            require(np.isfinite(selected).all() and (selected >= 0).all(), 'Invalid attention probabilities')
            for i, row in enumerate(row_positions):
                require(np.count_nonzero(selected[:, i, row + 1:]) == 0, 'Future attention leakage')
            result = np.zeros((weights.shape[1], len(ROLES), len(ids)), dtype=np.float64)
            for j, name in enumerate(ROLES):
                if roles[name]:
                    result[:, j, :] = selected[:, [row_index[v] for v in roles[name]], :].mean(axis=1, dtype=np.float64)
            maps[layer] = result
            return None
        return collect

    try:
        if capture:
            for layer, block in enumerate(model.model.layers):
                handles.append(block.self_attn.register_forward_hook(hook(layer)))
        with torch.inference_mode():
            out = model.model(**tensors, use_cache=False, output_attentions=False, return_dict=True)
            require(out.past_key_values is None, 'Unexpected KV cache')
            logits = model.lm_head(out.last_hidden_state[:, p['last_valid_index']:p['last_valid_index'] + 1, :])
            require(logits.dtype == torch.float32 and bool(torch.isfinite(logits).all()), 'Invalid logits')
            vector = logits[0, 0].detach().cpu().numpy().copy()
    finally:
        for handle in handles:
            handle.remove()
    attention = None
    if capture:
        require(set(maps) == set(range(len(model.model.layers))), 'Incomplete layer capture')
        attention = np.stack([maps[i] for i in range(len(maps))])
    return vector, attention, roles


def validate_attention(attention, request, roles=None, cap=0.000002):
    roles = request['roles'] if roles is None else roles
    a = np.asarray(attention)
    require(a.dtype == np.float64 and a.ndim == 4 and a.shape[2] == len(ROLES), 'Invalid attention axes/dtype')
    require(np.isfinite(a).all() and (a >= 0).all() and (a <= 1 + cap).all(), 'Invalid attention values')
    error = 0.0
    for i, name in enumerate(ROLES):
        if not roles[name]:
            require(np.count_nonzero(a[:, :, i]) == 0, 'Missing role must have explicit zero storage')
            continue
        require(max(roles[name]) < a.shape[-1], 'Role outside keys')
        require(np.count_nonzero(a[:, :, i, max(roles[name]) + 1:]) == 0, 'Future keys have nonzero probability')
        error = max(error, float(np.abs(a[:, :, i].sum(axis=-1) - 1).max()))
    require(error <= cap, 'Attention row normalization failed')
    return error


def difference(left, right, roles=None):
    require(left.shape == right.shape, 'Attention comparison axes differ')
    delta = np.abs(left - right)
    if roles is not None:
        selected = [i for i, name in enumerate(ROLES) if roles[name]]
        delta = delta[:, :, selected, :]
    return {'max_element': float(delta.max(initial=0)), 'max_row_l1': float(delta.sum(axis=-1).max(initial=0))}


def aggregate(attention, request):
    """FP64 sums; denominator is mean number of causally available source tokens.

    Containers overlap their component columns by design, so only the disjoint
    token-owner partition is eligible for a mass-conservation assertion.
    """
    validate_attention(attention, request)
    result = []
    for ri, role in enumerate(ROLES):
        rows = request['roles'][role]
        for span in request['spans']:
            keys = span['token_positions']
            available = math.fsum(sum(k <= q for k in keys) for q in rows) / len(rows) if rows else 0
            visible = bool(rows and available > 0)
            mass = attention[:, :, ri, keys].sum(axis=-1) if keys else np.zeros(attention.shape[:2])
            result.append({'role': role, 'span_id': span['id'], 'label': span['label'], 'kind': span['kind'],
                           'visible': visible, 'missing_reason': None if visible else ('role_absent' if not rows else 'not_yet_visible_or_empty'),
                           'span_tokens': len(keys), 'mean_visible_tokens': available,
                           'mass': mass.tolist() if visible else None,
                           'density': (mass / available).tolist() if visible else None})
    return result


def qualify(comparisons, rules):
    require(comparisons, 'No engineering comparisons')
    max_margin = max_attention_l1 = max_attention_element = 0.0
    maxima = {}
    for row in comparisons:
        kind = row['kind']
        exact = kind in ('hook', 'repeat', 'reverse', 'production_replay')
        margin_cap = rules['margin_repeat_order_hook_cap'] if exact else rules['margin_padding_prefix_cap']
        # A prefix's last logits are not an answer readout; only its matching
        # attention boundary is compared. Hook invariance covers full vectors.
        if row.get('margin_difference') is not None:
            require(row['margin_difference'] <= margin_cap, f'{kind}: margin gate failed')
            max_margin = max(max_margin, row['margin_difference'])
        if kind == 'hook':
            require(row['vector_max_difference'] == 0, 'Read-only hook changed logits')
        if row.get('attention') is not None:
            d = row['attention']
            require(d['max_element'] <= (rules['attention_repeat_order_cap'] if exact else rules['attention_element_cap']), f'{kind}: attention element gate failed')
            require(d['max_row_l1'] <= (0 if exact else rules['attention_row_l1_cap']), f'{kind}: attention L1 gate failed')
            max_attention_l1 = max(max_attention_l1, d['max_row_l1'])
            max_attention_element = max(max_attention_element, d['max_element'])
        m = maxima.setdefault(kind, {'margin': 0.0, 'attention_element': 0.0, 'attention_row_l1': 0.0, 'comparisons': 0})
        m['comparisons'] += 1
        m['margin'] = max(m['margin'], row.get('margin_difference') or 0)
        if row.get('attention'):
            m['attention_element'] = max(m['attention_element'], row['attention']['max_element'])
            m['attention_row_l1'] = max(m['attention_row_l1'], row['attention']['max_row_l1'])
    return {'status': 'pass', 'maxima': maxima,
            'margin_error_bound': max(rules['margin_bound_floor'], 2 * max_margin),
            'attention_mass_bound': max(rules['attention_mass_bound_floor'], 2 * max_attention_l1),
            'max_attention_element_difference': max_attention_element,
            'meaning': 'Engineering reproducibility envelope, not uncertainty over cases or causal evidence.'}
