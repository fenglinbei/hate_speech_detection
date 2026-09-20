"""Sparse, read-only native Qwen3 state and attention-value telemetry.

The original logits/attention collector remains the native forward. These hooks
only copy tensors and perform side computations. Source groups partition keys;
they are not independent semantic mechanisms. No reference labels enter capture.
"""
from __future__ import annotations

import numpy as np

from diagnostics import case_attention_capture_v1 as base
from diagnostics.case_attention_capture_v1 import validate_attention, difference, aggregate, qualify
from diagnostics.case_attention_inputs_v1 import require
from diagnostics.cross_model_applicability_models_v1 import prepared_input
from diagnostics.hehe_presentation_inputs_v1 import GROUPS, MECHANISM_RULES

STATE_SITES = ['pre', 'mid', 'post']
BRANCHES = ['attention', 'mlp']
ROW_ARRAYS = ['states', 'normalized', 'lens_logits', 'branches']
AV_ARRAYS = ['av_heads', 'source_outputs', 'native_heads', 'native_attention', 'source_mass']


def forward(model, request, pad_token, padding='none', capture=True, prefix_role=None):
    import torch
    if not capture:
        v, a, roles = base.forward(model, request, pad_token, padding, False, prefix_role)
        return v, a, roles, None
    ids = request['input_ids']
    if prefix_role:
        ids = ids[:request['roles'][prefix_role][0] + 1]
    p = prepared_input(ids, pad_token, padding)
    valid = [i for i, value in enumerate(p['attention_mask']) if value]
    geo = request['mechanism']
    positions = [i for i in geo['positions'] if i < len(ids)]
    av_positions = [i for i in geo['av_positions'] if i < len(ids)]
    require(positions, 'No selected state positions')
    selected = [valid[i] for i in positions]
    av_selected = [valid[i] for i in av_positions]
    # Candidate IDs are derived from the model-independent, native-token freeze.
    candidate_ids = request['mechanism']['candidate_ids']
    weight = model.lm_head.weight[candidate_ids].detach()
    final_norm = model.model.norm
    nheads, nkv = model.config.num_attention_heads, model.config.num_key_value_heads
    hd = model.config.head_dim
    require(nheads % nkv == 0, 'Invalid GQA layout')
    handles, layers, cache = [], {}, {}

    def rows(tensor, indices):
        require(tensor.dtype == torch.float32 and tensor.shape[0] == 1, 'Native FP32 batch-one tensor required')
        return tensor[0, indices, :].detach()

    def cpu(tensor):
        return tensor.detach().cpu().numpy().copy()

    def block_pre(li):
        def hook(module, args, kwargs):
            x = kwargs.get('hidden_states', args[0] if args else None)
            require(li not in cache, 'Repeated decoder call')
            cache[li] = {'pre': rows(x, selected)}
        return hook

    def mid_pre(li):
        def hook(module, args):
            cache[li]['mid'] = rows(args[0], selected)
        return hook

    def value_hook(li):
        def hook(module, args, output):
            cache[li]['value'] = output.detach()
        return hook

    def output_pre(li):
        def hook(module, args):
            cache[li]['native_heads'] = rows(args[0], av_selected).reshape(len(av_selected), nheads, hd)
        return hook

    def attention_hook(li):
        def hook(module, args, output):
            data = cache[li]
            data['attention'] = rows(output[0], selected)
            data['native_attention'] = rows(output[0], av_selected)
            require(module.o_proj.bias is None, 'Source decomposition assumes bias-free output projection')
            probs = output[1]
            require(probs is not None and probs.dtype == torch.float32, 'Native probabilities unavailable')
            values = data.pop('value')[0, valid, :].reshape(len(ids), nkv, hd).transpose(0, 1)
            values = values.repeat_interleave(nheads // nkv, dim=0)
            # Retain all original probability denominators; no renormalization.
            a = probs[0, :, av_selected, :][:, :, valid]
            grouped, mass = [], []
            for name in GROUPS:
                keys = [i for i in geo['source_groups'][name] if i < len(ids)]
                if keys:
                    part = torch.matmul(a[:, :, keys], values[:, keys, :]).transpose(0, 1)
                else:
                    part = torch.zeros((len(av_positions), nheads, hd), dtype=torch.float32, device=values.device)
                grouped.append(part)
                mass.append(a[:, :, keys].to(torch.float64).sum(dim=-1).transpose(0, 1))
            heads = torch.stack(grouped, dim=1)
            data['av_heads'] = heads
            data['source_mass'] = torch.stack(mass, dim=1)
            # Functional linear avoids recursively invoking the native o_proj hook.
            data['source_outputs'] = torch.nn.functional.linear(heads.flatten(-2), module.o_proj.weight)
        return hook

    def mlp_hook(li):
        def hook(module, args, output):
            cache[li]['mlp'] = rows(output, selected)
        return hook

    def block_post(li):
        def hook(module, args, output):
            require(isinstance(output, torch.Tensor), 'Decoder output contract changed')
            data = cache.pop(li)
            data['post'] = rows(output, selected)
            states = torch.stack([data[name] for name in STATE_SITES])
            normalized = final_norm(states)
            lens = torch.nn.functional.linear(normalized, weight)
            layers[li] = dict(states=cpu(states), normalized=cpu(normalized), lens_logits=cpu(lens),
                branches=cpu(torch.stack([data[name] for name in BRANCHES])),
                **{name: cpu(data[name]) for name in AV_ARRAYS})
        return hook

    try:
        for li, block in enumerate(model.model.layers):
            handles += [block.register_forward_pre_hook(block_pre(li), with_kwargs=True),
                block.self_attn.v_proj.register_forward_hook(value_hook(li)),
                block.self_attn.o_proj.register_forward_pre_hook(output_pre(li)),
                block.self_attn.register_forward_hook(attention_hook(li)),
                block.post_attention_layernorm.register_forward_pre_hook(mid_pre(li)),
                block.mlp.register_forward_hook(mlp_hook(li)), block.register_forward_hook(block_post(li))]
        v, a, roles = base.forward(model, request, pad_token, padding, True, prefix_role)
    finally:
        for handle in reversed(handles):
            handle.remove()
    require(not cache and set(layers) == set(range(len(model.model.layers))), 'Incomplete state capture')
    telemetry = {name: np.stack([layers[i][name] for i in range(len(layers))]) for name in ROW_ARRAYS + AV_ARRAYS}
    counts = np.asarray([[sum(k <= row for k in geo['source_groups'][name]) for name in GROUPS] for row in av_positions], dtype=np.int64).reshape(len(av_positions), len(GROUPS))
    telemetry.update(positions=np.asarray(positions, dtype=np.int64), av_positions=np.asarray(av_positions, dtype=np.int64),
        source_counts=counts, candidate_ids=np.asarray(candidate_ids, dtype=np.int64),
        label_weights=cpu(weight), norm_weight=cpu(final_norm.weight), norm_eps=np.asarray([final_norm.variance_epsilon], dtype=np.float64))
    return v, a, roles, telemetry


def scaled_error(left, right):
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    require(a.shape == b.shape, 'Mechanism comparison shape mismatch')
    absolute = np.max(np.abs(a - b), initial=0.0)
    scale = max(1.0, float(np.max(np.abs(a), initial=0.0)), float(np.max(np.abs(b), initial=0.0)))
    return {'absolute': float(absolute), 'scaled': float(absolute / scale)}


def validate_mechanism(m, req, profile, vector, prefix_role=None):
    require(m is not None, 'Missing mechanism telemetry')
    length = req['roles'][prefix_role][0] + 1 if prefix_role else req['prompt_tokens']
    pos = [i for i in req['mechanism']['positions'] if i < length]
    av = [i for i in req['mechanism']['av_positions'] if i < length]
    L, H, D, S, A, G = profile['layers'], profile['heads'], profile['hidden_size'], len(pos), len(av), len(GROUPS)
    shapes = {'states': (L, 3, S, D), 'normalized': (L, 3, S, D), 'lens_logits': (L, 3, S, 2),
        'branches': (L, 2, S, D), 'av_heads': (L, A, G, H, profile['head_dim']),
        'source_outputs': (L, A, G, D), 'native_heads': (L, A, H, profile['head_dim']),
        'native_attention': (L, A, D), 'label_weights': (2, D), 'norm_weight': (D,)}
    shapes['source_mass'] = (L, A, G, H)
    for name, shape in shapes.items():
        require(m[name].shape == shape and m[name].dtype == (np.float64 if name == 'source_mass' else np.float32) and np.isfinite(m[name]).all(), 'Invalid mechanism array ' + name)
    require(m['positions'].tolist() == pos and m['av_positions'].tolist() == av, 'State position binding changed')
    require(m['candidate_ids'].tolist() == [profile['candidate_tokens']['有'], profile['candidate_tokens']['无']], 'Probe output tokens changed')
    require(m['norm_eps'].tolist() == [profile['rms_norm_eps']], 'Norm epsilon changed')
    counts = np.asarray([[sum(k <= row for k in req['mechanism']['source_groups'][name]) for name in GROUPS] for row in av], dtype=np.int64).reshape(A, G)
    require(np.array_equal(m['source_counts'], counts), 'AV visibility changed')
    for ri in range(A):
        require(np.count_nonzero(m['av_heads'][:, ri, counts[ri] == 0]) == 0, 'Unavailable source has nonzero AV')
        require(np.count_nonzero(m['source_outputs'][:, ri, counts[ri] == 0]) == 0, 'Unavailable source has nonzero output')
    checks = {
        'AV_reconstruction': scaled_error(m['av_heads'].sum(axis=2, dtype=np.float64), m['native_heads']),
        'O_reconstruction': scaled_error(m['source_outputs'].sum(axis=2, dtype=np.float64), m['native_attention']),
        'residual_attention': scaled_error(m['states'][:, 0].astype(np.float64) + m['branches'][:, 0], m['states'][:, 1]),
        'residual_mlp': scaled_error(m['states'][:, 1].astype(np.float64) + m['branches'][:, 1], m['states'][:, 2]),
        'residual_continuity': scaled_error(m['states'][:-1, 2], m['states'][1:, 0]),
        'source_probability': scaled_error(m['source_mass'].sum(axis=2), np.ones((L, A, H))),
    }
    state = m['states'].astype(np.float64)
    normalized = state / np.sqrt(np.mean(state * state, axis=-1, keepdims=True) + float(m['norm_eps'][0])) * m['norm_weight']
    checks['normalization'] = scaled_error(normalized, m['normalized'])
    projected = m['normalized'].astype(np.float64) @ m['label_weights'].astype(np.float64).T
    checks['probe_linear'] = scaled_error(projected, m['lens_logits'])
    projection_error = checks['probe_linear']['absolute']
    if not prefix_role:
        i = pos.index(req['roles']['pre_answer'][0])
        projection_error = max(projection_error, float(np.abs(m['lens_logits'][-1, 2, i].astype(np.float64) - vector[m['candidate_ids']]).max()))
    for name, value in checks.items():
        require(value['scaled'] <= MECHANISM_RULES['reconstruction_scaled_max_cap'], name + ' mechanism reconstruction gate failed')
    require(projection_error <= MECHANISM_RULES['projection_absolute_cap'], 'Native probe logit gate failed')
    return {'checks': checks, 'projection_absolute_error': projection_error}


def mechanism_difference(left, right, kind):
    prefix = kind == 'prefix'
    common = right['positions'].tolist()
    li = [left['positions'].tolist().index(i) for i in common]
    common_av = right['av_positions'].tolist()
    ai = [left['av_positions'].tolist().index(i) for i in common_av]
    details = {}
    for name in ROW_ARRAYS + AV_ARRAYS:
        a, b = left[name], right[name]
        if prefix:
            a = a[:, :, li] if name in ROW_ARRAYS else a[:, ai]
        details[name] = scaled_error(a, b)
    exact = kind in ('repeat', 'reverse', 'production_replay')
    for name, value in details.items():
        cap = 0 if exact else MECHANISM_RULES['padding_prefix_scaled_max_cap']
        require(value['scaled'] <= cap, kind + ': mechanism ' + name + ' gate failed')
    require(details['lens_logits']['absolute'] <= (0 if exact else MECHANISM_RULES['projection_absolute_cap']), kind + ': probe difference gate failed')
    for name in ('candidate_ids', 'label_weights', 'norm_weight', 'norm_eps'):
        require(np.array_equal(left[name], right[name]), 'Static mechanism weights differ')
    return details


def patch_forward(model, request, pad_token, layer, positions, replacement):
    """Stage-2 primitive: explicit intervention, never called by acquisition.

    Decoder block *output* replacement only. A final-layer query patch has no
    subsequent attention route to the answer. The donor must be native-aligned.
    """
    import torch
    require(0 <= layer < len(model.model.layers), 'Bad patch layer')
    require(positions and len(set(positions)) == len(positions) and max(positions) < request['prompt_tokens'] - 1, 'Patch only explicit query positions')
    require(set(positions) <= set(request['roles']['query_all']), 'Patch outside query')
    require(replacement.shape == (len(positions), model.config.hidden_size) and np.isfinite(replacement).all(), 'Bad replacement geometry')
    calls = []
    def intervene(module, args, output):
        require(not calls, 'Repeated intervention hook')
        calls.append(True)
        result = output.clone()
        result[0, positions] = torch.as_tensor(replacement, dtype=output.dtype, device=output.device)
        return result
    handle = model.model.layers[layer].register_forward_hook(intervene)
    try:
        vector, _, _ = base.forward(model, request, pad_token, capture=False)
    finally:
        handle.remove()
    require(len(calls) == 1, 'Patch never applied')
    return vector
