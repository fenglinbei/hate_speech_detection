"""Native Qwen3 A/V factorial and fixed QAS; no checkpoint loading or gold access.

Layer numbers default to the frozen 17/18 and are configurable only for tiny CPU
engineering fixtures. The global eager adapter is restored even after failure;
one process must run one forward at a time. All other Qwen3 operations are native.
"""
from __future__ import annotations
import math
import threading
import numpy as np
from diagnostics import hehe_bridge_capture_v1 as bridge
from diagnostics.case_attention_inputs_v1 import require
from diagnostics.cross_model_applicability_models_v1 import prepared_input

_LOCK = threading.Lock()
donor_rows = bridge.donor_rows
array_sha = bridge.array_sha


def forward(model, request, pad_token, *, positions=(), observe=False,
            layer=None, replacement=None, padding='none', prefix_length=None,
            append_tokens=(), trajectory=False, capture_av=False, av_sources=None,
            av_layer=18, qas_factor=1, qas_layers=tuple(range(18, 36))):
    """Return vector, bank, patch proof, trajectory, actual A/V telemetry, proof.

    av_sources is a pair of same-recipient N/U telemetry dictionaries. Recombine
    before projection, then replace only the original p row AFTER native o_proj.
    Full-shaped matmuls retain native reduction shapes even for self rebuilding.
    Appended answer positions never replace the original intervention position.
    """
    import torch
    from transformers.models.qwen3 import modeling_qwen3 as native
    require(qas_factor in (1, 2, 4), 'QAS factor outside frozen grid')
    require(not (av_sources is not None and qas_factor != 1), 'AV and QAS cannot be combined')
    require(not model.training and model.config._attn_implementation == 'eager', 'Native eager eval required')
    require(prefix_length is None or (not capture_av and av_sources is None and qas_factor == 1), 'No answer-row intervention in true prefix')
    require(request['roles']['pre_answer'] == [request['prompt_tokens'] - 1], 'Original answer position differs')
    original_n = request['prompt_tokens']
    ids = request['input_ids'][:prefix_length] if prefix_length else request['input_ids'] + list(append_tokens)
    prep = prepared_input(ids, pad_token, padding)
    valid = [i for i, x in enumerate(prep['attention_mask']) if x]
    p = valid[original_n - 1] if prefix_length is None else None
    keys = valid[:original_n]
    telemetry, proofs, pending, handles = {}, [], {}, []
    if av_sources is not None:
        require(0 <= av_layer < len(model.model.layers), 'AV layer outside model')
        for source in av_sources:
            require(source['prompt_sha256'] == request['prompt_sha256'] and
                    source['input_ids_sha256'] == request['input_ids_sha256'] and
                    source['position'] == original_n - 1 and source['layer'] == av_layer,
                    'A/V donor must come from this same recipient prompt and layer')
            require(source['A'].shape == (model.config.num_attention_heads, original_n), 'AV source length/head mismatch')
            require(source['V'].shape == (model.config.num_key_value_heads, original_n, model.config.head_dim), 'AV value geometry mismatch')
            require(source['A'].dtype == source['V'].dtype == np.float32 and
                    np.isfinite(source['A']).all() and np.isfinite(source['V']).all(), 'AV source precision/finiteness')
            require((source['A'] >= 0).all() and np.max(np.abs(source['A'].sum(-1, dtype=np.float64) - 1)) <= 2e-6,
                    'AV probability row invalid')

    def adapter(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        li = module.layer_idx
        selected = (capture_av or av_sources is not None) and li == av_layer
        qas = qas_factor != 1 and li in qas_layers
        if not selected and not qas:
            return original(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
        require(dropout == 0 and not module.training and query.dtype == key.dtype == value.dtype == torch.float32,
                'Native FP32 eval without dropout required')
        require(query.shape[0] == 1 and query.shape[-2] == len(prep['input_ids']), 'Attention geometry changed')
        k = native.repeat_kv(key, module.num_key_value_groups)
        v = native.repeat_kv(value, module.num_key_value_groups)
        # Match the pinned native eager implementation operation for operation.
        scores = torch.matmul(query, k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, :k.shape[-2]]
        if qas:
            before = scores.clone()
            qkeys = [valid[i] for i in request['roles']['query_all']]
            require(qkeys and max(qkeys) < p, 'QAS keys must precede the answer')
            scores = scores.clone()
            scores[:, :, p, qkeys] += math.log(qas_factor)
            allowed = torch.zeros_like(scores, dtype=torch.bool)
            allowed[:, :, p, qkeys] = True
            require(torch.equal(scores[~allowed], before[~allowed]), 'QAS changed an unregistered score')
            proofs.append({'kind':'QAS', 'layer':li, 'factor':qas_factor,
                           'position':original_n-1, 'query_keys':request['roles']['query_all'],
                           'outside_scores_exact':True, 'V_native':True, 'mask_preserved':True})
        a = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        a = torch.nn.functional.dropout(a, p=dropout, training=module.training)
        av = torch.matmul(a, v)
        out = av.transpose(1, 2).contiguous()
        if selected:
            require(li not in pending, 'Attention adapter executed twice')
            data = None
            if capture_av:
                def array(t): return t.detach().cpu().numpy().copy()
                data = {'prompt_sha256':request['prompt_sha256'], 'input_ids_sha256':request['input_ids_sha256'],
                        'position':original_n-1, 'layer':li, 'Q':array(query[0, :, p]),
                        'K':array(key[0, :, keys]), 'V':array(value[0, :, keys]),
                        'scores':array(scores[0, :, p, keys]), 'A':array(a[0, :, p, keys]),
                        'head_AV':array(av[0, :, p])}
            hybrid = None
            if av_sources is not None:
                sa, sv = av_sources
                ah = a.clone()
                # Original p cannot see appended tokens or padding.
                ah[:, :, p, :] = 0
                ah[0, :, p, keys] = torch.as_tensor(sa['A'], device=a.device)
                vh = value.clone()
                vh[0, :, keys] = torch.as_tensor(sv['V'], device=value.device)
                hv = native.repeat_kv(vh, module.num_key_value_groups)
                hybrid = out.reshape(1, len(prep['input_ids']), -1).clone()
                hav = torch.matmul(ah, hv)
                hybrid[0, p] = hav[0, :, p].reshape(-1)
            pending[li] = (data, hybrid)
        return out, a

    def projection_hook(module, args, output):
        require(av_layer in pending, 'Projection without registered A/V telemetry')
        data, hybrid = pending.pop(av_layer)
        if data is not None:
            data['o_proj'] = output[0, p].detach().cpu().numpy().copy()
            telemetry.update(data)
        if hybrid is None: return None
        # functional.linear avoids recursively calling this projection hook.
        rebuilt = torch.nn.functional.linear(hybrid, module.weight, module.bias)
        changed = output.clone()
        changed[0, p] = rebuilt[0, p]
        changed_rows = torch.nonzero(torch.any(changed != output, dim=-1)[0]).flatten().tolist()
        require(set(changed_rows) <= {p}, 'AV changed a row outside original p')
        proofs.append({'kind':'AV', 'layer':av_layer, 'position':original_n-1,
                       'padded_position':p, 'outside_rows_exact':True,
                       'installed_sha256':array_sha(rebuilt[0,p].detach().cpu().numpy()),
                       'A_sha256':array_sha(av_sources[0]['A']), 'V_sha256':array_sha(av_sources[1]['V']),
                       'native_sha256':array_sha(output[0,p].detach().cpu().numpy())})
        if data is not None: telemetry['installed_o_proj'] = rebuilt[0,p].detach().cpu().numpy().copy()
        return changed

    require(_LOCK.acquire(blocking=False), 'Concurrent eager adapter is prohibited')
    original = native.eager_attention_forward
    try:
        if capture_av or av_sources is not None:
            handles.append(model.model.layers[av_layer].self_attn.o_proj.register_forward_hook(projection_hook))
        native.eager_attention_forward = adapter
        v, bank, patch, traj = bridge.forward(model, request, pad_token, positions=positions,
            observe=observe, layer=layer, replacement=replacement, padding=padding,
            prefix_length=prefix_length, append_tokens=append_tokens, trajectory=trajectory)
    finally:
        native.eager_attention_forward = original
        for handle in reversed(handles): handle.remove()
        _LOCK.release()
    require(not pending, 'Unfinished projection observation')
    require(bool(telemetry) == capture_av, 'Missing A/V telemetry')
    require(sum(x['kind']=='AV' for x in proofs) == int(av_sources is not None), 'AV intervention count differs')
    if qas_factor != 1:
        require([x['layer'] for x in proofs if x['kind']=='QAS'] == list(qas_layers), 'Missing QAS layer')
    return v, bank, patch, traj, telemetry or None, proofs


def structure(native, upstream, focal_positions):
    """Per-head outside-T invariance; no head averaging or log of probabilities."""
    require(native['prompt_sha256'] == upstream['prompt_sha256'] and native['position'] == upstream['position'], 'Different structural inputs')
    outside = [i for i in range(native['A'].shape[-1]) if i not in focal_positions]
    require(outside, 'No outside-focal keys')
    require(np.array_equal(native['Q'], upstream['Q']), 'Layer18 p Q changed')
    for key in ('K','V'):
        require(np.array_equal(native[key][:,outside], upstream[key][:,outside]), 'Layer18 outside-T '+key+' changed')
    n, u = native['scores'][:,outside].astype(np.float64), upstream['scores'][:,outside].astype(np.float64)
    # Since Q and outside K are identical, actual eager score rows are exact.
    require(np.array_equal(n,u), 'Outside-T pre-softmax scores changed')
    def lse(x):
        x=x.astype(np.float64); z=x.max(-1,keepdims=True)
        return z[:,0] + np.log(np.exp(x-z).sum(-1))
    denominator_shift = lse(native['scores']) - lse(upstream['scores'])
    return {'per_head_score_difference_exact':True, 'Q_exact':True, 'outside_KV_exact':True,
            'outside_key_count':len(outside), 'log_common_attention_scale_per_head':denominator_shift.tolist(),
            'interpretation':'structural local invariant, not stance understanding'}
