"""Prompt-only donor capture and scoped, auditable Qwen3 output hooks.

No model is loaded at import time. The registered numeric kernels are not changed.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib

from diagnostics.general_model_evidence_evaluation import require
from diagnostics.q01_mechanism_inputs import capture_specs, digest, q01_bindings


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().to(device='cpu').contiguous().numpy().tobytes()).hexdigest()


def hidden_output(output):
    import torch
    value = output[0] if isinstance(output, tuple) else output
    require(isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[0] == 1,
            'hook requires a batch-one [batch, sequence, hidden] tensor')
    require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()), 'hook state must be finite FP32')
    return value


def replace_output(output, value):
    return (value, *output[1:]) if isinstance(output, tuple) else value


def resolve_module(model, module, layer):
    backbone = model.model
    if module == 'embedding':
        require(layer is None, 'embedding hook cannot name a decoder layer')
        return backbone.embed_tokens
    require(type(layer) is int and 0 <= layer < len(backbone.layers), 'decoder layer out of range')
    block = backbone.layers[layer]
    if module == 'block': return block
    if module == 'attention': return block.self_attn
    if module == 'mlp': return block.mlp
    raise ValueError('unknown hook module: ' + str(module))


@dataclass(frozen=True)
class Donor:
    identity: dict
    vectors: object
    vector_sha256: str


class HookRuntime:
    def __init__(self, runner, contexts, positions, plan_id, *, expected_layers=36, expected_hidden=4096):
        self.runner = runner
        self.contexts = {c['record_id']: c for c in contexts}
        self.positions = positions
        self.plan_id = plan_id
        self.cache = {}
        self.capture_receipts = []
        self.forward_capture_count = 0
        require(len(self.contexts) == len(contexts), 'duplicate source context')
        require(len(runner.model.model.layers) == expected_layers, 'model layer count differs')
        require(runner.model.config.hidden_size == expected_hidden, 'model hidden width differs')
        require(not runner.model.training, 'patching requires eval mode')
        self.expected_hidden = expected_hidden
        self.active = False

    @contextmanager
    def guard(self, context, *, prompt_only=False):
        """Validate the actual root-forward payload, not only request metadata."""
        import torch
        calls = [0]

        def prehook(module, args, kwargs):
            require(not args, 'backbone must use explicit keyword inputs')
            ids = kwargs.get('input_ids')
            require(isinstance(ids, torch.Tensor) and ids.ndim == 2 and ids.shape[0] == 1,
                    'only batch-one token inputs are supported')
            n = context['prompt_tokens']
            require(ids.shape[1] == n if prompt_only else ids.shape[1] >= n,
                    'capture contains candidate tokens or scoring truncates prompt')
            require(ids[0, :n].tolist() == context['prompt_token_ids'], 'actual prompt differs from source')
            mask = kwargs.get('attention_mask')
            require(isinstance(mask, torch.Tensor) and mask.shape == ids.shape,
                    'missing or malformed attention mask')
            values = mask[0].tolist()
            require(values[:n] == [1] * n and all(v in (0, 1) for v in values)
                    and values == sorted(values, reverse=True), 'only right padding is supported')
            require(kwargs.get('use_cache') is False and kwargs.get('past_key_values') is None,
                    'KV cache is prohibited')
            require(kwargs.get('inputs_embeds') is None, 'implicit embedding input is prohibited')
            pids = kwargs.get('position_ids')
            if pids is not None:
                require(pids.shape == ids.shape and pids[0].tolist() == list(range(ids.shape[1])),
                        'position IDs differ from frozen zero-based geometry')
            calls[0] += 1

        handle = self.runner.model.model.register_forward_pre_hook(prehook, with_kwargs=True)
        try:
            yield calls
        finally:
            handle.remove()

    def capture(self, record_id):
        """One candidate-free forward collects all registered roles for this source."""
        if record_id in self.cache: return
        require(not self.active, 'cannot capture a donor during an intervention')
        import torch
        context = self.contexts[record_id]
        specs = capture_specs(self.positions[record_id])
        captured, handles = {}, []
        per_module = {}
        for spec in specs:
            per_module.setdefault((spec['module'], spec['layer']), []).append(spec)

        def collector(entries):
            def hook(module, inputs, output):
                hidden = hidden_output(output)
                require(hidden.shape == (1, context['prompt_tokens'], self.expected_hidden),
                        'donor capture shape includes non-prompt tokens')
                for spec in entries:
                    key = (spec['module'], spec['layer'], spec['role'])
                    require(key not in captured, 'donor hook fired more than once')
                    vectors = hidden[0, spec['positions'], :].detach().cpu().clone()
                    identity = {'plan_id': self.plan_id, 'runtime_sha256': digest(self.runner.identity),
                                'source_record_id': record_id, 'source_prompt_sha256': context['prompt_sha256'],
                                'source_context_sha256': context['context_sha256'],
                                'source_prompt_token_ids_sha256': context['prompt_token_ids_sha256'],
                                'source_prompt_tokens': context['prompt_tokens'], 'encoding': context['encoding'],
                                'probe_id': context['probe_id'], 'binding': q01_bindings(context)[0],
                                **spec, 'candidate_independent': True, 'source_is_prompt_only': True}
                    captured[key] = Donor(identity, vectors, tensor_hash(vectors))
                return None
            return hook

        try:
            for (module, layer), entries in per_module.items():
                handles.append(resolve_module(self.runner.model, module, layer).register_forward_hook(collector(entries)))
            ids = torch.tensor([context['prompt_token_ids']], dtype=torch.long, device=self.runner.device)
            with self.guard(context, prompt_only=True) as calls, torch.inference_mode():
                self.runner.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            require(calls == [1] and len(captured) == len(specs), 'incomplete prompt-only capture')
        finally:
            for handle in handles: handle.remove()
        self.cache[record_id] = captured
        self.forward_capture_count += 1
        self.capture_receipts.append({'source_record_id': record_id, 'prompt_tokens': context['prompt_tokens'],
            'source_is_prompt_only': True, 'forward_calls': 1,
            'states': [{'identity': d.identity, 'vector_sha256': d.vector_sha256,
                        'shape': list(d.vectors.shape)} for d in captured.values()]})

    def donor(self, request):
        recipient, source = (self.contexts[request[k]] for k in ('recipient', 'donor'))
        require(source['encoding'] == recipient['encoding'] and source['probe_id'] == recipient['probe_id'],
                'cross encoding/probe donor is prohibited')
        rb, db = q01_bindings(recipient)[0], q01_bindings(source)[0]
        require(all(rb[k] == db[k] for k in ('query_id', 'demo_family', 'demo_surface')),
                'cross family/query/surface donor is prohibited')
        require(source['prompt_tokens'] == recipient['prompt_tokens'], 'donor prompt length differs')
        positions = request['positions']
        require(positions and sorted(set(positions)) == positions and
                all(type(i) is int and 0 <= i < recipient['prompt_tokens'] for i in positions),
                'replacement must stay within a registered prompt span')
        target_role = self.positions[recipient['record_id']]['roles'][request['role']]
        require(target_role['applicable'] and set(positions) <= set(target_role['token_positions']),
                'replacement exceeds its recipient role')
        self.capture(source['record_id'])
        d = self.cache[source['record_id']][request['module'], request['layer'], request['role']]
        require(d.identity['runtime_sha256'] == digest(self.runner.identity) and
                d.identity['plan_id'] == self.plan_id and d.identity['source_is_prompt_only'], 'stale donor identity')
        require(tensor_hash(d.vectors) == d.vector_sha256, 'cached donor was modified')
        require(set(positions) <= set(d.identity['positions']), 'donor has no matching role positions')
        return d

    @contextmanager
    def intervention(self, request):
        import torch
        require(not self.active, 'nested hooks are prohibited')
        context = self.contexts[request['recipient']]
        d = self.donor(request)
        positions = request['positions']
        selected = d.vectors[[d.identity['positions'].index(i) for i in positions]].clone()
        state = {'donor_identity_sha256': digest(d.identity), 'donor_vector_sha256': d.vector_sha256,
                 'donor_positions': positions, 'hook_calls': 0, 'candidate_independent': True,
                 'source_is_prompt_only': True, 'recipient_state_spread_max': 0.0}
        first = [None]

        def hook(module, inputs, output):
            hidden = hidden_output(output)
            require(hidden.shape[2] == self.expected_hidden and hidden.shape[1] >= context['prompt_tokens'],
                    'recipient hidden geometry differs')
            before = hidden[0, positions, :].detach().cpu().clone()
            if first[0] is None:
                first[0] = before
                state.update(recipient_vector_sha256=tensor_hash(before),
                             recipient_norm=float(torch.linalg.vector_norm(before.double())),
                             donor_norm=float(torch.linalg.vector_norm(selected.double())),
                             difference_norm=float(torch.linalg.vector_norm((selected - before).double())))
            else:
                state['recipient_state_spread_max'] = max(state['recipient_state_spread_max'],
                    float((before - first[0]).abs().max()))
            patched = hidden.clone()
            patched[0, positions, :] = selected.to(hidden.device)
            state['hook_calls'] += 1
            return replace_output(output, patched)

        self.active = True
        handle = None
        try:
            handle = resolve_module(self.runner.model, request['module'], request['layer']).register_forward_hook(hook)
            with self.guard(context) as calls:
                yield state
            require(state['hook_calls'] == calls[0] and calls[0] > 0, 'hook did not fire once per scoring forward')
            require(tensor_hash(d.vectors) == d.vector_sha256, 'scoring mutated the shared donor')
        finally:
            if handle is not None: handle.remove()
            self.active = False

    @contextmanager
    def observation(self, context):
        """Read-only hooks during scoring; observed answer-bearing states never enter cache."""
        handles, counts = [], {}
        specs = capture_specs(self.positions[context['record_id']])
        modules = {(s['module'], s['layer']) for s in specs}
        for key in sorted(modules, key=str):
            counts[str(key)] = 0
            def hook(module, inputs, output, key=key):
                hidden = hidden_output(output)
                require(hidden.shape[1] >= context['prompt_tokens'], 'observer truncated prompt')
                # Exercise an actual read/copy while keeping donor cache exclusively prompt-only.
                hidden[:, :context['prompt_tokens']].detach()[0, -1].cpu().clone()
                counts[str(key)] += 1
                return None
            handles.append(resolve_module(self.runner.model, *key).register_forward_hook(hook))
        try:
            with self.guard(context) as calls:
                yield counts
            require(all(n == calls[0] and n > 0 for n in counts.values()), 'capture-only hook coverage differs')
        finally:
            for handle in handles: handle.remove()

    def score(self, request, catalog, options):
        from diagnostics.general_model_numeric_kernel_v2 import score_batch, score_prefix_block
        from diagnostics.general_model_numeric_analysis import candidate_scores
        context = self.contexts[request['recipient']]
        candidates = catalog[context['task']]
        ordered = list(reversed(candidates)) if options.get('permuted') else candidates
        prior_padding = self.runner.padding_extra
        self.runner.padding_extra = options.get('padding_extra', 0)
        require(not options.get('prefix') or not (self.runner.padding_extra or options.get('permuted')),
                'prefix challenge must be unpadded with canonical candidate order')

        def forward():
            if options.get('prefix'):
                return score_prefix_block(self.runner, context, ordered, reference=options.get('reference', False))
            return [score_batch(self.runner, [{'context': context, 'candidate': c}],
                                reference=options.get('reference', False))[0] for c in ordered]

        hook_receipt = None
        try:
            if request['kind'] == 'baseline':
                with self.guard(context): results = forward()
            elif request['kind'] == 'capture_only':
                self.capture(context['record_id'])
                with self.observation(context) as counts: results = forward()
                hook_receipt = {'observation_counts': counts, 'donor_cache_written_by_observer': False}
            else:
                with self.intervention(request) as hook_receipt: results = forward()
        finally:
            self.runner.padding_extra = prior_padding
        by_id = {}
        for candidate, result in zip(ordered, results, strict=True):
            by_id[candidate['candidate_id']] = {**result, **candidate,
                'scores': candidate_scores(result['token_logprobs'], result['eos_logprob'])}
        return {'request_id': request['request_id'], **{k: context[k] for k in
                ('record_id', 'query_id', 'task', 'condition', 'context_sha256', 'prompt_sha256')},
                'candidates': [by_id[c['candidate_id']] for c in candidates], 'hook_receipt': hook_receipt,
                'scoring_options': options, 'query_reference_loaded': False,
                'formal_test_or_reserve_access': False}
