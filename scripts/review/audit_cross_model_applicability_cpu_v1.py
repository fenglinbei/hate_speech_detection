"""CPU-only local identity and tokenizer audit. Never instantiates a model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import struct

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/research/experiment-plans/cross-model-applicability-v1/draft-01'
MODELS = [
    ('qwen3-8b', 'Qwen/Qwen3-8B', ROOT / 'models/base/Qwen3-8B'),
    ('qwen3-14b', 'Qwen/Qwen3-14B', Path('/data/models/Qwen3-14B')),
    ('glm4-9b-chat', 'zai-org/glm-4-9b-chat', ROOT / 'models/base/GLM-4-9B-Chat'),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def dump(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def file_info(path, full=True):
    before = path.stat()
    h = hashlib.sha256()
    if full:
        with path.open('rb') as f:
            while block := f.read(4 * 1024 * 1024):
                h.update(block)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
    return {'path': str(path.resolve()), 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns,
            'sha256': h.hexdigest() if full else None}


def inventory():
    result = {'schema_version': 'cross-model-local-identity/v1', 'model_weights_loaded': False,
              'model_forward_calls': 0, 'models': []}
    for key, official, path in MODELS:
        cfg = read(path / 'config.json')
        index = read(path / 'model.safetensors.index.json')
        shards = sorted(set(index['weight_map'].values()))
        sources, weight_files, shapes, tensor_bytes = [], [], {}, 0
        names = ['config.json', 'generation_config.json', 'model.safetensors.index.json',
                 'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt',
                 'tokenizer.model', 'tokenization_chatglm.py', 'configuration_chatglm.py',
                 'modeling_chatglm.py']
        for name in names:
            p = path / name
            if p.exists():
                sources.append(file_info(p))
        for name in shards:
            p = path / name
            with p.open('rb') as f:
                header_len = struct.unpack('<Q', f.read(8))[0]
                assert header_len < 100_000_000
                header_bytes = f.read(header_len)
                header = json.loads(header_bytes)
            for tensor, record in header.items():
                if tensor == '__metadata__':
                    continue
                assert index['weight_map'][tensor] == name
                assert tensor not in shapes
                shapes[tensor] = record['shape']
                tensor_bytes += record['data_offsets'][1] - record['data_offsets'][0]
            info = file_info(p)
            info['header_sha256'] = sha(header_bytes)
            weight_files.append(info)
            print(f'{key}: hashed {name}', flush=True)
        assert set(shapes) == set(index['weight_map'])
        assert tensor_bytes == index['metadata']['total_size']
        snapshot_payload = {Path(v['path']).name: v['sha256'] for v in sources + weight_files}
        result['models'].append({
            'model_key': key, 'official_model_id': official, 'local_directory': str(path.resolve()),
            'upstream_revision': None,
            'identity_limit': 'Local bytes are pinned; no claim that an upstream commit or official weight equivalence was proved.',
            'local_snapshot_sha256': sha(json.dumps(snapshot_payload, sort_keys=True).encode()),
            'architecture': cfg.get('architectures'), 'model_type': cfg.get('model_type'),
            'layers': cfg.get('num_hidden_layers', cfg.get('num_layers')),
            'config_weight_dtype': cfg.get('torch_dtype'), 'tensor_count': len(shapes),
            'stored_tensor_elements': sum(math.prod(v) for v in shapes.values()),
            'tensor_data_bytes': tensor_bytes,
            'fp32_stored_tensor_bytes_estimate': 4 * sum(math.prod(v) for v in shapes.values()),
            'memory_estimate_limit': 'Weights only; excludes activations, runtime buffers, attention workspaces and device allocation overhead.',
            'metadata_sources': sources, 'weight_sources': weight_files,
            'GPU_qualified_for_this_protocol': False,
        })
    dump(OUT / 'cpu-model-inventory.json', result)
    print('CPU identity inventory complete', flush=True)


def load_tokenizer(key, path):
    if key.startswith('glm'):
        spec = importlib.util.spec_from_file_location('cpu_review_glm_tokenizer', path / 'tokenization_chatglm.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ChatGLM4Tokenizer.from_pretrained(str(path), local_files_only=True), {}
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=True), {'enable_thinking': False}


def tokenize():
    inventory_data = read(OUT / 'cpu-model-inventory.json')
    model_identity = {m['model_key']: m for m in inventory_data['models']}
    frames = {}
    for name in ['new-model-inputs', 'legacy-model-inputs']:
        frames[name] = [json.loads(x) for x in (OUT / f'{name}.jsonl').read_text().splitlines()]
    all_results = {'schema_version': 'cross-model-tokenizer-cpu-audit/v1',
                   'model_weights_loaded': False, 'model_forward_calls': 0,
                   'reference_fields_read_for_rendering': False, 'models': [],
                   'versions': {x: importlib.metadata.version(x) for x in ['transformers', 'tokenizers', 'torch', 'tiktoken', 'jinja2']}}
    (OUT / 'tokenized').mkdir(exist_ok=True)
    for key, official, path in MODELS:
        identity = model_identity[key]
        for row in identity['metadata_sources']:
            assert file_info(Path(row['path'])) == row
        for row in identity['weight_sources']:
            p = Path(row['path']); st = p.stat()
            assert (st.st_size, st.st_mtime_ns) == (row['bytes'], row['mtime_ns'])
        tok, kwargs = load_tokenizer(key, path)
        label_ids = {label: tok.encode(label, add_special_tokens=False) for label in ['有', '无']}
        assert all(len(v) == 1 for v in label_ids.values())
        summary = {'model_key': key, 'official_model_id': official, 'tokenizer_class': type(tok).__name__,
                   'local_snapshot_sha256': identity['local_snapshot_sha256'],
                   'candidate_tokens': {k: v[0] for k, v in label_ids.items()},
                   'template_kwargs': kwargs, 'frames': {}, 'GPU_numerical_qualification': False,
                   'free_generation_compliance_checked': False}
        for name, rows in frames.items():
            tokenized, lengths = [], []
            for row in rows:
                assert set(row['messages'][0]) == {'role', 'content'}
                prompt = tok.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True, **kwargs)
                ids = tok.encode(prompt, add_special_tokens=False)
                assert ids == tok.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True, **kwargs)
                assert 0 < len(ids) < 8192
                for label, suffix_ids in label_ids.items():
                    assert tok.encode(prompt + label, add_special_tokens=False) == ids + suffix_ids
                if key == 'qwen3-8b' and name == 'legacy-model-inputs':
                    assert sha(prompt.encode()) == row['original_8b_prompt_sha256']
                    assert ids == row['original_8b_input_ids']
                tokenized.append({'condition_id': row['condition_id'], 'model_key': key,
                                  'messages_sha256': sha(json.dumps(row['messages'], ensure_ascii=False, sort_keys=True).encode()),
                                  'chat_prompt': prompt, 'prompt_sha256': sha(prompt.encode()),
                                  'input_ids': ids, 'prompt_tokens': len(ids),
                                  'last_input_token_index': len(ids)-1, 'next_token_position': len(ids),
                                  'candidate_tokens': summary['candidate_tokens']})
                lengths.append(len(ids))
            dest = OUT / 'tokenized' / f'{key}-{name}.jsonl'
            dest.write_text(''.join(json.dumps(v, ensure_ascii=False, separators=(',', ':'))+'\n' for v in tokenized), encoding='utf-8')
            summary['frames'][name] = {'prompts': len(rows), 'candidate_boundaries': 2*len(rows),
                                      'minimum_tokens': min(lengths), 'maximum_tokens': max(lengths),
                                      'all_single_token_and_boundaries_pass': True, 'artifact': file_info(dest)}
            summary['rendered_suffix_example'] = tokenized[0]['chat_prompt'][-85:]
        all_results['models'].append(summary)
        print(f'{key}: all new and legacy tokenizer checks pass', flush=True)
    import torch
    assert not torch.cuda.is_initialized()
    all_results['cuda_initialized'] = False
    all_results['total_prompt_reconstructions'] = sum(v['prompts'] for m in all_results['models'] for v in m['frames'].values())
    all_results['total_candidate_boundaries'] = 2*all_results['total_prompt_reconstructions']
    dump(OUT / 'cpu-tokenizer-audit.json', all_results)


if __name__ == '__main__':
    if (OUT / 'manifest.json').exists():
        raise SystemExit('Delivered review draft is immutable; use a separately versioned revision.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inventory', 'tokenize'])
    args = parser.parse_args()
    (inventory if args.action == 'inventory' else tokenize)()
