#!/usr/bin/env python3
"""Reconstruct and freeze the Q01 mechanism protocol without loading weights."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.general_model_evidence_evaluation import require, file_sha, json_bytes, jsonl, sha, write_output
from diagnostics.q01_mechanism_inputs import (select_contexts, locate, build_requests, schedule, budget,
    DESCRIPTOR, digest, LAYERS, MAIN_ROLES, capture_specs)
from diagnostics.q01_mechanism_package import (BASE, WORK, FREEZE, PUBLIC, PARENT, PARENT_HASH,
    read_json, read_lines, load_frozen)

PROTOCOL = ROOT / 'docs/research/experiment-plans/q01-local-mechanism-protocol-v1.md'
NEW_CODE = ('src/diagnostics/q01_mechanism_inputs.py', 'src/diagnostics/q01_mechanism_hooks.py',
            'src/diagnostics/q01_mechanism_scoring.py', 'src/diagnostics/q01_mechanism_package.py',
            'src/diagnostics/q01_mechanism_execution.py', 'scripts/review/prepare_q01_local_mechanism.py',
            'scripts/review/run_q01_local_mechanism.py', 'scripts/review/test_q01_local_mechanism.py',
            'scripts/review/schedule_q01_gpu_window.py')


def csv_data(rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, list(rows[0]), lineterminator='\n'); writer.writeheader()
    for row in rows:
        writer.writerow({k: (__import__('json').dumps(v, ensure_ascii=False, separators=(',', ':'))
                            if isinstance(v, (dict, list)) else v) for k, v in row.items()})
    return out.getvalue().encode('utf-8-sig')


def build():
    from scripts.review.freeze_evidence_functional_queries import load_frozen as load_parent
    require(file_sha(PARENT / 'manifest.json') == PARENT_HASH, 'parent stage-1 manifest differs')
    parent, original, _ = load_parent(PARENT)
    run = PARENT.parent / 'run-stage-1-01'
    run_state = read_json(run / 'run_manifest.json')
    require(run_state['status'] == 'complete' and run_state['numerical_validation_passed'], 'source run is not sealed')
    raw_path = run / parent['raw_pass'] / 'scores.jsonl'
    require(file_sha(raw_path) == run_state['raw_scores_sha256'], 'historical score seal differs')
    contexts, index = select_contexts(original)
    history_by_id = {r['record_id']: r for r in read_lines(raw_path)}
    history = [history_by_id[c['record_id']] for c in contexts]
    package = Path(parent['runtime_parent_plan']['package_path'])
    model_record = read_json(package / 'models.json')[0]
    tokenizer_inventory = model_record['tokenizer_inventory']
    tokenizer_path = ROOT / tokenizer_inventory['logical_repo_path']
    for item in tokenizer_inventory['files']:
        p = tokenizer_path / item['path']
        require(p.is_file() and not p.is_symlink() and file_sha(p) == item['sha256'], 'tokenizer source is not the pinned regular file')
    for key in ('USE_TORCH', 'USE_TF', 'USE_FLAX', 'USE_TORCH_XLA'): os.environ[key] = '0'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False, use_fast=True)
    positions = {c['record_id']: locate(c, tokenizer) for c in contexts}
    requests, proofs = build_requests(contexts, index, positions)
    passes = schedule(requests, contexts, parent['catalog'])
    counts = budget(requests, passes)
    boundaries = []
    for c in contexts:
        for candidate in parent['catalog'][c['task']]:
            ids = tokenizer.encode(candidate['canonical_answer'], add_special_tokens=False)
            require(ids == candidate['answer_token_ids'], 'candidate token identity differs')
            require(tokenizer.encode(c['prompt_text'] + candidate['canonical_answer'], add_special_tokens=False)
                    == c['prompt_token_ids'] + ids, 'candidate concatenation boundary changed')
            boundaries.append({'source_record_id': c['record_id'], 'candidate_id': candidate['candidate_id'],
                               'prompt_sha256': c['prompt_sha256'], 'candidate_start': c['prompt_tokens'],
                               'prediction_positions': list(range(c['prompt_tokens'] - 1, c['prompt_tokens'] + len(ids))),
                               'answer_token_ids': ids, 'eos_token_id': parent['eos_token_id'],
                               'main_score_excludes_eos': True, 'concatenation_stable': True})
    reference = next(r for r in read_json(PARENT / 'analysis_references.json') if r['query_id'] == 'FD-3169-Q01')
    position_csv = []
    for p in positions.values():
        for role, r in p['roles'].items():
            position_csv.append({'source_record_id': p['record_id'], **p['binding'], 'encoding': p['encoding'],
                'probe_id': p['probe_id'], 'prompt_tokens': p['prompt_tokens'], 'prompt_sha256': p['prompt_sha256'],
                'role': role, 'applicable': r['applicable'], 'semantic_char_span': r['semantic_char_span'],
                'token_positions': r['token_positions'], 'position_ids': r['position_ids'], 'token_ids': r['token_ids'],
                'token_char_spans': r['token_char_spans'], 'candidate_start': p['candidate_start'],
                'first_prediction_position': p['first_prediction_position']})
    matrix_summary = Counter((r['category'], r['kind'], r['module'], r['layer'], r['role'], r['encoding'], r['probe_id']) for r in requests)
    compact = [{'category': k[0], 'kind': k[1], 'module': k[2], 'layer': k[3], 'role': k[4],
                'encoding': k[5], 'probe_id': k[6], 'prompt_requests': n, 'candidates': n * 2}
               for k, n in sorted(matrix_summary.items(), key=lambda item: str(item[0]))]
    sources = dict(parent['source_files'])
    manifest = read_json(PARENT / 'manifest.json')
    for name, h in manifest['artifacts'].items(): sources[str((PARENT / name).relative_to(ROOT))] = h
    for p in (PARENT / 'manifest.json', run / 'run_manifest.json', raw_path, PROTOCOL,
              BASE / 'functional-query-results-v1/current.json', BASE / 'functional-query-results-v1/INTERPRETATION.md',
              ROOT / 'docs/research/experiment-plans/q01-local-mechanism-protocol-draft-v1.md'):
        sources[str(p.relative_to(ROOT))] = file_sha(p)
    code = dict(parent['code_sha256'])
    for name in NEW_CODE: sources[name] = code[name] = file_sha(ROOT / name)
    for name in ('.conda/stage1-p0/lib/python3.11/site-packages/transformers/models/qwen3/modeling_qwen3.py',
                 '.conda/stage1-p0/lib/python3.11/site-packages/torch/nn/modules/module.py'):
        sources[name] = code[name] = file_sha(ROOT / name)
    protocol = {'scope': 'one exposed Q01, four selected input contrasts; exploratory local intervention',
        'user_authorization': '可以开始 正式协议、逐提示位置表、双向干预与控制矩阵、评分规范、hook 实现和验收调度 的交付，待确认项清询问我',
        'scientific_inputs_are_exact_parent_bytes': True, 'new_human_reference_created': False,
        'main_score': 'original/answer_sum', 'margin': 'non-hate minus hate', 'layers_zero_based': list(LAYERS),
        'primary_roles': list(MAIN_ROLES), 'module': 'block output after both residual additions',
        'donor': 'prompt-only FP32 state; shared across candidate branches; same F/surface/encoding/probe',
        'full_vector_replacement_without_scaling': True, 'ncc_fixed_target': 'original answer-mean target with recipient prior held fixed',
        'ncc_recalibration_query_hehe': 'not_applicable', 'ncc_all_five_single_and_loo': True,
        'selection': 'C and I each top one by original-sum worst-group/direction closeness, at most two after deduplication',
        'selection_requires_residual_improvement_beyond_conservative_numeric_bound': True,
        'failed_secondary_views_do_not_replace_winner': True, 'refinement_requires_separate_freeze': True,
        'refinement_maximum_units': 12, 'gpu_acceptance_pending': True, 'mechanism_ready': False,
        'query_gold_loaded_during_scoring': False, 'formal_test_or_reserve_access': False,
        'complete_specification_file': str(PROTOCOL.relative_to(ROOT))}
    files = {'contexts.jsonl': jsonl(contexts), 'positions.jsonl': jsonl(list(positions.values())),
             'positions.csv': csv_data(position_csv), 'candidate-boundaries.jsonl': jsonl(boundaries),
             'pair-proofs.jsonl': jsonl(proofs), 'requests.jsonl': jsonl(requests),
             'intervention-matrix.csv': csv_data(requests), 'compact-matrix.csv': csv_data(compact),
             'capture-specs.jsonl': jsonl([{'record_id': c['record_id'], 'specs': capture_specs(positions[c['record_id']])} for c in contexts]),
             'historical-scores.jsonl': jsonl(history), 'analysis-reference.json': json_bytes(reference),
             'protocol.json': json_bytes(protocol), 'budget.json': json_bytes(counts),
             'acceptance-schedule.json': json_bytes(passes), 'PROTOCOL.md': PROTOCOL.read_bytes()}
    files['scoring-spec.json'] = json_bytes({'margin': 'non-hate-minus-hate',
        'raw_modes': ['answer_sum', 'answer_mean', 'total_with_eos', 'mean_with_eos'],
        'encodings': ['original', 'ab_forward', 'ab_reverse'], 'primary': 'original/answer_sum',
        'C_coefficients_H_A': [0.5, 0.5], 'I_coefficients_H_A': [1.0, -1.0],
        'R': 'patched(O<-N)-O', 'K': 'N-patched(N<-O)', 'raw_reverse_shift_saved': True,
        'ncc_background': 'logit(mean(sigmoid(original_answer_mean_margin_each_probe)))',
        'probe_texts': {'empty': '', 'space': ' ', 'na': 'N/A', 'mask': '[MASK]', 'lorem': 'Lorem ipsum'},
        'fixed_prior': 'subtract recipient unpatched prior from recipient, patched and target donor',
        'query_hehe_recalibrated_ncc': None, 'recalibrated_aggregate_single_and_loo': True,
        'physical_shared_terms_merge_before_bounds': True,
        'raw_margin_bound_epsilon_units': 1, 'background_bound_epsilon_units': 1,
        'raw_C_bound_epsilon_units': 2, 'raw_I_bound_epsilon_units': 4,
        'recalibrated_ncc_C_bound_epsilon_units': 4, 'recalibrated_ncc_I_bound_epsilon_units': 8,
        'numeric_policy': parent['numeric_policy'], 'statistical_confidence_claimed': False})
    files['implementation-snapshot.json'] = json_bytes({name: {'sha256': h, 'text': (ROOT / name).read_text()}
                                                       for name, h in code.items()})
    runtime_parent = deepcopy(parent['runtime_parent_plan'])
    runtime_parent['blocks'] = [{k: c[k] for k in DESCRIPTOR} for c in contexts]
    plan = {'schema_version': 'q01-mechanism-plan/v1', 'status': 'frozen',
        'parent_plan_id': parent['plan_id'], 'parent_manifest_sha256': PARENT_HASH,
        'source_files': sources, 'code_sha256': code, 'catalog': parent['catalog'],
        'numeric_policy': parent['numeric_policy'], 'runtime': parent['config']['runtime'],
        'runtime_parent_plan': runtime_parent, 'eos_token_id': parent['eos_token_id'], 'pad_token_id': parent['pad_token_id'],
        'schedule': passes, 'budget': counts, 'data_sha256': {name: sha(data) for name, data in files.items()},
        'device_allocation': 'bind 2-4 distinct idle physical GPUs at a separately authorized future launch',
        'same_allocation_required_for_paused_run_resume': True, 'analysis_requires_complete_run_and_all_gates': True,
        'query_gold_loaded_during_scoring': False, 'formal_test_or_reserve_access': False, 'mechanism_ready': False}
    plan['plan_id'] = 'q01-mechanism-' + digest(plan)
    files['plan.json'] = json_bytes(plan)
    files['cpu-input-audit.json'] = json_bytes({'status': 'passed', 'prompt_replays': 96, 'source_pair_proofs': len(proofs),
        'original_O_N_pair_proofs': 64, 'neutral_N1_N2_pair_proofs': 32, 'candidate_boundaries': len(boundaries),
        'position_roles': len(position_csv), 'source_files': len(sources), 'torch_imported': 'torch' in sys.modules,
        'weights_loaded': False, 'qwen_forward_executed': False, 'gpu_forward_executed': False})
    return files, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check'))
    parser.add_argument('--output', type=Path, default=FREEZE)
    args = parser.parse_args()
    files, sources = build()
    if args.command == 'check':
        manifest = read_json(args.output / 'manifest.json')
        require(manifest['source_files'] == sources and set(manifest['artifacts']) == set(files), 'reconstruction inventory differs')
        for name, data in files.items(): require((args.output / name).read_bytes() == data, 'reconstruction differs: ' + name)
        load_frozen(args.output)
    else:
        files['manifest.json'] = json_bytes({'schema_version': 'q01-mechanism-freeze/v1', 'status': 'frozen',
            'created_at': datetime.now(timezone.utc).isoformat(), 'source_files': sources,
            'artifacts': {name: sha(data) for name, data in files.items()}})
        write_output(args.output, files)
    print(__import__('json').dumps({'status': args.command, 'budget': read_json(args.output / 'budget.json'),
                                  'gpu_forward_executed': False}, ensure_ascii=False))


if __name__ == '__main__': main()
