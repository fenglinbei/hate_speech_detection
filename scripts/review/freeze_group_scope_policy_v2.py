#!/usr/bin/env python3
"""Freeze the user-approved group amendment without touching experiment inputs."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDES = ROOT / 'docs/research/annotation-guidelines'
EXP = ROOT / 'exps/causal_context/general_model_evidence_applicability_v1'
OUT = EXP / 'policies/group-scope-v2'
PARENT = GUIDES / 'evidence-applicability-annotation-policy-v1.md'
DRAFT = GUIDES / 'evidence-applicability-group-scope-v2-draft1.md'
FINAL = GUIDES / 'evidence-applicability-annotation-policy-v2.md'
PARENT_SHA = '660d973c6248b17068ce44f68ebc4680515e54fd17354bd51650ace7c436557a'
DRAFT_SHA = 'f79f8c4fbf5dc1ba22b5d3aa706312c941fc42c84a6de51d90c6ffb598d77712'
BUNDLE_SHA = 'a541349cdd248032f56fad1bb2618abc9569fbe1548f3c4864691e6850d7f3b4'
VERSION = 'evidence-applicability-annotation-policy/v2'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def put(path: Path, raw: bytes) -> None:
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Refusing to overwrite a different frozen artifact: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def main() -> None:
    parent_raw, draft_raw = PARENT.read_bytes(), DRAFT.read_bytes()
    bundle_raw = (EXP / 'bundle/evidence_bundle.json').read_bytes()
    assert sha(parent_raw) == PARENT_SHA and sha(draft_raw) == DRAFT_SHA
    assert sha(bundle_raw) == BUNDLE_SHA
    parent, draft, bundle = parent_raw.decode(), draft_raw.decode(), json.loads(bundle_raw)
    body = parent[parent.index('## 1.'):parent.index('## 10.')]
    group = draft[draft.index('## 1. 可执行定义'):draft.index('## 4. 演示')]
    group = group.replace('## 1. 可执行定义', '## 4. group：实际身份对象与类别范围')
    group = group.replace('## 2. 本次具体化的范围', '### 4.1 已确认的对象范围')
    group = group.replace('## 3. 三种结果与决定顺序', '### 4.2 类别、无类别与未决')
    group = group.replace('拟议规则', '规则').replace('本稿', '本版')
    group += '以上范围是本版人工参考的语义规则，不证明原数据已完整采用相同范围。原标注未覆盖与同口径漏标必须区分；规则变化不能汇报成源数据误标。\n\n'
    body = body[:body.index('## 4.')] + group + body[body.index('## 5.'):]
    body = body.replace('| 本稿的边界处理 | 下文的保守审核流程及未决处理 | 是待负责人确认的操作规则，不能宣称为源数据原始标注指南 |',
                        '| 本版的边界处理 | 已确认的 group 身份／角色范围及保留未决的规则 | 是本轮人工参考的规则，不能宣称为源数据原始标注指南 |')
    body = body.replace('`context_insufficient`：缺判别语境 |', '`context_insufficient`：缺判别语境；`policy_changed`：group 在新规则下已裁决，差异来自规则变化 |')
    marker = '多个问题同时存在时保存完整理由，选一个真正阻止当前任务裁决的主状态。'
    body = body.replace(marker, '`policy_changed` 仅用于本次 group 范围修订后的已解决裁决，正式 group 为完整集合（可为 `[]`），不能用于 hate、未决标签或冒充源数据错误。`policy_ambiguous` 仍表示规则疑问尚未解决，标签必须为 `null`。\n\n' + marker)
    body = body.replace('负责人可以确认包含“保留未决”的规则，不必先裁清每个边界。建议确认范围为：',
                        '本版根据用户对 group 范围修订草案的明确接受冻结；未修改 hate 的语义边界。确认范围为：')
    body = body.replace('2. 对纯个体/机构冒犯、个人/机构是否映射 others、国家/民族等无法唯一归类的边界，按本稿保留 `policy_ambiguous`；不移用其他数据集的规则强裁。',
                        '2. group 按 §4 确定身份／角色对象、others、纯个人及机构本身的 `[]`。hate 的个体／机构冒犯范围，以及仍无法唯一映射的国家／民族类别，继续保留未决；不因 group 归入 others 自动扩大 hate 的攻击范围。')
    hate_section = lambda text: text[text.index('## 3. hate：'):text.index('## 4.')]
    assert hate_section(body) == hate_section(parent), 'Hate policy must remain byte-identical'
    header = f'''# 证据适用性联合审核：判定规则 v2（已确认冻结）

日期：2026-09-09（Asia/Shanghai）。

| 身份 | 当前值 |
| --- | --- |
| 冻结版本 | `{VERSION}` |
| 状态 | `FROZEN`：用户已明确接受 group 修订草案；人工重核进度另记 |
| 确认者 | 本任务用户 liaozijie |
| 确认时间 | 2026-09-08 16:48:05 UTC / 2026-09-09 00:48:05 Asia/Shanghai |
| 确认原文 | 好，接受这版草案 |
| 已接受草案 | [group-scope/v2-draft.1](evidence-applicability-group-scope-v2-draft1.md)，SHA-256 `{DRAFT_SHA}` |
| 父规则 | [判定规则 v1](evidence-applicability-annotation-policy-v1.md)，SHA-256 `{PARENT_SHA}` |
| 语义修订范围 | group 对象、others、无类别与未决；hate §3 原文保持 |

本版把用户接受的 group 修订合入完整规则。四个专门类别以外，明确作为评价、辩护或攻击对象的身份／角色归 others；异性恋这一性取向属性在此范围。无类别取决于没有实际群体身份对象，不取决于有没有出现身份词。

原 AI bundle、模型输入、原 Gold、预测及 v1 人工记录保留原身份。AI 预填仍来自 v1，不能被展示成按 v2 新生成；已确认的 hate 可沿用父版，受新范围影响的 group 需要人类明确重核。规则冻结不代表记录已完成重核；[执行记录](../../../exps/causal_context/general_model_evidence_applicability_v1/execution-status.md)登记线上切换与进度。

'''
    footer = '''## 10. 本次版本衔接

本次 group 对象定义适用于整个授权审核范围。32 个查询、280 个去重示例及 320 个查询—示例关系的 group 任务登记新规则核对范围；同一对象共享引用去重。现有确认、草稿、材料快照、暴露历史、作者身份与原始时间保留，迁移不代替人工确认。对象 ID 与案例 ID 清单见冻结回执及附属规则文件。

hate 判断、词典定义和语义适配不会仅因 group 政策换版自动作废。若人类另行改变原文解释、共同依据或其他任务判断，则按实际依赖版本重核。仅重核 group 时保留先前 hate 依据；修改共同字段或 hate 应显式重新打开。

`policy_changed` 将新口径下已裁决的 group 与同口径误标区分。导出按任务记录采用的政策与可用性；旧 group 不能在未经重核时当成新规则参考。此前已见过的结果不会伪装成未见；原材料快照保持不可覆盖。

本版没有重新生成 AI 标签，没有读取 reserve 正文，没有修改模型提示或执行输入实验。若今后按新口径改变示例答案或模型任务提示，应作为另一个明确登记的输入实验，不回写既有预测。

## 11. 冻结回执

[冻结回执](../../../exps/causal_context/general_model_evidence_applicability_v1/policies/group-scope-v2/policy_freeze.json)记录确认、父规则、批准草案、最终文件、AI bundle 与受影响 ID 清单的身份。批准草案和父版原字节保留；线上可执行规则文件另附全文及相同 SHA-256。
'''
    final_raw = (header + body + footer).encode()
    final_sha = sha(final_raw)
    impacted = sorted(oid for oid, obj in bundle['objects'].items() if obj['kind'] in {'query', 'demo', 'relation'})
    case_ids = list(bundle['order'])
    assert len(impacted) == 632 and len(case_ids) == 32
    approval = {'reviewer_id': 'liaozijie', 'confirmed_at': '2026-09-08T16:48:05.067209+00:00',
                'statement': '好，接受这版草案', 'accepted_draft_sha256': DRAFT_SHA}
    impact = {'tasks': ['group'], 'object_ids': impacted, 'case_ids': case_ids,
              'reason': 'group 身份/角色对象和无类别范围对全体适用；既有 hate 与无关词典语义保留',
              'object_counts': dict(Counter(bundle['objects'][oid]['kind'] for oid in impacted)),
              'excluded_kinds': ['definition', 'hit'], 'reserve_body_read': False}
    amendment = {'schema_version': 'general-model-evidence-policy-amendment/v1', 'version': VERSION,
                 'title': '证据适用性判定规则 v2 · group 对象范围', 'sha256': final_sha,
                 'document_text': final_raw.decode(), 'parent_policy': {'version': bundle['policy']['version'], 'sha256': PARENT_SHA},
                 'approval': approval, 'impact': impact, 'bundle_sha256': BUNDLE_SHA}
    amendment_raw = json_bytes(amendment)
    receipt = {'schema_version': 'evidence-policy-freeze/v2', 'status': 'frozen', 'approval': approval,
               'parent_policy': amendment['parent_policy'], 'accepted_draft': {'path': str(DRAFT.relative_to(ROOT)), 'sha256': DRAFT_SHA},
               'frozen_policy': {'path': str(FINAL.relative_to(ROOT)), 'version': VERSION, 'sha256': final_sha},
               'amendment': {'path': str((OUT / 'policy_amendment.json').relative_to(ROOT)), 'sha256': sha(amendment_raw)},
               'bundle_sha256': BUNDLE_SHA, 'impact': impact,
               'hate_section_unchanged': True, 'human_records_written': False, 'ai_labels_regenerated': False,
               'generator_sha256': sha(Path(__file__).read_bytes())}
    put(FINAL, final_raw)
    put(OUT / 'policy_amendment.json', amendment_raw)
    put(OUT / 'policy_freeze.json', json_bytes(receipt))
    print(json.dumps({'version': VERSION, 'policy_sha256': final_sha, 'amendment_sha256': sha(amendment_raw),
                      'object_counts': impact['object_counts'], 'case_count': len(case_ids), 'human_records_written': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
