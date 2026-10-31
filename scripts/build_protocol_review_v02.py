"""Export exact public/default/private protocol records and prompt text for review.

Run: PYTHONPATH=src python scripts/build_protocol_review_v02.py
This is a private human audit, never an agent input or a performance evaluation.
"""
import hashlib
import json
from pathlib import Path

from wia.resources import ROOT
from wia.tasks.loader import load_task


def main():
    output = ROOT / 'docs/releases/v0.2/protocol_review'
    output.mkdir(parents=True, exist_ok=True)
    catalogue = json.loads((ROOT / 'explore_to_achieve/wia_v1/release/catalogue.json').read_text())
    rows = []
    table = ['任务索引；详细评分参数见同目录 task_contracts.json。', '',
             '| 任务 | 主分类 | 评分器模块 / 操作 | 步数 / 推理时推进世界 |',
             '| --- | --- | --- | --- |']
    for row in catalogue['tasks']:
        loaded = load_task(row['alias'])
        definition = loaded.definition.model_dump(mode='json')
        evaluation = definition['private']['evaluation']
        defaults = definition['defaults']
        rows.append({**row, 'definition_sha256': hashlib.sha256(loaded.path.read_bytes()).hexdigest(),
                     'public': loaded.public_payload, 'defaults': defaults,
                     'evaluation': evaluation, 'requirements': definition['requirements']})
        table.append(f"| {row['alias']} | {row['family']} / {row['subclass']} | "
                     f"`{evaluation['module'].removeprefix('wia.authoring.converters.')}` / `{evaluation['operation']}` | "
                     f"{defaults['max_steps']} / {defaults['world_advances_during_inference']} |")
    (output / 'task_contracts.json').write_text(json.dumps({
        'scope': 'Private human review only; never supply this file to a benchmark policy.',
        'selection_revision': catalogue['selection_revision'], 'task_count': len(rows), 'tasks': rows},
        indent=2, ensure_ascii=False) + '\n')
    (output / 'TASK_INDEX.zh-CN.md').write_text('\n'.join(table) + '\n')
    parts = ['实际提示词模板全文。动态任务、传感器、动作边界等槽位在运行时填充；以下不是一份可直接发给模型的完整请求。', '']
    hashes = {}
    for path in sorted((ROOT / 'explore_to_achieve/prompts').rglob('*.md')):
        relative = str(path.relative_to(ROOT / 'explore_to_achieve/prompts'))
        if relative == 'README.md':
            continue
        content = path.read_text()
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        parts.extend([f'### {relative}', '', '````text', content.rstrip(), '````', ''])
    (output / 'PROMPT_TEMPLATES.zh-CN.md').write_text('\n'.join(parts))
    (output / 'prompt_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({'task_count': len(rows), 'prompt_files': len(hashes), 'output': str(output)}))


if __name__ == '__main__':
    main()
