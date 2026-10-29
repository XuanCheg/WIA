#!/usr/bin/env python3
"""Audit saved actual rollouts: boundaries, sensor provenance and public memory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def audit(run):
    episode = run/'episode'
    summary = json.loads((episode/'summary.json').read_text())
    steps = lines(episode/'steps.jsonl')
    inputs = [x for x in lines(episode/'model_inputs.jsonl') if x.get('event')=='step_input']
    errors, notes = [], []
    def sensor_identity(item):
        return item.get('model_sensor_snapshot_id',item['observation']['snapshot_id'])
    initial_sensor_id = sensor_identity(inputs[0]) if inputs else None
    initial_depth = inputs[0].get('raw_depth') if inputs else None
    def camera_geometry(item):
        marker = 'Calibrated camera geometry: '
        return (json.loads(item['prompt'].split(marker, 1)[1].split('\n', 1)[0])
                if marker in item['prompt'] else None)
    initial_geometry = camera_geometry(inputs[0]) if inputs else None
    accepted = {s['step']:s for s in steps if s.get('memory_update')}
    for item in inputs:
        step = item['step']
        obs = item['observation']
        sensor_id = sensor_identity(item)
        expected_id = initial_sensor_id if summary.get('observation_condition')=='initial_view_only' else obs['snapshot_id']
        if sensor_id != expected_id:
            errors.append(f'Decision {step}: wrong supplied sensor identity')
        depth = item.get('raw_depth')
        if depth and depth['snapshot_id'] != sensor_id:
            errors.append(f'Decision {step}: depth belongs to a different sensor snapshot')
        if summary.get('observation_condition') == 'initial_view_only':
            if depth != initial_depth or camera_geometry(item) != initial_geometry:
                errors.append(f'Decision {step}: frozen depth metadata or calibration changed')
        text = item['prompt']
        for key in ('selected_region','target_position_requested','source_esi_question','bbox_xyxy_px','visible_pixels','oracle_replay','seg_instance'):
            if key in text or key in json.dumps(obs):
                errors.append(f'Decision {step}: private field {key} reached model input')
        if summary.get('memory_mode')=='model_notes':
            marker='Public observation notes: '
            given=json.loads(text.split(marker,1)[1].split('\n',1)[0])
            expected=[accepted[n]['memory_update'] for n in sorted(accepted) if n<step][-4:]
            if given != expected or len(given)>4:
                errors.append(f'Decision {step}: public memory differs from prior accepted model notes')
    motion_count = 0
    for item in steps:
        step=item['step']
        before=json.loads((episode/item['observation']).read_text())
        note=item.get('memory_update')
        if note:
            raw=item['attempts'][-1]['raw_output']
            provider=json.loads(raw)
            if note['text'] != provider['memory_update'] or len(note['text'])>512:
                errors.append(f'Decision {step}: note changed or exceeded bound')
            source=next((x for x in inputs if x['step']==step),None)
            if note['action_step'] != before['action_step'] or note['snapshot_id'] != sensor_identity(source):
                errors.append(f'Decision {step}: note not tied to actual supplied snapshot')
            notes.append(note)
        if item.get('action_result') is None:
            continue
        motion_count+=1
        if not item.get('next_observation'):
            errors.append(f'Decision {step}: no subsequent authoritative observation saved')
            continue
        after=json.loads((episode/item['next_observation']).read_text())
        if after['action_step'] != before['action_step']+1 or after['snapshot_id'] != before['snapshot_id']+1:
            errors.append(f'Decision {step}: accepted macro did not create exactly one boundary')
    if motion_count != summary['macro_actions']:
        errors.append('Recorded macro count differs from summary')
    depth_responses = [x for x in lines(episode/'model_inputs.jsonl')
                       if x.get('event') == 'mcp_response' and x.get('passed_to_model') and
                       x.get('tool') in ('read_depth_pixels', 'read_depth_region')]
    for response in depth_responses:
        supplied = next((x for x in inputs if x['step'] == response['step'] and
                         x.get('attempt') == response.get('attempt')), None)
        envelope = response.get('result') or {}
        result = envelope.get('structured_content') or envelope
        if supplied and result.get('ok'):
            sensor = (supplied.get('raw_depth') or {}).get('sensors', {}).get(result.get('sensor'), {})
            if result.get('snapshot_id') != sensor_identity(supplied) or result.get('sha256') != sensor.get('sha256'):
                errors.append(f"Decision {response['step']}: depth tool returned a different snapshot or array")
    frozen_images=0
    rgb_sources=0
    for item in inputs:
        directory = episode/f"step_{item['step']:04d}"
        source_step = inputs[0]['step'] if summary.get('observation_condition') == 'initial_view_only' else item['step']
        source = episode/f'step_{source_step:04d}'/'before'
        projection = json.loads((directory/'model_inputs.json').read_text())
        for entry in projection:
            model_image = episode/entry['path']
            canonical_image = source/(entry['name']+'.png')
            if entry['transform'] != 'identity':
                errors.append(f"Decision {item['step']}: unexpected RGB transformation in these native conditions")
                continue
            rgb_sources += 1
            if hashlib.sha256(model_image.read_bytes()).digest() != hashlib.sha256(canonical_image.read_bytes()).digest():
                with Image.open(model_image) as model, Image.open(canonical_image) as canonical:
                    if model.size != canonical.size or model.convert('RGB').tobytes() != canonical.convert('RGB').tobytes():
                        errors.append(f"Decision {item['step']}: {entry['name']} differs from its actual sensor source")
    if summary.get('observation_condition')=='initial_view_only' and inputs:
        first=episode/f"step_{inputs[0]['step']:04d}"/'model_images'
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in first.glob('*.png')}
        for directory in episode.glob('step_*/model_images'):
            for image in directory.glob('*.png'):
                frozen_images+=1
                if hashes.get(image.name)!=hashlib.sha256(image.read_bytes()).hexdigest():
                    errors.append(f'{image.parent.parent.name}: frozen RGB changed')
    return {'run':run.name,'passed':not errors,'macro_boundaries_checked':motion_count,
            'model_inputs_checked':len(inputs),'accepted_model_notes_checked':len(notes),
            'depth_tool_responses_checked':len(depth_responses),
            'model_rgb_sources_checked':rgb_sources,
            'frozen_rgb_files_checked':frozen_images,'errors':errors,
            'limitation':'Checks supplied metadata/prompts and stored RGB; does not infer semantic recognition.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    rows=[audit(p.parent.parent) for p in sorted(args.root.glob('*/episode/summary.json'))]
    result={'schema':'actual_rollout_contract_audit_v1','passed':all(r['passed'] for r in rows),'records':rows}
    (args.root/'rollout_contract_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'runs':len(rows),'errors':[r for r in rows if r['errors']]}))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
