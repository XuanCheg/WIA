"""Summarize exact furnished-scene capture receipts without modifying Review.

Run: python -B scripts/report_wia_furnished_round.py
Reads latest native attempts and main-authored visual reviews. It neither
qualifies tasks nor runs actions, and never changes user feedback.


Canonical invocation: python -m wia.authoring.commands.report_wia_furnished_round
"""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from wia.resources import ROOT
from wia.authoring.commands._inputs import ARTIFACT_ROOT
OUT=ARTIFACT_ROOT / 'wia_v1_20261024_furnished'
QUEUE=ARTIFACT_ROOT / 'wia_v1_20261023/capture_campaign/queue'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    # Explicit selection separates primary review entries from superseded pilots.
    selected=read(OUT/'selected_manifest.json')
    if selected is None:
        entries=[]
        for name in ['furnished_batch_a_manifest.json','furnished_batch_b_manifest.json']:
            entries.extend(read(OUT/name,{'entries':[]})['entries'])
        entries.extend(e for e in read(OUT/'spatial_pilot_manifest.json')['entries'] if e['document_alias']=='T30')
        selected={'schema':'wia_variant_manifest_v1','entries':entries}
    indexed={}
    for path in sorted(OUT.glob('*records.submission.json')):
        for item in read(path)['submitted']:
            indexed[(item['document_alias'],item['variant_id'])]=item
    reviews=read(OUT/'visual_reviews.json',{})
    rows=[]
    for entry in selected['entries']:
        key=(entry['document_alias'],entry['variant_id'])
        submitted=indexed.get(key,{})
        job=QUEUE/'jobs'/submitted.get('job_id','missing')
        status=read(job/'status.json',{})
        receipts=sorted((job/'attempts').glob('*/preview_complete.json')) if job.exists() else []
        receipt_path=receipts[-1] if receipts else None
        receipt=read(receipt_path,{}) if receipt_path else {}
        setup=receipt.get('interaction_setup') or {}
        optics=(receipt.get('initial_agent_bundle') or {}).get('review_diagnostics') or {}
        # Capture formats keep the read-only mirror diagnostics outside or
        # inside the frozen bundle; do not infer success from missing evidence.
        if not optics:
            optics=receipt.get('initial_review_diagnostics') or {}
        selection=(optics.get('geometry') or {}).get('optical_selection') or {}
        review=reviews.get(entry['variant_id'],{})
        bad_supports={name:value for name,value in setup.get('supported_items',setup.get('inventory_supports',{})).items()
                      if value.get('accepted') is not True}
        rows.append({'document_alias':key[0],'variant_id':key[1],'label':entry['label'],
            'task_file':entry['task_file'],'scene_file':entry['scene_file'],
            'task_sha256':entry.get('task_sha256'),'scene_sha256':entry.get('scene_sha256'),
            'job_id':submitted.get('job_id'),'state':status.get('state','NOT_SUBMITTED'),
            'receipt_file':str(receipt_path) if receipt_path else None,
            'images_complete':receipt.get('complete') is True,
            'initial_setup_accepted':setup.get('accepted'),
            'failed_supports':bad_supports,'robot_initialization':setup.get('robot',setup.get('robot_initialization')),
            'native_optical_selection_accepted':selection.get('accepted'),
            'native_optical_errors':selection.get('errors'),
            'main_visual_review':review.get('decision','pending'), 'main_visual_notes':review.get('notes'),
            'human_accepted':None,'robot_actions':receipt.get('robot_actions'),'model_calls':receipt.get('model_calls'),
            'physical_success_verified':False,'scientific_controls_verified':False,
            'images':receipt.get('images',{}),'duration_s':receipt.get('duration_s'),
            'failure':status.get('error'),'review_role':entry.get('review_role','primary')})
    states=dict(Counter(row['state'] for row in rows))
    summary={'selected_versions':len(rows),'original_proposals':len({row['document_alias'] for row in rows}),
        'complete_images':sum(row['images_complete'] for row in rows),
        'initial_setup_accepted':sum(row['initial_setup_accepted'] is True for row in rows),
        'initial_setup_rejected':sum(row['initial_setup_accepted'] is False for row in rows),
        'main_visual_accepted':sum(row['main_visual_review']=='accepted_initial_design' for row in rows),
        'states':states}
    result={'schema':'wia_furnished_initial_review_v1','updated_at':datetime.now(timezone.utc).isoformat(),
        'evidence_scope':'Native initialization/settling/images only; no operational success or scientific qualification.',
        'summary':summary,'entries':rows}
    (OUT/'scene_review_status.json').write_text(json.dumps(result,indent=2)+'\n')
    fields=['document_alias','variant_id','state','images_complete','initial_setup_accepted','main_visual_review',
            'native_optical_selection_accepted','robot_actions','model_calls','receipt_file','task_file','scene_file']
    with (OUT/'scene_review_status.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({key:row[key] for key in fields} for row in rows)
    print(json.dumps(summary))


if __name__=='__main__':main()
