"""Append the T52 control to the existing screenshot queue without changing old jobs.

Run: PYTHONPATH=src python scripts/append_inside_control_capture.py
The immutable receipt is reused on subsequent calls. No worker is started here.
"""
import fcntl
import json

from wia.resources import ROOT
from wia.runtime.esi_category_capture import freeze_working_code, publish_record
from wia.runtime.scene_capture_queue import read_json, write_json
from refresh_wia_captures import _definition, _record


def main():
    campaign = ROOT / 'outputs/edge_review_20261029'
    queue = campaign / 'queue'
    output = ROOT / 'outputs/release_v02_followup_20261031/capture_receipt.json'
    with (queue / '.inside_control_append.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            print(output.read_text())
            return
        catalogue = read_json(ROOT / 'explore_to_achieve/wia_v1/release/catalogue.json')
        selected = next(row for row in catalogue['tasks'] if row['alias'] == 'T52')
        definition = _definition(selected)
        frozen = freeze_working_code(queue)
        record = _record(definition, frozen, campaign)
        policy_path = queue / 'capture_runtime_policy.json'
        policy = read_json(policy_path)
        assert isinstance(policy['allowed_code_commits'], list) and policy['allowed_code_commits']
        if frozen['code_commit'] not in policy['allowed_code_commits']:
            policy['allowed_code_commits'].append(frozen['code_commit'])
            write_json(policy_path, policy)
        # Fence the new job before publication so retained GPU 7 does not close
        # its existing world merely because this new queue token appears.
        pause_path = queue / 'capture_pause.json'
        pause = read_json(pause_path)
        assert pause['state'] == 'PAUSED_BY_USER'
        if any((queue / 'queued').glob('*.json')):
            raise RuntimeError('Unexpected concurrent queued work; retain its dispatch policy.')
        write_json(output.parent / 'capture_pause_before_t52.json', pause)
        pause['authorized_gpu_pool']['state'] = 'PRESERVE_COMPLETED_WORLDS'
        worker_id = 'v02-inside-control-gpu4-20261031'
        pause['authorized_pilot'] = {'state': 'RUNNING', 'worker_id': worker_id,
            'gpu_uuid': 'GPU-1e5f5478-0045-65b1-3e09-a981a64277aa', 'job_ids': []}
        pause['reason'] = 'Preserve completed worlds on GPUs 5/6/7; dispatch only T52 to its explicit GPU 4 worker.'
        write_json(pause_path, pause)
        result = publish_record(record, queue, frozen=frozen, priority=0,
                                timeout_s=0, max_attempts=1)
        if not result.get('job_id'):
            raise RuntimeError(result.get('capture_submission_error', 'Capture publication failed'))
        pause['authorized_pilot']['job_ids'] = [result['job_id']]
        write_json(pause_path, pause)
        value = {'alias': 'T52', 'job_id': result['job_id'], 'candidate_id': record['candidate_id'],
                 'task_sha256': definition['task_sha256'], 'scene_sha256': definition['scene_sha256'],
                 'code_commit': frozen['code_commit'], 'queue_root': str(queue),
                 'priority': 0, 'timeout_s': 0, 'existing_jobs_modified': False, 'worker_id': worker_id,
                 'retention': 'The new worker must use --retain-world; no time-based termination.'}
        write_json(output, value)
        print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
