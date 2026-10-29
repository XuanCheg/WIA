#!/usr/bin/env python3
"""Build measured native/model CSV, JSON and standalone timing plots."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


METRICS = ('macro_wall_time', 'controller_env_step_count', 'controller_step_wall_time',
           'native_sim_step_wall_time', 'render_wall_time', 'sensor_read_wall_time',
           'observation_wall_time', 'observation_capture_inclusive_wall_time',
           'observation_serialization_time', 'observation_rpc_wall_time', 'video_encode_wall_time',
           'video_finalize_wall_time', 'recording_source_write_wall_time', 'image_encode_wall_time',
           'model_image_preparation_time', 'model_input_preparation_time', 'temporary_workspace_preparation_time',
           'model_process_startup_time', 'model_application_startup_time',
           'depth_tool_initialization_time', 'depth_tool_wall_time', 'depth_tool_compute_wall_time',
           'model_and_tool_wall_time', 'rpc_queue_wait_time', 'total_episode_wall_time')
PROFILES = {'Before': 'A-baseline-replay', 'After': 'D-metric-optimized'}
COUNTS = ('explicit_render_count', 'sensor_read_count', 'full_observation_captures',
          'camera_readiness_retries', 'reused_step_capture_count')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    output = root/'report'
    output.mkdir(exist_ok=True)
    profiles, actions, agents = [], [], []
    reference_commands = None
    for profile, name in PROFILES.items():
        path = root/name/'performance_summary.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        trajectory = [row for row in data['actions'] if row['label'].startswith('trajectory_')]
        commands = [row['command'] for row in trajectory]
        reference_commands = commands if reference_commands is None else reference_commands
        same_commands = commands == reference_commands
        timings = {key: sum(row['timings'].get(key, 0.) for row in trajectory) for key in METRICS}
        for key in METRICS:
            if not any(key in row['timings'] for row in trajectory):
                timings[key] = None
        timings['total_episode_wall_time'] = sum(row['wall_time_s'] for row in trajectory)
        # This replay makes no model/RPC calls. Report their absence explicitly.
        for key in METRICS:
            if key.startswith(('model_', 'depth_tool_', 'temporary_workspace_', 'rpc_queue_')):
                timings[key] = 0.
        timings['observation_rpc_wall_time'] = 0.
        settings = data['metadata'].get('performance', {})
        if settings.get('human_recording') is False:
            timings['video_encode_wall_time'] = 0.
            timings['video_finalize_wall_time'] = 0.
            timings['recording_source_write_wall_time'] = 0.
        row = dict(profile=profile, run=name, task_success=data['task_success'],
            same_trajectory_commands=same_commands, completed_commands=len(trajectory),
            all_trajectory_motions_completed=all(row['result']['success'] for row in trajectory),
            global_camera_count=len(data['metadata'].get('global_camera_names', ()))
                if 'global_camera_names' in data['metadata'] else 6 if data['metadata'].get('scene_camera') else 0,
            timings=timings, performance_settings=settings,
            counts={key: sum(item['timings'].get(key, 0) for item in trajectory)
                    if any(key in item['timings'] for item in trajectory) else None for key in COUNTS},
            single_boundary_per_command=all(item['action_boundaries'] == 1 for item in trajectory),
            controller_steps=int(sum(item['timings'].get('controller_env_step_count', 0) for item in trajectory)))
        profiles.append(row)
        for item in data['actions']:
            result, accounting = item['result'], item['result'].get('execution', {})
            actual = result.get('actual', {})
            scalar = actual.get('delta_m', actual.get('distance_m', actual.get('angle_deg')))
            effective = result['compilation'].get('effective_distance_m', result['compilation'].get('effective_angle_deg'))
            actions.append(dict(profile=profile, label=item['label'], action=item['command']['action'],
                requested=item['command'].get('distance_m', item['command'].get('angle_deg')),
                effective=effective, actual=scalar, scalar_error=None if scalar is None or effective is None else abs(abs(scalar)-effective),
                completed=result['success'], reason=result.get('reason'), wall_time_s=item['wall_time_s'],
                controller_env_steps=item['timings'].get('controller_env_step_count'),
                action_boundaries=item['action_boundaries'], equivalent_fixed_primitives=result['compilation']['planned_count'],
                actual_fixed_primitive_executions=accounting.get('actual_fixed_primitive_executions', accounting['executed_count']),
                metric_target_executions=accounting.get('metric_target_executions', 0),
                final_position_error_m=result.get('position_error_m'),
                final_orientation_error_deg=result.get('orientation_error_deg', result.get('yaw_error_deg')),
                stop_steps=result.get('stop_steps'), recovery_steps=result.get('recovery_steps'),
                recovery_completed=result.get('recovery_completed'),
                cancelled=result.get('cancelled'), timeout=result.get('timeout')))
    for label, name in [('before', 'A-baseline-agent'), ('after', 'D-optimized-agent')]:
        path = root/name/'episode/summary.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        measurement = root/name/'episode/performance_agent.json'
        counters = json.loads(measurement.read_text())['metrics'] if measurement.exists() else summary.get('timings', {})
        native = root/name/'performance_counters.json'
        native_counters = json.loads(native.read_text())['metrics'] if native.exists() else {}
        metrics = {key: counters.get(key, native_counters.get(key)) for key in METRICS}
        settings_path = root/name/'metadata.json'
        if settings_path.exists():
            settings = json.loads(settings_path.read_text()).get('performance', {})
            if settings.get('human_recording') is False:
                for key in ('video_encode_wall_time', 'video_finalize_wall_time', 'recording_source_write_wall_time'):
                    metrics[key] = 0.
        if 'image_encode_wall_time' not in counters and counters.get('model_input_preparation_time') is not None:
            metrics['image_encode_wall_time'] = 0.
        equivalent = summary.get('equivalent_fixed_primitives')
        if equivalent is None:
            receipts = [json.loads(line).get('action_result', {}).get('macro')
                        for line in (root/name/'episode/steps.jsonl').read_text().splitlines()]
            equivalent = sum(item['compilation']['planned_count'] for item in receipts if item)
        agents.append(dict(phase=label, run=name, model=summary['agent']['model'],
            reasoning_effort=summary['agent']['reasoning_effort'], model_calls=summary['model_calls'],
            agent_decisions=summary['agent_decisions'], macro_actions=summary['macro_actions'], task_success=summary['task_success'],
            termination_reason=summary['termination_reason'], timings=metrics,
            actual_fixed_primitive_executions=summary.get('actual_fixed_primitive_executions', summary.get('primitive_actions')),
            equivalent_fixed_primitives=equivalent,
            metric_target_executions=summary.get('metric_target_executions', 0),
            controller_env_steps=summary.get('controller_env_steps', summary.get('physics_steps')),
            depth_tool_calls=counters.get('depth_tool_calls'), image_encode_count=counters.get('image_encode_count', 0),
            native_totals_include_idle_steps=True, action_execution_wall_time=counters.get('action_execution_wall_time')))
    snapshots = []
    for name in ('snapshot-sequential-legacy', 'snapshot-sequential-optimized'):
        path = root/name/'measurement.json'
        if path.exists():
            snapshots.append(json.loads(path.read_text()))
    report = dict(timing_semantics='inclusive; nested timers overlap and must not be added',
        native_episode_definition='six identical grasp/lift commands; diagnostic probes excluded',
        simulation_accuracy='unchanged env.step, physics/controller rates, contacts, native IK and safety limits',
        profiles=profiles, actions=actions, agents=agents, snapshot_preparation=snapshots,
        limitations=['Single native replay per profile and one Agent run per phase; no statistical inference speedup claim.',
            'Native sim.step includes rendering internally; explicit render_wall_time measures extra renders only.',
            'A replay recorder encoder wall time and baseline Agent RPC queue wait were not independently instrumented; null means unmeasured.',
            'The baseline diagnostic base probe uses BASE_BACK; optimized validation uses BASE_FWD. It is excluded from paired trajectory comparisons.',
            'At the user request, validation is before/after only; intermediate combinations are not run.'])
    (output/'timing_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    def write_csv(name, rows):
        if not rows:
            return
        with (output/name).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_csv('native_profiles.csv', [dict(profile=row['profile'], task_success=row['task_success'],
        controller_steps=row['controller_steps'], **row['timings']) for row in profiles])
    write_csv('native_actions.csv', actions)
    write_csv('agent_timings.csv', [dict(phase=row['phase'], model_calls=row['model_calls'],
        agent_decisions=row['agent_decisions'], macro_actions=row['macro_actions'],
        equivalent_fixed_primitives=row['equivalent_fixed_primitives'],
        actual_fixed_primitive_executions=row['actual_fixed_primitive_executions'],
        metric_target_executions=row['metric_target_executions'], controller_env_steps=row['controller_env_steps'],
        task_success=row['task_success'], action_execution_wall_time=row['action_execution_wall_time'],
        depth_tool_calls=row['depth_tool_calls'], image_encode_count=row['image_encode_count'], **row['timings']) for row in agents])
    write_csv('snapshot_preparation.csv', [{key: value for key, value in row.items() if key != 'image_order'} for row in snapshots])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    if profiles:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([row['profile'] for row in profiles], [row['timings']['total_episode_wall_time'] for row in profiles], color='#2563eb')
        for i, row in enumerate(profiles):
            ax.text(i, row['timings']['total_episode_wall_time'], f"{row['timings']['total_episode_wall_time']:.1f} s", ha='center', va='bottom')
        ax.set(ylabel='Measured wall time (seconds)', xlabel='Baseline debug/replay → complete optimization',
               title='Same six-command tomato grasp/lift replay')
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            fig.savefig(output/f'native_trajectory.{suffix}', dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, label, title in zip(axes, ('arm_1cm', 'arm_8cm'), ('Completed 1 cm arm translation', 'Completed 8 cm arm translation')):
            selected = [row for row in actions if row['label'] == label and row['completed']]
            ax.bar([row['profile'] for row in selected], [row['wall_time_s'] for row in selected], color='#059669')
            ax.set(title=title, ylabel='Wall time (seconds)')
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            fig.savefig(output/f'arm_motion.{suffix}', dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, metric, title in zip(axes,
                ('controller_step_wall_time', 'render_wall_time', 'sensor_read_wall_time'),
                ('Native env.step (inclusive)', 'Additional explicit rendering', 'Sensor acquisition (inclusive)')):
            ax.bar([row['profile'] for row in profiles],
                   [row['timings'].get(metric) or 0 for row in profiles], color='#0891b2')
            ax.set(title=title, ylabel='Wall time (seconds)')
        fig.suptitle('Same six commands: overlapping categories; do not sum')
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            fig.savefig(output/f'native_categories.{suffix}', dpi=180)
        plt.close(fig)
    if agents:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, title, values in (
                (axes[0], 'Total episode elapsed', [row['timings']['total_episode_wall_time'] for row in agents]),
                (axes[1], 'Action RPC elapsed (inclusive)', [row['action_execution_wall_time'] for row in agents]),
                (axes[2], 'Model + tools (includes startup)', [row['timings']['model_and_tool_wall_time'] for row in agents])):
            ax.bar([row['phase'] for row in agents], [value or 0 for value in values], color='#d97706')
            ax.set(title=title, ylabel='Wall time (seconds)')
        fig.suptitle('One Luna episode per phase; different decision trajectories')
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            fig.savefig(output/f'agent_episode.{suffix}', dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        for ax, metric, title in zip(axes,
                ('model_input_preparation_time', 'model_application_startup_time',
                 'depth_tool_initialization_time', 'depth_tool_wall_time'),
                ('Input workspace preparation', 'Codex application startup',
                 'Depth MCP initialization', 'Depth tool round trips')):
            ax.bar([row['phase'] for row in agents],
                   [(row['timings'].get(metric) or 0)/row['model_calls'] for row in agents], color='#d97706')
            ax.set(title=title, ylabel='Seconds / model decision')
        fig.suptitle('One Luna episode per phase; decisions and trajectories can differ')
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            fig.savefig(output/f'agent_preparation_tools.{suffix}', dpi=180)
        plt.close(fig)
    if snapshots:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([row['mode'] for row in snapshots], [row['preparation_wall_time_s'] for row in snapshots], color='#7c3aed')
        ax.set(title='Identical saved snapshot: logging + model preparation', ylabel='Wall time (seconds)')
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            fig.savefig(output/f'snapshot_preparation.{suffix}', dpi=180)
        plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
