"""Private native stack terminal evidence; never supplied to the Agent.

Run through src/wia/runtime/commands/run_esi_robot.py --task-file a qualified category task.
Contact, release and consecutive physics-step stability are measured separately
from original ESI QA. No object pose is changed by this evaluator.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np


class UnavailableEvaluator:
    """An explicit diagnostic receipt for a missing native scorer, never a zero."""
    def __init__(self, config, stage, error):
        self.config, self.stage, self.error = config, stage, f'{type(error).__name__}: {error}'
        self.items = None

    def begin_episode(self, observation):
        return self.record(observation, action=None)

    def record(self, observation, *, action=None):
        from wia.tasks.completion import declared_items, progress
        value = {'schema': 'private_unavailable_evaluation_v1', 'task_success': False,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'evaluation_validity': 'unavailable', 'failure_stage': self.stage,
            'failure_reason': self.error, 'completion_items': self.items if self.items is not None else declared_items(self.config)}
        value.update(progress(value))
        return value


class CategoryEvaluationEnvelope:
    """Bind category-native results to the exact loaded episode and boundary."""

    def __init__(self, native, task, output):
        self.native, self.task, self.output = native, task, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode = 0
        self.failure = None
        self.last_native = None
        self.done_result = None
        self.done_error = None

    @property
    def requires_done_declaration(self):
        return getattr(self.native, 'requires_done_declaration', False) is True

    def begin_episode(self, observation):
        self.task.verify_unchanged()
        self.done_result = None
        self.done_error = None
        self.episode += 1
        try:
            native = self.native.begin_episode(observation)
        except Exception as error:
            self._fail('begin_episode', error)
            native = self.failure.record(observation)
        if native is None:
            try:
                # Read the already sampled boundary without sampling twice.
                state = self.output / 'state.json'
                native = json.loads(state.read_text()) if state.is_file() else self.native.record(observation, action=None)
            except Exception as error:
                self._fail('initial_result_read', error)
                native = self.failure.record(observation)
        return self._seal(native, observation)

    def sample_step(self):
        if self.done_result is not None:
            return
        sample = getattr(self.native, 'sample_step', None)
        if callable(sample) and self.failure is None:
            try:
                sample()
            except Exception as error:
                self._fail('control_sample', error)

    def record(self, observation, *, action):
        if self.done_result is not None:
            return deepcopy(self.done_result)
        if not self.episode:
            return None
        try:
            native = (self.failure or self.native).record(observation, action=action)
        except Exception as error:
            self._fail('observation_boundary', error)
            native = self.failure.record(observation)
        if native is None:
            try:
                native = json.loads((self.output / 'state.json').read_text())
            except Exception as error:
                self._fail('boundary_result_read', error)
                native = self.failure.record(observation)
        return self._seal(native, observation)

    def report_done(self, observation):
        """Latch one terminal native result without executing or stepping a robot."""
        if not self.requires_done_declaration:
            return None
        if self.done_result is not None:
            if self.done_error is not None:
                raise RuntimeError(self.done_error)
            return deepcopy(self.done_result)
        if not self.episode:
            raise RuntimeError('DONE declarations require an active episode.')
        self.task.verify_unchanged()
        try:
            native = self.native.declare_done(observation)
        except Exception as error:
            self._fail('done_declaration', error)
            native = self.failure.record(observation)
        self.done_result = self._seal(native, observation)
        if self.done_result.get('evaluation_finalized') is not True:
            self.done_error = 'Terminal task evaluation did not finalize.'
            raise RuntimeError(self.done_error)
        return deepcopy(self.done_result)

    def _fail(self, stage, error):
        self.failure = UnavailableEvaluator(self.task.definition.private.evaluation.model_dump(), stage, error)
        if self.last_native is not None:
            from wia.tasks.completion import progress
            self.failure.items = progress(self.last_native)['completion_items']
            if self.failure.items is not None:
                for item in self.failure.items:
                    item['evaluation_validity'] = 'unavailable'

    def report_arrivals(self, rooms, observation):
        """Record Agent declarations at this boundary without stepping physics."""
        if not self.episode:
            raise RuntimeError('Arrival declarations require an active episode.')
        self.task.verify_unchanged()
        report = getattr(self.native, 'report_arrival', None)
        if not callable(report):
            return
        for room in rooms:
            try:
                report(room, observation)
            except Exception as error:
                self._fail('arrival_report', error)
        self.record(observation, action=None)

    def prepare_temporal_transition(self, observation):
        """Seal the old phase's read-only native evidence before scene mutation."""
        if not self.episode:
            raise RuntimeError('Temporal transitions require an active episode.')
        self.task.verify_unchanged()
        prepare = getattr(self.native, 'prepare_phase_transition', None)
        if not callable(prepare):
            raise ValueError('Native temporal transition support is unavailable.')
        prepare(observation)
        self.record(observation, action=None)

    def report_temporal_answer(self, answer, observation):
        """Keep Agent answers and measured correctness in private native evidence."""
        if not self.episode:
            raise RuntimeError('Temporal answers require an active episode.')
        self.task.verify_unchanged()
        report = getattr(self.native, 'report_answer', None)
        if not callable(report):
            raise ValueError('Native temporal answer support is unavailable.')
        report(answer, observation)
        self.record(observation, action=None)

    def report_answer(self, answer, observation):
        """Score an explicit public answer without returning private correctness."""
        if not self.episode:
            raise RuntimeError('Answer declarations require an active episode.')
        self.task.verify_unchanged()
        report = getattr(self.native, 'report_answer', None)
        if not callable(report):
            raise ValueError('This task does not accept answer declarations.')
        try:
            report(answer, observation)
        except Exception as error:
            self._fail('answer_report', error)
        self.record(observation, action=None)

    def _seal(self, native, observation):
        invalid = None
        if not isinstance(native, dict) or type(native.get('task_success')) is not bool:
            invalid = 'Native category evaluation must report measured boolean task_success.'
        elif (native.get('snapshot_id') != observation.snapshot_id
                or native.get('action_step') != observation.action_step
                or native.get('schema') == 'private_category_task_v1'):
            invalid = 'Native category result does not describe the current observation boundary.'
        if invalid is not None:
            self._fail('result_validation', ValueError(invalid))
            native = self.failure.record(observation)
        session = getattr(self.native, 'session', None)
        private = getattr(self.task.definition, 'private', None)
        evaluation = getattr(private, 'evaluation', None)
        from wia.runtime.prechecks import enabled
        if session is not None and evaluation is not None and enabled(session):
            config = evaluation.model_dump()
            parameters = config.get('parameters', {})
            if parameters.get('destination_contract') or parameters.get('region_label_contract'):
                config['bindings'] = {key: value.name for key, value in private.bindings.items()}
                contracts = qualify_definition_contracts(session, config, stage='evaluation')
            else:
                contracts = {}
            if contracts:
                native = deepcopy(native)
                native['definition_qualification'] = contracts
                if not all(row['accepted'] for row in contracts.values()):
                    native['unqualified_diagnostic_score_percent'] = native.get('score_percent')
                    native['evaluation_validity'] = 'unavailable'
                    native['task_success'] = False
                    for item in native.get('completion_items') or []:
                        item['evaluation_validity'] = 'unavailable'
                        item['reason'] = 'public_definition_native_qualification_pending_or_failed'
                    for key in ('answer_accuracy_percent', 'discovery_percent', 'comparison_accuracy_percent',
                                'score_percent', 'completion_score'):
                        if key in native:
                            native[key] = None
        self.last_native = native
        value = {'schema': 'private_category_task_v1', 'episode': self.episode,
            'task_id': self.task.definition.id, 'task_text': self.task.task_text,
            'task_file_sha256': self.task.file_sha256, 'source_sha256': self.task.source_sha256,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'task_success': native['task_success'], 'native_schema': native.get('schema'),
            'native_result': native}
        from wia.tasks.completion import progress
        value['progress'] = progress(native)
        value['progress']['status'] = ('pending_done' if native.get('evaluation_validity') == 'pending_done'
                                       else 'measured')
        if self.requires_done_declaration:
            value['requires_done_declaration'] = True
            value['evaluation_finalized'] = native.get('evaluation_finalized', False) is True
            if native.get('evaluation_validity') == 'pending_done':
                # Intermediate physical evidence is not a numerical task score.
                value['progress'].update({key: None for key in (
                    'completion_items', 'completed_goals', 'total_goals', 'completion_score',
                    'score_percent', 'diagnostic_lower_bound')})
                value['progress']['evaluation_validity'] = 'pending_done'
        if getattr(self.task.definition, 'repair', {}).get('source_conflict'):
            disputed = dict(native)
            disputed['evaluation_validity'] = 'unavailable'
            disputed['completion_items'] = value['progress']['completion_items']
            for item in disputed['completion_items'] or []:
                item['evaluation_validity'] = 'unavailable'
                item['reason'] = 'source_label_dependent_obligation_unresolved'
            value['progress'] = progress(disputed)
            value['progress'].update(status='measured',
                reason='Source label-dependent selection is unresolved; physical actions remain diagnostic.')
        value.update({key: value['progress'][key] for key in (
            'evaluation_validity', 'scoring_version', 'completion_score', 'score_percent')})
        # A question set's primary semantic metric must not be overwritten by
        # the diagnostic fraction of mixed semantic and physical obligations.
        metrics = ('answer_accuracy_percent', 'required_unique_questions', 'correct_answers',
            'answer_evaluation_validity', 'diagnostic_answer_lower_bound_percent',
            'discovery_percent', 'declaration_precision_percent', 'false_positive_count',
            'required_distinct_targets', 'verified_distinct_targets',
            'comparison_accuracy_percent', 'exact_ranking_success',
            'physical_obligations', 'semantic_questions', 'per_subclass', 'per_group',
            'agent_response_errors', 'obligation_score_percent')
        for key in metrics:
            if key in native:
                value[key] = native[key]
        primary = next((key for key in ('answer_accuracy_percent', 'discovery_percent',
            'comparison_accuracy_percent') if key in native), None)
        if primary is not None:
            value['primary_metric'] = primary
            value['score_percent'] = (None if getattr(self.task.definition, 'repair', {}).get('source_conflict')
                                      else native[primary])
            value['progress']['obligation_score_percent'] = value['progress']['score_percent']
            value['progress']['score_percent'] = value['score_percent']
            value['progress']['primary_metric'] = primary
            for key in metrics:
                if key in value:
                    value['progress'][key] = value[key]
        value['run_mode'] = getattr(self.task, 'run_mode', 'benchmark')
        value['progress']['run_mode'] = value['run_mode']
        value['review_status'] = getattr(self.task.definition, 'status', 'pending')
        value['task_success'] = bool(native['task_success'] and value['evaluation_validity'] == 'complete')
        if self.requires_done_declaration and not value['evaluation_finalized']:
            value['task_success'] = False
        value['progress']['goals'] = native.get('goals', [])
        session = getattr(self.native, 'session', None)
        if session is not None:
            from wia.tasks.background_policy import record_background_movements
            value['background_movements'] = record_background_movements(session)
            safety = getattr(session, 'safety_observer', None)
            if safety is not None:
                from wia.runtime.sim.esi_safety import private_safety_result
                value['safety'] = private_safety_result(safety)
                value['safety_score'] = value['safety']['safety_score']

        from wia.authoring.io import write_json
        write_json(self.output / 'state.json', value)
        with (self.output / 'episode_boundaries.jsonl').open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def validate_episode_evaluation(data, observation, task_text, loaded_task=None):
    """Require current native evidence; category and legacy protocols stay distinct."""
    category = loaded_task is not None and loaded_task.definition.kind == 'category_robot_task'
    expected = 'private_category_task_v1' if category else 'private_grasp_lift_v1'
    valid = (data.get('schema') == expected and type(data.get('task_success')) is bool
             and data.get('task_text') == task_text
             and data.get('action_step') == observation.action_step)
    if loaded_task is not None:
        valid = valid and data.get('snapshot_id') == observation.snapshot_id and data.get('episode') == 1
    if category:
        valid = valid and all(data.get(field) == expected_value for field, expected_value in (
            ('task_id', loaded_task.definition.id), ('task_file_sha256', loaded_task.file_sha256),
            ('source_sha256', loaded_task.source_sha256)))
    if not valid:
        raise RuntimeError('Private evaluation does not match the current episode/action boundary.')
    if category and data.get('requires_done_declaration') is True and data.get('evaluation_finalized') is not True:
        return None
    return data['task_success']


def owned_category_module(module_name):
    """Resolve only a local category module owned by a conversion supervisor."""
    from wia.authoring.modules import import_owned_module
    return import_owned_module(module_name)


def category_operation_module(config):
    """Resolve trusted operation code; source answers never choose success."""
    if config.get('type') != 'category_operation':
        raise ValueError('Use an owned local category operation adapter.')
    return owned_category_module(config.get('module', ''))


def qualify_category_operation(session, controller, config, bound):
    """Adapters inspect native preconditions without changing the environment."""
    module = category_operation_module(config)
    qualifier = getattr(module, 'qualify_operation', None)
    if not callable(qualifier):
        raise ValueError('Category operation adapters must implement native qualification.')
    result = qualifier(session, controller, config, bound)
    if not isinstance(result, dict) or type(result.get('accepted')) is not bool:
        raise ValueError('Operation qualification requires an explicit measured acceptance result.')
    contracts = qualify_definition_contracts(session, config, bound, stage='initialization')
    if contracts:
        result = deepcopy(result)
        result['definition_qualification'] = contracts
        for name, receipt in contracts.items():
            result.setdefault('checks', {})['public_definition:' + name] = receipt['accepted']
        result['accepted'] = result['accepted'] and all(row['accepted'] for row in contracts.values())
    return result


def qualify_definition_contracts(session, config, bound=None, *, stage):
    """Keep structural preconditions separate from later episode qualification."""
    parameters = config.get('parameters', {})
    results = {}
    if parameters.get('destination_contract'):
        from wia.authoring.converters.task_edit2_destinations import validate_destination
        results['destination'] = validate_destination(session, config, bound, stage=stage)
    if parameters.get('region_label_contract'):
        from wia.authoring.converters.enumerative_perception_spatial_segmentation import validate_region_labels
        results['region_labels'] = validate_region_labels(session, config, bound, stage=stage)
    return results


def create_category_operation_evaluator(session, controller, config, output):
    """Reuse the normal episode/evidence hooks for category-owned operation goals."""
    factory = getattr(category_operation_module(config), 'create_evaluator', None)
    if not callable(factory):
        raise ValueError('Category operation adapters must implement create_evaluator.')
    evaluator = factory(session, controller, config, output)
    if not all(callable(getattr(evaluator, method, None)) for method in ('begin_episode', 'record')):
        raise ValueError('Operation evaluators must implement begin_episode and record.')
    if config.get('parameters', {}).get('arrival_report_required') and not callable(getattr(evaluator, 'report_arrival', None)):
        raise ValueError('Arrival-report tasks require native report_arrival support.')
    return evaluator


def stack_terminal_checks(relations, held_name, *, max_linear_speed=.02, max_angular_speed=.1):
    """Consume measured native relations, never source QA labels or screenshot scores."""
    return bool(relations) and all(
        r['on_top'] and r['touching'] and r['upright'] and
        held_name not in (r['upper'], r['lower']) and
        r['upper_linear_speed'] <= max_linear_speed and r['lower_linear_speed'] <= max_linear_speed and
        r['upper_angular_speed'] <= max_angular_speed and r['lower_angular_speed'] <= max_angular_speed
        for r in relations)


class StackRelationsEvaluator:
    def __init__(self, session, controller, config, output):
        if (config['type'] != 'stack_relations' or config['scene'] != session.scene_name
                or config['room'] != session.room_name):
            raise ValueError('Stack evaluation must match the native scene and room.')
        self.session, self.controller, self.config = session, controller, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.stable_steps, self.initial = 0, 0, None
        self.objects = {}
        for relation in config['relations']:
            for name in (relation['upper'], relation['lower']):
                obj = session.env.scene.object_registry('name', name)
                if obj is None or obj.visual_only or obj.fixed_base:
                    raise ValueError('Stack bindings require original movable collidable objects.')
                self.objects[name] = obj

    def measure(self):
        from omnigibson.object_states import OnTop, Touching
        from scipy.spatial.transform import Rotation
        from wia.native.utils import _array
        rows = []
        for relation in self.config['relations']:
            upper, lower = (self.objects[relation[name]] for name in ('upper', 'lower'))
            row = {**relation, 'on_top': bool(upper.states[OnTop].get_value(lower)),
                'touching': bool(upper.states[Touching].get_value(lower))}
            upright = []
            for label, obj in (('upper', upper), ('lower', lower)):
                _, quaternion = obj.get_position_orientation()
                upright.append(float(Rotation.from_quat(_array(quaternion)).as_matrix()[2, 2]) >= .9)
                row[label + '_linear_speed'] = float(np.linalg.norm(_array(obj.get_linear_velocity())))
                row[label + '_angular_speed'] = float(np.linalg.norm(_array(obj.get_angular_velocity())))
            row['upright'] = all(upright)
            rows.append(row)
        held = self.controller.debug_gripper_state().get('object_name')
        return rows, held, stack_terminal_checks(rows, held)

    def begin_episode(self, observation):
        self.episode += 1
        self.stable_steps = 0
        self.initial = self.measure()[0]
        self.record(observation, action=None)

    def sample_step(self):
        if self.initial is not None:
            valid = self.measure()[2]
            self.stable_steps = self.stable_steps + 1 if valid else 0

    def record(self, observation, *, action):
        if self.initial is None:
            return None
        rows, held, valid = self.measure()
        value = {'schema': 'private_stack_relations_v1', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'action': action, 'relations': rows, 'held_name': held, 'stable_steps': self.stable_steps,
            'required_stability_steps': self.config['stability_steps'],
            'task_success': bool(valid and self.stable_steps >= self.config['stability_steps']),
            'configuration': self.config, 'initial_relations': self.initial}
        from wia.authoring.io import write_json
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False) + '\n')
        return value


class ExtractLiftEvaluator:
    """Compose existing native grasp/lift evidence with measured container exit."""
    def __new__(cls, session, controller, config, output):
        from wia.native.esi_task_evaluation import GraspLiftEvaluator
        from omnigibson.object_states import Inside

        class NativeExtractLift(GraspLiftEvaluator):
            def additional_evidence(self, target):
                inside = bool(self.target.states[Inside].get_value(self.container))
                return {'outside_source_container': not inside}, {'inside_source_container': inside,
                    'source_container': self.container.name, 'predicate': 'Native Inside plus independent held/lift evidence.'}

        container = session.env.scene.object_registry('name', config['container'])
        if container is None or container.visual_only:
            raise ValueError('Extraction requires the original collidable container.')
        base = {k: config[k] for k in ('scene', 'room', 'target', 'minimum_lift_m')}
        evaluator = NativeExtractLift(session, controller, {**base, 'type': 'grasp_lift'}, output)
        evaluator.container = container
        evaluator.config = dict(config)
        return evaluator
