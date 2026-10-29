"""Exercise native review camera integration with render-only simulator stubs.

Run ``python -m pytest -q tests/test_review_camera_native_adapter.py
tests/test_review_camera_geometry.py`` in the behavior environment.
"""
from copy import deepcopy
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.runtime.sim.esi_review_cameras import capture_review_views, review_view_names


class Actor:
    """Native-shaped actor whose pose must never be authored by review capture."""

    def __init__(self, name, category, low, high):
        self.name, self.category = name, category
        self.prim_path = '/World/' + name
        self.aabb = (np.asarray(low, float), np.asarray(high, float))
        self.visible = True
        self.position = np.array([41., 42., 43.])  # Deliberately differs from AABB center.
        self.orientation = np.array([0., 0., 0., 1.])

    def get_position_orientation(self):
        return self.position.copy(), self.orientation.copy()

    def set_position_orientation(self, **kwargs):
        raise AssertionError('Human review must not move actors.')


@pytest.fixture
def native_review(monkeypatch):
    apple = Actor('apple', 'fruit', [1., 2., 3.], [2., 4., 5.])
    box = Actor('box', 'box', [-3., -1., .3], [-1., 1., 2.])
    floor = Actor('floor', 'floors', [-100., -100., -.2], [100., 100., 0.])
    robot = Actor('robot', 'robot', [-1., -2., 0.], [0., -1., 1.])
    robot.sensors = {f'camera_{i}': NS(modalities={'rgb'}, clipping_range=(.02, 30.),
                                     pose=(i, i + 1, i + 2)) for i in range(8)}
    actors = {obj.name: obj for obj in (apple, box, floor, robot)}
    segmentation = np.zeros((16, 16), dtype=int)
    segmentation[:8] = 1
    segmentation[8:12] = 2
    scene = NS(segmentation=segmentation, labels={1: apple.prim_path + '/base_link', 2: box.name})
    scene.object_registry = lambda key, name: actors.get(name)
    calls = NS(render=0, invalidate=0, ray=0, created=0, raw_reads=[], conversions=[])

    class RawAnnotator:
        def __init__(self, sensor, modality):
            self.sensor, self.modality = sensor, modality

        def get_data(self, device):
            calls.raw_reads.append((self.modality, device))
            scene = self.sensor.scene
            if self.modality == 'rgb':
                return np.repeat(scene.segmentation[..., None], 3, axis=2).astype(np.uint8)
            return {'data': scene.segmentation.astype(np.uint32),
                    'info': {'idToLabels': scene.labels.copy()}}

    class HumanSensor:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            assert set(self.modalities) == {'rgb'}
            self.modalities = set(self.modalities)
            self.horizontal_aperture = 24.
            calls.created += 1

        def load(self, scene):
            self.scene = scene

        def initialize(self):
            self._annotators = {'rgb': RawAnnotator(self, 'rgb')}

        def add_modality(self, modality):
            self.modalities.add(modality)
            self._annotators[modality] = RawAnnotator(self, modality)

        def _preprocess_cpu_obs(self, pixels, modality):
            calls.conversions.append(('cpu', modality))
            return pixels.astype(np.int64) if 'seg_' in modality else pixels

        def _preprocess_gpu_obs(self, pixels, modality):
            calls.conversions.append(('cuda', modality))
            return pixels.view(np.int32) if 'seg_' in modality else pixels

        def set_position_orientation(self, position, orientation, frame):
            assert frame == 'world'
            self.position = np.asarray(position).copy()
            self.orientation = np.asarray(orientation).copy()

        def get_position_orientation(self):
            return self.position.copy(), self.orientation.copy()

        def get_obs(self):
            raise AssertionError('Class table does not exist in the semantic class name to id mapping!')

    def raytest(start, end, only_closest=True):
        calls.ray += 1
        hit = {'hit': False}
        return hit if only_closest else [hit]

    def render():
        calls.render += 1

    def step(*args, **kwargs):
        raise AssertionError('Capture must never advance physics.')

    def invalidate():
        calls.invalidate += 1

    monkeypatch.setitem(sys.modules, 'omnigibson.sensors', NS(VisionSensor=HumanSensor))
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.sampling_utils', NS(raytest=raytest))
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.usd_utils', NS(PoseAPI=NS(invalidate=invalidate)))
    monkeypatch.setitem(sys.modules, 'wia.tasks.support_sampling', NS(
        same_body=lambda path, body: body == path or body.startswith(path + '/')))
    session = NS(env=NS(scene=scene), robot=robot, task_object_names=['apple', 'box', 'floor'],
                 pipeline=NS(og=NS(sim=NS(render=render, step=step, device='cpu'))),
                 _rgb_from_data=lambda data, name: data['rgb'])
    definition = NS(bindings={name: NS(name=name, category=obj.category) for name, obj in actors.items()
                              if name != 'robot'}, source=NS(
        json_path='dataset/Physical Structure/Rigid Containment/house/kitchen/task.json'))
    return NS(session=session, selected=NS(definition=definition), actors=actors, calls=calls, scene=scene)


def test_native_bounds_pixels_and_render_only_capture_preserve_actors_and_robot_sensors(native_review):
    fixture = native_review
    sensors_before = deepcopy(vars(fixture.session.robot)['sensors'])
    poses_before = {name: obj.get_position_orientation() for name, obj in fixture.actors.items()}
    frames = capture_review_views(fixture.session, fixture.selected,
                                  ['object_overview', 'object_top', 'object_side', 'robot_head'])
    assert set(frames) == {'object_overview', 'object_top', 'object_side'}
    report = fixture.session.human_review_camera_evidence
    assert report['profile']['known_profile']
    assert report['native_objects'] == ['apple', 'box']
    assert report['physics_steps'] == report['actor_mutations'] == 0
    assert not report['robot_sensors_modified']
    assert fixture.calls.render == 27 and fixture.calls.invalidate == 3
    assert fixture.calls.ray > 0
    assert set(fixture.calls.raw_reads) == {('rgb', 'cpu'), ('seg_instance_id', 'cpu')}
    assert set(fixture.calls.conversions) == {('cpu', 'rgb'), ('cpu', 'seg_instance_id')}
    assert fixture.session._category_review_sensor.modalities == {'rgb', 'seg_instance_id'}
    for record in report['views'].values():
        assert record['bounds_world_m'] == [[-3., -1., .3], [2., 4., 5.]]
        assert np.allclose(record['target_world_m'], [-.5, 1.5, 2.65])
        assert record['object_pixels'] == {'apple': 128, 'box': 64}
        assert record['position_error_m'] == 0.
        assert not record['visibility_certifies_task_success']
    for name, obj in fixture.actors.items():
        for before, after in zip(poses_before[name], obj.get_position_orientation()):
            assert np.array_equal(before, after)
    assert set(fixture.session.robot.sensors) == set(sensors_before)
    for name, sensor in fixture.session.robot.sensors.items():
        assert vars(sensor) == vars(sensors_before[name])


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_raw_instance_paths_bypass_unknown_semantic_classes_and_keep_uint32_ids(native_review, device):
    fixture = native_review
    fixture.session.pipeline.og.sim.device = device
    fixture.actors['apple'].category = 'table'
    # A real mesh instance ID can occupy the unsigned half of the uint32 range.
    # The fake GPU conversion reproduces VisionSensor's signed int32 view.
    instance_id = (1 << 31) + 17
    fixture.scene.segmentation[:8] = instance_id
    fixture.scene.labels = {str(instance_id): '/World/apple/base_link/visual_mesh',
                            '2': '/World/box/base_link/visual_mesh'}
    capture_review_views(fixture.session, fixture.selected, ['object_top'])
    record = fixture.session.human_review_camera_evidence['views']['object_top']
    assert record['object_pixels'] == {'apple': 128, 'box': 64}
    assert set(fixture.calls.raw_reads) == {('rgb', device), ('seg_instance_id', device)}
    assert set(fixture.calls.conversions) == {(device.split(':')[0], 'rgb'),
                                             (device.split(':')[0], 'seg_instance_id')}
    assert fixture.actors['apple'].category == 'table'


def test_second_phase_reuses_exact_camera_positions_when_scene_content_changes(native_review):
    fixture = native_review
    names = ['global_front', 'object_top', 'object_detail_00']
    phase_one = capture_review_views(fixture.session, fixture.selected, names)
    fixed = deepcopy(fixture.session.human_review_camera_evidence['views'])
    old_fixed = deepcopy(fixed)
    old_ray_calls = fixture.calls.ray
    fixture.actors['apple'].aabb = (np.array([9., 10., 11.]), np.array([10., 12., 13.]))
    fixture.scene.segmentation[:] = 2
    phase_two = capture_review_views(fixture.session, fixture.selected, names, fixed_views=fixed)
    assert fixture.calls.created == 1
    assert fixture.calls.ray == old_ray_calls
    assert fixed == old_fixed
    for name, record in fixture.session.human_review_camera_evidence['views'].items():
        assert record['position_world_m'] == fixed[name]['position_world_m']
        assert record['actual_position_world_m'] == fixed[name]['actual_position_world_m']
        assert record['actual_quaternion_xyzw'] == fixed[name]['actual_quaternion_xyzw']
        assert record['target_world_m'] == fixed[name]['target_world_m']
        assert record['object_pixels'] == {'apple': 0, 'box': 256}
        assert 'apple' in record['objects_below_64_pixels']
        assert not np.array_equal(phase_one[name], phase_two[name])


def test_labeled_global_views_keep_distinct_bearings_when_one_opening_scores_best(native_review, monkeypatch):
    fixture = native_review
    monkeypatch.setattr('wia.runtime.sim.esi_review_cameras._ray_score',
        lambda position, objects: float(position[0] > 0))
    names = ['global_front', 'global_back', 'global_left', 'global_right']
    capture_review_views(fixture.session, fixture.selected, names)
    evidence = fixture.session.human_review_camera_evidence['views']
    expected = {'global_front': [1., 0.], 'global_back': [-1., 0.],
                'global_left': [0., 1.], 'global_right': [0., -1.]}
    for name, direction in expected.items():
        row = evidence[name]
        vector = np.asarray(row['position_world_m'])[:2] - np.asarray(row['target_world_m'])[:2]
        np.testing.assert_allclose(vector / np.linalg.norm(vector), direction, atol=1e-10)
        assert row['azimuth_preserved']


def test_missing_targets_and_zero_pixels_never_become_fake_visibility(native_review):
    fixture = native_review
    fixture.scene.segmentation[:] = 0
    capture_review_views(fixture.session, fixture.selected, ['object_overview'])
    record = fixture.session.human_review_camera_evidence['views']['object_overview']
    assert record['object_pixels'] == {'apple': 0, 'box': 0}
    assert set(record['objects_below_64_pixels']) == {'apple', 'box'}
    fixture.actors.pop('apple')
    fixture.actors.pop('box')
    capture_review_views(fixture.session, fixture.selected, ['object_overview'])
    report = fixture.session.human_review_camera_evidence
    assert report['missing_task_objects']
    assert report['native_objects'] == []
    assert report['views']['object_overview']['object_pixels'] == {}
    assert report['views']['object_overview']['intended_objects'] == []


@pytest.mark.parametrize('failure', ['missing', 'invalid_aabb', 'unreadable_aabb'])
def test_partial_target_geometry_failures_are_named_without_hiding_valid_targets(native_review, failure):
    fixture = native_review
    if failure == 'missing':
        fixture.actors.pop('apple')
    elif failure == 'invalid_aabb':
        fixture.actors['apple'].aabb = (np.array([np.nan, 0., 0.]), np.ones(3))
    else:
        class UnreadableActor:
            name, category, visible = 'apple', 'fruit', True

            @property
            def aabb(self):
                raise RuntimeError('Native AABB is unavailable.')

        fixture.actors['apple'] = UnreadableActor()
    capture_review_views(fixture.session, fixture.selected, ['object_overview'])
    report = fixture.session.human_review_camera_evidence
    assert report['native_objects'] == ['box']
    assert not report['missing_task_objects']
    assert any(issue['name'] == 'apple' and issue['reason'] for issue in report['object_issues'])
    assert report['views']['object_overview']['object_pixels'] == {'box': 64}


def test_all_missing_without_robot_cannot_invent_framing(native_review):
    fixture = native_review
    fixture.actors.pop('apple')
    fixture.actors.pop('box')
    fixture.session.robot = None
    with pytest.raises(RuntimeError, match='no native object or robot bounds'):
        capture_review_views(fixture.session, fixture.selected, ['object_overview'])


def test_manifest_excludes_floor_bindings_from_detail_count_and_bounds_large_lists():
    definition = NS(bindings={str(i): NS(category='floor') for i in range(30)})
    definition.bindings['target'] = NS(category='fruit')
    names = review_view_names(definition)
    assert names == ['object_overview', 'object_top', 'object_side', 'object_detail_00']
    definition.bindings.update({f'target_{i}': NS(category='box') for i in range(100)})
    names = review_view_names(definition)
    assert len(names) == 11 and len(set(names)) == 11
    assert names[-1] == 'object_detail_07'
