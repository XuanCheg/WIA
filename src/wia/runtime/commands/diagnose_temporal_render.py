"""Inspect native temporal teleports and rendered actor positions.

Run with behavior Python, CUDA_VISIBLE_DEVICES and OMNIGIBSON_APPDATA_PATH:
python -B scripts/diagnose_temporal_render.py --scene-file SCENE --output-dir DIR
This diagnostic includes an explicitly recorded physics step, never Agent credit.
"""
import argparse
from pathlib import Path
import os
import sys
import traceback

ROOT = __import__('wia.resources', fromlist=['ROOT']).ROOT
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    from wia.runtime.scene_capture_queue import read_json, write_json
    from wia.runtime.commands.run_esi_capture_worker import allocation_gpu
    from wia.runtime.scene_capture_adapters import install_gpu_selection, isolate_texture_cache
    gpu = allocation_gpu(0, execution_mode='local', renderer_order='pci')
    install_gpu_selection(gpu)
    from omnigibson import simulator
    isolate_texture_cache(simulator, os.environ['OMNIGIBSON_APPDATA_PATH'])
    import yaml
    from wia.tasks.loader import LoadedScene, sha256
    from wia.tasks.schema import SceneDefinition
    from wia.tasks.scene import ScenePatches
    from wia.runtime.sim.esi_omnigibson import ESISession
    from wia.runtime.sim.esi_review_cameras import capture_review_views
    from wia.tasks.interaction_setup import array
    from core.record.images import save_png
    from omnigibson.utils.usd_utils import PoseAPI

    path = args.scene_file.resolve()
    definition = SceneDefinition.model_validate(yaml.safe_load(path.read_text()))
    from wia.tasks.sources import resolve_source
    source = resolve_source(definition.source.json_path)
    raw = read_json(source)
    loaded = LoadedScene(definition, path, source, sha256(path), sha256(source),
        'adapted', definition.seed, raw['scene'], raw['room'])
    adapter = ScenePatches(loaded, definition.bindings)
    settings = adapter.observation_settings()
    settings['camera_size'] = list(definition.runtime.camera_size)
    session = ESISession(source, require_robot=True, grasp_mode='assisted', **settings)
    session.scene_adapter = adapter
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    report = {'gpu': gpu, 'robot_actions': 0, 'physics_steps_after_transition': 0}
    try:
        session.reset()
        controller = session.temporal_phase_controller
        def record(stage):
            actors = {}
            for key in controller.contents:
                obj = controller.objects[key]
                actors[key] = {'pose': [array(v).tolist() for v in obj.get_position_orientation()],
                    'aabb': [array(v).tolist() for v in obj.aabb], 'visible': obj.visible,
                    'expected': controller.parameters['source_objects'][key],
                    'links': {name: {'visible': link.visible,
                        'pose': [array(v).tolist() for v in link.get_position_orientation()]}
                        for name, link in obj.links.items()}}
            report[stage] = actors
            write_json(output / 'report.json', report)
            print(stage, flush=True)
        def render(stage):
            views = capture_review_views(session, loaded, ['object_top'])
            save_png(output / (stage + '.png'), views['object_top'])
            report[stage + '_camera'] = session.human_review_camera_evidence
            record(stage + '_after_render')
        record('phase_1')
        render('phase_1')
        controller.prepared_boundary = (0, 0)
        controller.advance()
        record('phase_2')
        render('phase_2')
        # Repeat the authored pose after visibility and native-property updates.
        from wia.native.esi_temporal_phases import _set_pose
        for key in controller.active.values():
            if key is not None:
                expected = controller.parameters['source_objects'][key]
                _set_pose(controller.objects[key], expected['position'], expected['quaternion_xyzw'])
        PoseAPI.invalidate()
        PoseAPI._refresh()
        record('pose_reapplied')
        render('pose_reapplied')
        session.pipeline.og.sim.step()
        report['physics_steps_after_transition'] = 1
        record('one_physics_step')
        render('one_physics_step')
        report['complete'] = True
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
    finally:
        write_json(output / 'report.json', report)
        session.close()


if __name__ == '__main__':
    main()
