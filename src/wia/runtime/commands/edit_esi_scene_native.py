#!/usr/bin/env python3
"""Open the native OmniGibson authoring GUI and export separate scene snapshots.

Run in the behavior environment with DISPLAY and CUDA_VISIBLE_DEVICES set:
  python -B src/wia/runtime/commands/edit_esi_scene_native.py --scene-file SCENE --task-file TASK \
    --output-dir OUTPUT [--variant adapted] [--source-only]
Root may send OUTPUT/editor_request.json with request_id and action save, quit,
or load (scene_file, task_file, variant, source_only, optional scene_id). Scene
switches autosave before discarding the old world and retain the native app.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.runtime.scene_capture_queue import read_json, write_json
from wia.tasks.loader import LoadedScene, LoadedTask, UniqueLoader, sha256, source_scene_room, task_definition
from wia.tasks.schema import SceneDefinition
from wia.tasks.scene import ScenePatches
from wia.runtime.commands.check_esi_core_native import select_native_variant
import yaml


@dataclass
class EditorInputs:
    scene: LoadedScene
    task: LoadedTask
    source_only: bool = False

    def provenance(self):
        return {'scene_file':str(self.scene.path),'scene_sha256':self.scene.file_sha256,
            'task_file':str(self.task.path),'task_sha256':self.task.file_sha256,
            'source_file':str(self.task.source_path),'source_sha256':self.task.source_sha256,
            'source_identity':self.task.definition.source.json_path,
            'variant':self.scene.variant,'seed':self.scene.seed,
            'source_only':self.source_only,'adapted_patches_omitted':self.source_only,
            'authoring_only':True,'qualification':False,'model_calls':0}

    def verify_unchanged(self):
        self.task.verify_unchanged()
        self.scene.verify_unchanged()


def load_editor_inputs(scene_file, task_file, *, variant=None, source_only=False, source_root=None):
    """Validate actual input schemas and shared initialization without fallback."""
    scene_file,task_file=Path(scene_file).expanduser().resolve(),Path(task_file).expanduser().resolve()
    scene_def=SceneDefinition.model_validate(yaml.load(scene_file.read_text(),Loader=UniqueLoader))
    task_def=task_definition(yaml.load(task_file.read_text(),Loader=UniqueLoader))
    if task_def.source.json_path != scene_def.source.json_path:
        raise ValueError('Native editor task and scene must declare the same exact source identity.')
    if not hasattr(task_def,'scene'):
        raise ValueError('Native editor requires explicit task/scene initialization variants; no parent remapping is inferred.')
    selected=select_native_variant(scene_def,task_def,variant)
    if (task_def.defaults.seed != scene_def.seed or task_def.scene.variants[selected].model_dump(mode='json')
            != scene_def.scene.variants[selected].model_dump(mode='json')):
        raise ValueError('Native editor task/scene selected initialization and declared seeds must match exactly.')
    from wia.tasks.sources import resolve_source
    source=resolve_source(scene_def.source.json_path, data_root=source_root)
    scene_name,room_name=source_scene_room(scene_def.source.json_path,read_json(source))
    loaded=LoadedScene(scene_def,scene_file,source,sha256(scene_file),sha256(source),
        selected,scene_def.seed,scene_name,room_name)
    task=LoadedTask(task_def,task_file,source,sha256(task_file),sha256(source),
        selected,scene_def.seed,scene_name,room_name,run_mode='diagnostic')
    return EditorInputs(loaded,task,bool(source_only))


def validate_output_directory(output, inputs):
    """Never write editor artifacts inside a sealed capture input directory."""
    output=Path(output).expanduser().resolve()
    for filename in [output/'editor_status.json',*inputs]:
        for parent in Path(filename).expanduser().resolve().parents:
            if parent.name=='input' and (parent.parent/'accepted.json').is_file() and output.is_relative_to(parent):
                raise ValueError('Authoring output cannot be inside frozen queue inputs.')
    return output


def configure_native_gui():
    """Set GUI macros before importing the ESI pipeline or constructing an app."""
    os.environ['OMNIGIBSON_HEADLESS']='0'
    from wia.runtime.commands.run_esi_capture_worker import allocation_gpu
    from wia.runtime.scene_capture_adapters import install_gpu_selection, isolate_texture_cache
    gpu=allocation_gpu(0,execution_mode='local',renderer_order='pci')
    install_gpu_selection(gpu)
    from omnigibson import simulator
    isolate_texture_cache(simulator,os.environ['OMNIGIBSON_APPDATA_PATH'])
    from omnigibson.macros import gm
    gm.HEADLESS=False
    gm.GUI_VIEWPORT_ONLY=False
    gm.ENABLE_FLATCACHE=False
    return gpu


def create_editor_session(inputs):
    from wia.runtime.sim.esi_omnigibson import ESISession
    adapter=None if inputs.source_only else ScenePatches(inputs.scene,inputs.scene.definition.bindings)
    settings=adapter.observation_settings() if adapter is not None else {}
    settings['camera_size']=list(inputs.scene.definition.runtime.camera_size)
    session=ESISession(inputs.scene.source_path,require_robot=not inputs.source_only,grasp_mode='assisted',**settings)
    if adapter is not None:
        session.scene_adapter=adapter
    session.loaded_task=inputs.task
    session.public_task_text=inputs.task.task_text
    return session


class NativeSceneEditor:
    """All native operations and request handling run on the owner main thread."""
    def __init__(self, inputs, session, output):
        self.inputs,self.session,self.output=inputs,session,Path(output).resolve()
        self.close_requested=False
        self.request_ids=set()
        self.window=self.status_label=None
        self.scene_generation=1
        self.scene_id=None
        self.application=None
        self.native_og=getattr(getattr(session,'pipeline',None),'og',None)
        self.world_available=False
        self.safe_empty_world=False
        self.runtime_reusable=True
        self.last_load_context={}

    def runtime_context(self):
        og=getattr(getattr(self.session,'pipeline',None),'og',None) or self.native_og
        return {'scene_generation':self.scene_generation,'pid':os.getpid(),'scene_id':self.scene_id,
            'app_identity':id(og.app) if og is not None and og.app is not None else None,
            'native_simulator_identity':id(og.sim) if og is not None and og.sim is not None else None,
            'world_available':self.world_available,
            'recovery_load_allowed':self.runtime_reusable and (self.world_available or self.safe_empty_world)}

    def status(self, stage, **extra):
        record={'schema':'native_scene_editor_status_v1','stage':stage,'pid':os.getpid(),
            'output_dir':str(self.output),'gpu':getattr(self.session,'authoring_gpu_evidence',None),
            **self.inputs.provenance(),**self.runtime_context(),**extra}
        write_json(self.output/'editor_status.json',record)
        if self.status_label is not None:
            self.status_label.text=stage + (': ' + str(extra['error']) if extra.get('error') else '')
        return record

    def save_snapshot(self):
        if not self.world_available:
            raise ValueError('No successfully loaded authoring world is available to save.')
        sim=self.session.pipeline.og.sim
        if sim.is_stopped():
            raise ValueError('Cannot save a stopped simulation. Use the native Play/Pause controls first; no physics step was issued.')
        self.inputs.verify_unchanged()
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        directory=self.output/'exports'/('scene_'+stamp+'_'+uuid.uuid4().hex[:8])
        directory.mkdir(parents=True,exist_ok=False)
        scene=directory/'scene.json'
        sim.save(json_paths=[str(scene)])
        if not scene.is_file():
            raise RuntimeError('Native simulator did not create the requested scene JSON; no automatic stepping was attempted.')
        self.inputs.verify_unchanged()
        (directory/'task_input.yaml').write_bytes(self.inputs.task.path.read_bytes())
        (directory/'scene_input.yaml').write_bytes(self.inputs.scene.path.read_bytes())
        manifest={'schema':'native_scene_authoring_export_v1',**self.inputs.provenance(),**self.runtime_context(),
            'save_timestamp_utc':datetime.now(timezone.utc).isoformat(),'export_version':directory.name,
            'saved_scene_file':str(scene),'saved_scene_sha256':sha256(scene),
            'task_input_copy':str(directory/'task_input.yaml'),'scene_input_copy':str(directory/'scene_input.yaml'),
            'benchmark_integration':'separate_review_required','physics_state':'playing' if sim.is_playing() else 'paused'}
        write_json(directory/'manifest.json',manifest)
        self.status('exported',export_manifest=str(directory/'manifest.json'),saved_scene_file=str(scene))
        return manifest

    def save_from_ui(self):
        try:
            self.save_snapshot()
        except Exception as error:
            self.status('error',error=f'{type(error).__name__}: {error}')

    def create_window(self):
        import omni.ui as ui
        if self.window is not None:
            self.window.destroy()
        self.status_label=None
        self.window=ui.Window('Core Scene Authoring',width=490,height=260)
        with self.window.frame:
            with ui.VStack(spacing=8):
                ui.Label('Task: '+self.inputs.task.definition.id,word_wrap=True)
                ui.Label('Source: '+self.inputs.task.definition.source.json_path,word_wrap=True)
                ui.Label('Output: '+str(self.output),word_wrap=True)
                ui.Label('Authoring only. Exports do not modify task definitions or certify qualification.',word_wrap=True)
                ui.Button('Save snapshot',clicked_fn=self.save_from_ui)
                self.status_label=ui.Label('ready',word_wrap=True)

    def discard_session_world(self, session):
        """Reuse the resident capture lifecycle, keeping the native app alive."""
        from wia.runtime.commands.run_esi_capture_service import discard_whole_stage, reset_discarded_sensor_graphs
        from omni.syntheticdata import SyntheticData
        pipeline=session.pipeline
        og=pipeline.og
        application,simulator=og.app,og.sim
        reset_discarded_sensor_graphs(simulator,SyntheticData.Get())
        with discard_whole_stage(og):
            if session.task_module is not None and session.env is not None:
                pipeline.task_cleanup_runtime(session.task_module,session.env,session.payload,session.task_state)
            if og.sim is simulator:
                og.clear()
        if og.app is not application or og.sim is None or og.sim.scenes:
            raise RuntimeError('Authoring world discard did not preserve an empty live native application.')
        session.closed=True

    def initialize_current_scene(self):
        """Apply this input's seed/policy to a fresh session on the retained app."""
        import random
        import numpy as np
        import torch
        random.seed(self.inputs.scene.seed)
        np.random.seed(self.inputs.scene.seed)
        torch.manual_seed(self.inputs.scene.seed)
        self.status('loading')
        self.session.reset()
        og=self.session.pipeline.og
        if self.application is not None and og.app is not self.application:
            self.runtime_reusable=False
            raise RuntimeError('Scene loading replaced the persistent native application.')
        self.application=og.app
        self.native_og=og
        if getattr(self.session,'authoring_gpu',None) is not None:
            from wia.runtime.scene_capture_adapters import gpu_evidence
            self.session.authoring_gpu_evidence=gpu_evidence(self.session,self.session.authoring_gpu)
        if getattr(self.session,'scene_camera_position',None) is not None:
            og.sim.viewer_camera.set_position_orientation(
                position=torch.as_tensor(self.session.scene_camera_position,dtype=torch.float32),
                orientation=torch.as_tensor(self.session.scene_camera_quaternion,dtype=torch.float32))
        og.sim.pause()
        self.world_available=True
        self.safe_empty_world=False
        self.create_window()
        self.status('ready',camera_framing='existing_source_reference_view',
            gui_viewport_only=False,enable_flatcache=False,physics_state='paused')

    def load_scene(self, request):
        """Validate, autosave, then replace the world without closing the app."""
        self.last_load_context={'previous_world_retained':self.world_available,'scene_cleared':False}
        if not self.runtime_reusable:
            raise RuntimeError('Native runtime is not safely reusable; no scene switch was attempted.')
        if not isinstance(request.get('source_only',False),bool):
            raise ValueError('Load source_only must be an explicit boolean.')
        new_inputs=load_editor_inputs(request['scene_file'],request['task_file'],
            variant=request.get('variant'),source_only=request.get('source_only',False))
        validate_output_directory(self.output,[new_inputs.scene.path,new_inputs.task.path])
        if not self.world_available and not self.safe_empty_world:
            raise RuntimeError('Current native world is neither ready nor verified empty; no scene switch is safe.')
        autosave=self.save_snapshot() if self.world_available else None
        previous=self.session
        gpu=getattr(previous,'authoring_gpu',None)
        if self.world_available:
            self.world_available=False
            self.runtime_reusable=False
            self.last_load_context={'previous_world_retained':False,'scene_cleared':False}
            self.discard_session_world(previous)
            self.safe_empty_world=True
            self.runtime_reusable=True
        else:
            previous.closed=True
        self.last_load_context={'previous_world_retained':False,'scene_cleared':True}
        self.inputs=new_inputs
        self.scene_id=request.get('scene_id')
        self.scene_generation+=1
        try:
            self.session=create_editor_session(new_inputs)
            self.session.authoring_gpu=gpu
            self.initialize_current_scene()
        except Exception:
            self.world_available=False
            self.safe_empty_world=False
            try:
                failed=self.session
                if failed is previous or getattr(failed,'pipeline',None) is None:
                    og=previous.pipeline.og
                    if og.app is not self.application or og.sim is None or og.sim.scenes:
                        raise RuntimeError('Failed session creation left an unsafe native world.')
                    failed.pipeline=previous.pipeline
                    failed.closed=False
                else:
                    self.discard_session_world(failed)
                    # Keep normal ESISession.close available for final app shutdown,
                    # without invoking source cleanup on already discarded actors.
                    failed.closed=False
                failed.env=None
                failed.task_module=None
                self.safe_empty_world=True
            except Exception:
                self.runtime_reusable=False
            raise
        return {'autosave':autosave,**self.inputs.provenance(),**self.runtime_context(),**self.last_load_context}

    def process_request(self):
        path=self.output/'editor_request.json'
        if not path.is_file():
            return
        request=read_json(path,{})
        request_id=request.get('request_id')
        if not isinstance(request_id,str) or not request_id.strip():
            raise ValueError('Editor requests require a nonempty string request_id.')
        if request_id in self.request_ids:
            return
        self.request_ids.add(request_id)
        response={'request_id':request_id,'action':request.get('action'),'authoring_only':True}
        try:
            if request.get('action')=='save':
                response.update(complete=True,export=self.save_snapshot())
            elif request.get('action')=='load':
                response.update(complete=True,**self.load_scene(request))
            elif request.get('action')=='quit':
                self.close_requested=True
                response.update(complete=True)
            else:
                raise ValueError('Editor action must be save, load or quit.')
        except Exception as error:
            response.update(complete=False,error=f'{type(error).__name__}: {error}',
                **self.inputs.provenance(),**self.runtime_context(),
                **(self.last_load_context if request.get('action')=='load' else {}))
            self.status('error',error=response['error'],
                **(self.last_load_context if request.get('action')=='load' else {}))
        write_json(self.output/'editor_response.json',response)

    def pump(self):
        og=self.session.pipeline.og
        if og.sim.is_playing():
            og.sim.step()
        else:
            og.app.update()

    def run(self):
        self.initialize_current_scene()
        og=self.session.pipeline.og
        while not self.close_requested and og.app.is_running():
            began=time.monotonic()
            self.process_request()
            if self.close_requested:
                break
            self.pump()
            time.sleep(max(0.,1/30-(time.monotonic()-began)))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-file',type=Path,required=True)
    parser.add_argument('--task-file',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--variant',help='Shared declared variant; adapted when available, otherwise original.')
    parser.add_argument('--source-only',action='store_true',help='Explicit raw source authoring: omit adapted patches, with no fallback.')
    args=parser.parse_args(argv)
    output=validate_output_directory(args.output_dir,[args.scene_file,args.task_file])
    output.mkdir(parents=True,exist_ok=True)
    base={'schema':'native_scene_editor_status_v1','stage':'startup','pid':os.getpid(),
        'scene_file':str(args.scene_file.resolve()),'task_file':str(args.task_file.resolve()),
        'output_dir':str(output),'authoring_only':True,'qualification':False,'model_calls':0,
        'scene_generation':0,'app_identity':None,'native_simulator_identity':None,
        'world_available':False,'recovery_load_allowed':False}
    write_json(output/'editor_status.json',base)
    editor=session=inputs=None
    previous_handlers={}
    try:
        inputs=load_editor_inputs(args.scene_file,args.task_file,variant=args.variant,source_only=args.source_only)
        os.environ.setdefault('OMNIGIBSON_APPDATA_PATH',str(output/'appdata'))
        gpu=configure_native_gui()
        session=create_editor_session(inputs)
        session.authoring_gpu=gpu
        editor=NativeSceneEditor(inputs,session,output)
        def request_close(signum, frame):
            editor.close_requested=True
        for signum in (signal.SIGTERM,signal.SIGINT):
            previous_handlers[signum]=signal.signal(signum,request_close)
        editor.run()
        editor.status('closed')
        return 0
    except Exception as error:
        current_inputs=editor.inputs if editor is not None else inputs
        write_json(output/'editor_status.json',{**base,**(current_inputs.provenance() if current_inputs else {}),
            **(editor.runtime_context() if editor is not None else {}),'stage':'error',
            'error':f'{type(error).__name__}: {error}','traceback':traceback.format_exc()})
        return 1
    finally:
        if editor is not None:
            editor.session.close()
        elif session is not None:
            session.close()
        for signum,handler in previous_handlers.items():
            signal.signal(signum,handler)


if __name__=='__main__':
    raise SystemExit(main())
