"""One GUMI monitor; MCP consumes the separate allowlisted Agent projections."""
from __future__ import annotations

import threading
import base64

import cv2
import numpy as np

from core.observations import CameraSnapshot
from core.robot_cameras import ADDITIONAL_RGB_NAMES, RGB_TO_VIEW, ROBOT_CAMERA_LABELS
from core.viewer_config import ViewerConfig
from wia.runtime.web.web_teleop.esi_recording import HumanVideoRecorder


class ESIViewerBackend:
    # Human recording is enabled even when underlying robot controls are read-only.
    read_only = False
    dual_arm = False
    human_viewer = True

    def __init__(self, robot_backend, recording_dir, config: ViewerConfig):
        self.robot_backend = robot_backend
        self.config = config
        self.frame_cond = threading.Condition()
        self.frames = {}
        self.frame_seq = 0
        self.frame_time = None
        performance = robot_backend.metadata.get('performance', {})
        debug_cameras = performance.get('debug_cameras', True)
        depth_views = performance.get('human_depth_visualizations', True)
        self.recording_enabled = performance.get('human_recording', True)
        self.camera_catalog = [
            {'view': 'scene', 'label': 'ESI viewer camera', 'group': 'ESI'},
            *({'view': cam.name, 'label': cam.label, 'group': 'Global'} for cam in config.global_views),
            {'view': 'head', 'label': 'R1 head · RGB', 'group': 'Robot'},
            {'view': 'wrist', 'label': 'R1 wrist · RGB', 'group': 'Robot'},
            *({'view': RGB_TO_VIEW[name], 'label': ROBOT_CAMERA_LABELS[name], 'group': 'Robot'}
              for name in ADDITIONAL_RGB_NAMES
              if name in robot_backend.metadata.get('robot_camera_mounts', {})),
            {'view': 'head_depth', 'label': 'R1 head · depth', 'group': 'Robot'},
            {'view': 'wrist_depth', 'label': 'R1 wrist · depth', 'group': 'Robot'},
        ]
        self.camera_catalog = [item for item in self.camera_catalog
            if (debug_cameras or item['group'] == 'Robot')
            and (depth_views or not item['view'].endswith('_depth'))]
        session = getattr(getattr(robot_backend, 'observation_source', None), 'session', None)
        self.recorder = HumanVideoRecorder(recording_dir, config, metadata={'camera_catalog': self.camera_catalog},
                                          performance=getattr(session, 'performance', None))
        policy = getattr(robot_backend, 'world_policy', None)
        if policy is not None:
            policy.on_phase = lambda phase: self.recorder.set_model_wait(phase == 'awaiting_model')
            self.recorder.set_model_wait(policy.phase == 'awaiting_model')

    @property
    def ALLOWED_COMMANDS(self):
        return ('start', 'stop', 'cancel', *getattr(self.robot_backend, 'ALLOWED_COMMANDS', ()))

    @property
    def STREAM_VIEWS(self):
        return tuple(item['view'] for item in self.camera_catalog) + ('agentview',)

    @property
    def agent_backend(self):
        return self.robot_backend

    def agent_observation(self):
        return self.robot_backend.agent_observation()

    def camera_groups(self, cameras):
        return [{'id': key, 'label': label,
                 'views': [camera['view'] for camera in cameras if camera['group'] == group]}
                for key, label, group in (('esi', 'ESI', 'ESI'), ('global', 'Global', 'Global'),
                                          ('robot', 'Robot', 'Robot'))]

    def camera_group(self, group):
        """One short HTTP read contains all cached images from the same cycle."""
        with self.frame_cond:
            groups = self.camera_groups(self.camera_catalog)
            selected = next((item for item in groups if item['id'] == group), None)
            if selected is None:
                return None
            return {'group': group, 'frame_seq': self.frame_seq, 'frame_time': self.frame_time,
                    'images': {name: base64.b64encode(self.frames[name]).decode('ascii')
                               for name in selected['views'] if name in self.frames}}

    @property
    def STEP_TOKENS(self):
        return self.robot_backend.STEP_TOKENS

    @property
    def action_groups(self):
        return getattr(self.robot_backend, 'action_groups', {})

    @property
    def rotation_enabled(self):
        return getattr(self.robot_backend, 'rotation_enabled', False)

    def action_allowed(self, token):
        allowed = getattr(self.robot_backend, 'action_allowed', None)
        return bool(allowed and allowed(token))

    def publish(self, snapshot: CameraSnapshot):
        self._publish_frames(snapshot.human_views(), snapshot.observation.snapshot_id,
                             snapshot.observation.captured_at)

    def publish_preview(self, preview):
        self._publish_frames(preview.frames, f'preview_{preview.preview_id}', preview.captured_at)

    def _publish_frames(self, rgb, frame_id, captured_at):
        allowed = {item['view'] for item in self.camera_catalog}
        rgb = {name: frame for name, frame in rgb.items() if name in allowed}
        encoded = {}
        with self.robot_backend.frame_cond:
            cached = dict(self.robot_backend.frames) if self.robot_backend.frame_seq == frame_id else {}
        for name, frame in rgb.items():
            if name in cached:
                encoded[name] = cached[name]
                continue
            bgr = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
            ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError(f'Cannot encode human camera {name}.')
            encoded[name] = jpeg.tobytes()
        if 'head' in encoded:
            encoded['agentview'] = encoded['head']
        with self.frame_cond:
            self.frames = encoded
            self.frame_seq = frame_id
            self.frame_time = captured_at
            self.frame_cond.notify_all()
        if self.recording_enabled:
            self.recorder.update(rgb, self.frame_seq, captured_at)

    def state(self):
        state = self.robot_backend.state()
        with self.frame_cond:
            available = sorted(self.frames)
            cameras = list(self.camera_catalog)
            frame_seq, frame_time = self.frame_seq, self.frame_time
        state.update(self.recorder.state())
        groups = self.camera_groups(cameras)
        state.update(human_viewer=True, frame_seq=frame_seq, frame_time=frame_time,
                     recording_enabled=self.recording_enabled, require_task_done=False,
                     camera_views=[item['view'] for item in cameras], camera_catalog=cameras,
                     camera_groups=groups,
                     main_camera_group=next((group['id'] for group in groups
                                             if self.config.main_view in group['views']), 'robot'),
                     camera_labels={item['view']: item['label'] for item in cameras},
                     main_view=self.config.main_view, available_views=available,
                     depth_display={'min_m': self.config.depth_min_m, 'max_m': self.config.depth_max_m,
                                    'near': 'warm', 'far': 'cool', 'invalid': 'black'})
        if state.get('recording_error'):
            state['message'] = state['recording_error']
        return state

    def submit(self, kind, arg=None, timeout=None):
        if kind == 'start':
            if not self.recording_enabled:
                return {'ok': False, 'status': 405, 'message': 'Human recording is disabled for this profile.'}
            return self.recorder.start()
        if kind == 'stop':
            return self.recorder.finish()
        if kind == 'cancel':
            cancel = getattr(self.robot_backend, 'cancel_active_action', None)
            stopped = bool(cancel and cancel())
            result = self.recorder.finish(discard=True) if self.recorder.state()['recording'] else {
                'ok': stopped, 'message': 'Stopping active robot command.' if stopped else 'No recording or command is active.',
                **({} if stopped else {'status': 409})}
            if stopped and result.get('ok'):
                result['message'] += ' Stopping active robot command.'
            return result
        return self.robot_backend.submit(kind, arg, timeout=timeout)

    def recording_file(self, path):
        return self.recorder.recording_file(path)

    def close(self):
        self.recorder.close()
