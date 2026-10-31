"""Read previously delivered episode images. Run: python -m pytest -q tests/harness_integration/test_vercel_history.py."""
import base64
import json

HISTORY_IMAGE_TOOL = 'read_historical_rgb'


def historical_image_tool():
    return {'type': 'function', 'function': {'name': HISTORY_IMAGE_TOOL,
        'description': 'Recall one RGB image previously delivered in this episode, by decision step and camera name. '
            'Historical evidence only: not current vision, no robot motion. At most four recalled images per decision. '
            'Call separately from current sensor tools. Camera names are listed in earlier observation text.',
        'parameters': {'type': 'object', 'properties': {
            'decision_step': {'type': 'integer', 'minimum': 1},
            'camera': {'type': 'string'}}, 'required': ['decision_step', 'camera'], 'additionalProperties': False}}}


def register_current_images(session, request):
    session._current_decision_step = request.decision_step
    session._history_images_used = 0
    manifest = session.output / f'step_{request.decision_step:04d}' / 'model_inputs.json'
    if manifest.exists():
        session._history_images[request.decision_step] = {
            row['name']: {**row, 'snapshot_id': request.sensor_inputs.snapshot_id,
                          'captured_at': request.sensor_inputs.captured_at}
            for row in json.loads(manifest.read_text())}


def read_historical_image(session, arguments):
    result = {'ok': False, 'historical': True}
    if set(arguments) != {'decision_step', 'camera'}:
        return {**result, 'error': 'Specify decision_step and camera only.'}, None
    step, camera = arguments['decision_step'], arguments['camera']
    if type(step) is not int or step < 1 or step >= session._current_decision_step or not isinstance(camera, str):
        return {**result, 'error': 'Only earlier decisions in this episode may be recalled.'}, None
    record = session._history_images.get(step, {}).get(camera)
    if record is None:
        return {**result, 'error': 'That image was not delivered at the requested decision.'}, None
    if session._history_images_used >= 4:
        return {**result, 'error': 'Four historical images have already been recalled in this decision.'}, None
    path = (session.output / record['path']).resolve()
    if not path.is_relative_to(session.output.resolve()) or not path.is_file():
        return {**result, 'error': 'Archived episode image is unavailable.'}, None
    session._history_images_used += 1
    result.update(ok=True, decision_step=step, camera=camera, snapshot_id=record['snapshot_id'],
                  captured_at=record['captured_at'], shape=record['shape'],
                  note='This image is from a past decision, not the current robot view.')
    url = 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
    return result, url
