"""Match trusted release evaluators. Run `python -m pytest -q tests/harness_integration/test_task_edit2_mapping_world.py`."""
from copy import deepcopy

from wia.authoring.modules import module_identity


def fixed_release_evaluation(session, config):
    """Trust only the loaded release definition or one of its exact declared members.

    This selects a definition policy; it supplies no physical qualification proof
    and does not certify any Agent answer, visit, or other task completion.
    """
    task = getattr(session, 'loaded_task', None)
    if not getattr(task, 'release_v1', False):
        return False
    if not callable(getattr(task, 'verify_unchanged', None)):
        return False
    task.verify_unchanged()
    declared = task.definition.private.evaluation.model_dump()
    fields = ('type', 'module', 'operation', 'binding_keys', 'parameters')

    def normalized(value, *, member=False):
        result = {key: deepcopy(value.get(key)) for key in fields}
        result['module'] = module_identity(result['module'])
        if member:
            # The parent sets compatibility on its private per-observation copy.
            result['parameters'].pop('mapping_composition_contract', None)
        return result

    if normalized(config) == normalized(declared):
        return True
    if module_identity(declared['module']) != 'wia.authoring.converters.task_edit2_mapping_world':
        return False
    return any(normalized(config, member=True) == normalized(member['evaluation'], member=True)
               for member in declared['parameters']['members'])
