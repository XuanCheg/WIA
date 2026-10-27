"""Register complete-ground loading inside an existing native OmniGibson process.

Use wia_background.configure_scene with mode='complete_ground' before native
scene construction through scripts/check_esi_core_native.py. Importing this
module requires the installed OmniGibson runtime; CPU tests stub its parent API.
 Run: python -m pytest tests/test_extraction.py. """
from copy import deepcopy

from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.utils.constants import GROUND_CATEGORIES


class WiaCompleteGroundScene(InteractiveTraversableScene):
    """Keep the native predicate, relaxing only rejected ground room membership."""

    def _should_load_object(self, obj_info, task_metadata):
        original = super()._should_load_object(obj_info, task_metadata)
        if original or obj_info['args'].get('category', 'object') not in GROUND_CATEGORIES:
            return original
        amended = deepcopy(obj_info)
        in_rooms = amended['args'].get('in_rooms', [])
        in_rooms = [in_rooms] if isinstance(in_rooms, str) else in_rooms
        amended['args']['in_rooms'] = list(dict.fromkeys([*in_rooms, *(self.load_room_instances or [])]))
        return super()._should_load_object(amended, task_metadata)
