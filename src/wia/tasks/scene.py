"""Private minimal ESI patches and precondition checks. Use through run_esi_robot.py. Run: python -m pytest tests/test_extraction.py. """
from __future__ import annotations

from wia.authoring.modules import module_identity

import copy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .schema import ObjectPerturbationPatch, ObjectPosePatch, PrimitivePatch, RobotPosePatch


@dataclass(frozen=True)
class QualificationResult:
    status: str
    reason: str | None
    checks: dict

    def save(self, output, *, patches):
        path = Path(output) / "private_scene"
        path.mkdir(parents=True, exist_ok=True)
        (path / "qualification.json").write_text(json.dumps({"status": self.status,
            "reason": self.reason, "checks": self.checks, "applied_scene_patches": patches},
            indent=2, allow_nan=False) + "\n", encoding="utf-8")


class ScenePatches:
    def __init__(self, task, bindings):
        self.task = task
        self.bindings = bindings
        self.variant = task.definition.scene.variants[task.variant]
        self.seed = task.seed
        self.applied = []
        self._before_applied = self._after_applied = False
        # Preserve the existing placement / robot pose / nuisance RNG streams.
        placement_rng = np.random.default_rng(self.seed)
        robot_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1]))
        nuisance_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 2]))
        self._settings = {"camera_size": [1024, 1024]}
        self.restore_robot_head_mount = False
        self._realized = []
        shared_shift = None
        for patch in self.variant.patches:
            item = patch.model_dump(mode="json")
            if patch.operation == 'restore_robot_head_mount':
                self.restore_robot_head_mount = True
            elif isinstance(patch, RobotPosePatch):
                position = np.array(patch.position)
                yaw = patch.yaw_deg + float(robot_rng.choice(patch.heading_offsets_deg))
                position[:2] += robot_rng.uniform(-np.array(patch.jitter_xy_m), patch.jitter_xy_m)
                item.update(position=position.tolist(), yaw_deg=yaw)
                self._settings.update(robot_position=position.tolist(), robot_yaw_deg=yaw,
                                      head_pitch_deg=patch.head_pitch_deg)
            elif isinstance(patch, ObjectPosePatch):
                if patch.candidates:
                    index = int(placement_rng.integers(len(patch.candidates)))
                    candidate = patch.candidates[index]
                    position = np.array(candidate.position)
                    item.update(selected_index=index, selected_region=candidate.region)
                else:
                    position = np.array(patch.position)
                position[:2] += placement_rng.uniform(-np.array(patch.jitter_xy_m), patch.jitter_xy_m)
                item["position"] = position.tolist()
            elif isinstance(patch, ObjectPerturbationPatch):
                item["offsets_xy_m"] = {binding: nuisance_rng.uniform(-np.array(patch.jitter_xy_m),
                    patch.jitter_xy_m).tolist() for binding in patch.bindings}
            elif isinstance(patch, PrimitivePatch):
                if shared_shift is None:
                    shared_shift = float(placement_rng.uniform(-patch.shared_jitter_x_m, patch.shared_jitter_x_m))
                position = list(patch.position)
                position[0] += shared_shift
                item["position"] = position
            self._realized.append(item)

    def observation_settings(self):
        return copy.deepcopy(self._settings)

    def _binding(self, key):
        return self.bindings[key]

    def apply_to_env_config(self, config):
        if self._before_applied:
            raise RuntimeError("Scene patches are initialization-only.")
        objects = config.setdefault("objects", [])
        for item in self._realized:
            if item["phase"] != "before_construction":
                continue
            operation = item["operation"]
            if operation == 'category_initialization':
                from wia.native.esi_category_evaluation import owned_category_module
                module = owned_category_module(item['module'])
                hook = getattr(module, 'configure_scene', None)
                if not callable(hook):
                    raise ValueError('Owned category initialization requires configure_scene(config, parameters).')
                item['initialization_evidence'] = hook(config, copy.deepcopy(item['parameters']))
            elif operation == 'add_existing_r1':
                if config.get('robots'):
                    raise ValueError('Adding the existing R1 cannot replace or duplicate a source robot.')
                import omnigibson as og
                import yaml
                native = yaml.safe_load((Path(og.example_config_path) / 'r1_primitives.yaml').read_text())
                robots = native.get('robots', [])
                if len(robots) != 1 or robots[0].get('type') != 'R1':
                    raise ValueError('The installed R1 configuration is unavailable or mismatched.')
                config['robots'] = copy.deepcopy(robots)
            elif operation == "robot_pose":
                if not config.get("robots") or config["robots"][0].get("type") != "R1":
                    raise ValueError("Declared robot pose requires the existing R1.")
                robot = config["robots"][0]
                half = np.deg2rad(item["yaw_deg"]) / 2
                robot.update(position=item["position"], orientation=[0., 0., float(np.sin(half)), float(np.cos(half))],
                             pose_frame="world")
            elif operation in ("object_pose", "object_scale", "object_physics", "object_perturbation"):
                keys = [item["binding"]] if operation != "object_perturbation" else item["bindings"]
                for key in keys:
                    binding = self._binding(key)
                    obj = next((obj for obj in objects if obj.get("name") == binding.name), None)
                    if obj is None or any(getattr(binding, field) is not None and
                                          obj.get(field) != getattr(binding, field) for field in ("category", "model")):
                        raise ValueError("An explicit object patch does not match an existing ESI object.")
                    if operation == "object_pose":
                        obj["position"] = item["position"]
                        if item["quaternion_xyzw"] is not None:
                            obj["orientation"] = item["quaternion_xyzw"]
                    elif operation == "object_scale":
                        scale = np.asarray(obj.get("scale", [1., 1., 1.]), dtype=float)
                        obj["scale"] = (np.broadcast_to(scale, (3,)) * item["multiplier"]).tolist()
                    elif operation == 'object_physics':
                        obj['visual_only'] = False
                        if item['fixed_base'] is not None:
                            obj['fixed_base'] = False
                    else:
                        position = np.array(obj["position"], dtype=float)
                        position[:2] += item["offsets_xy_m"][key]
                        obj["position"] = position.tolist()
            elif operation == "add_primitive":
                if any(obj.get("name") == item["name"] for obj in objects):
                    raise ValueError("An added primitive cannot replace an existing object.")
                objects.append({"type": "PrimitiveObject", "name": item["name"], "primitive_type": "Cube",
                    "category": "object", "size": 1., "scale": item["dimensions_m"], "fixed_base": True,
                    "visual_only": False, "rgba": item["rgba"], "position": item["position"],
                    "orientation": item["quaternion_xyzw"]})
            elif operation in ("add_dataset_object", "add_usd_object"):
                if any(obj.get("name") == item["name"] for obj in objects):
                    raise ValueError("An added object cannot replace an existing ESI object.")
                obj = {"name": item["name"], "position": item["position"],
                       "orientation": item["quaternion_xyzw"], "visual_only": False}
                if operation == "add_dataset_object":
                    obj.update(type="DatasetObject", category=item["category"], model=item["model"],
                               scale=item["scale"], fixed_base=False)
                else:
                    from .loader import ROOT
                    asset = (ROOT / item["asset"]).resolve()
                    if not asset.is_relative_to(ROOT / "assets") or not asset.is_file():
                        raise ValueError("USD object asset must exist inside repository assets/.")
                    obj.update(type="USDObject", usd_path=str(asset), fixed_base=True,
                               kinematic_only=False, self_collisions=False)
                objects.append(obj)
            self.applied.append(copy.deepcopy(item))
        if any(item['operation'] == 'prepare_category_interaction' and item.get('allow_fallback_table')
               for item in self._realized):
            # Source count/inspection builders often filter the room to walls
            # and floors. Restore existing furniture before searching surfaces.
            native_scene = config.get('scene', {})
            if native_scene.get('load_object_categories'):
                support_categories = ('table', 'breakfast_table', 'coffee_table', 'dining_table',
                    'desk', 'countertop', 'counter', 'straight_chair', 'stool', 'bench', 'shelf')
                native_scene['load_object_categories'] = list(dict.fromkeys(
                    [*native_scene['load_object_categories'], *support_categories]))
            from .interaction_supports import preload_supports
            slots = max((len(item.get('placement_groups', ())) or 1 for item in self._realized
                         if item['operation'] == 'prepare_category_interaction'), default=1)
            preload_supports(config, slots)
        self._before_applied = True

    def source_postprocess_override(self, session):
        """Use an explicit owned joint-world initializer instead of one child hook.

        Ordinary source tasks retain their original postprocessor. A composed
        world may have exactly one category initializer that explicitly owns
        every source mechanism and returns its native construction receipt.
        """
        from wia.native.esi_category_evaluation import owned_category_module
        hooks = []
        for item in self._realized:
            if item['operation'] != 'category_initialization' or item['phase'] != 'before_construction':
                continue
            module = owned_category_module(item['module'])
            callback = getattr(module, 'postprocess_joint_world', None)
            if callable(callback):
                hooks.append((item, callback))
        if not hooks:
            return None
        if len(hooks) != 1:
            raise ValueError('A composed world requires one owner of source mechanism initialization.')
        item, callback = hooks[0]
        result = callback(session, copy.deepcopy(item['parameters']))
        if not isinstance(result, dict) or result.get('joint_world_postprocess') is not True:
            raise ValueError('Joint-world postprocessing requires an explicit native receipt.')
        self.applied.append({'operation': 'joint_world_postprocess', 'module': item['module'],
                             'initialization_evidence': copy.deepcopy(result)})
        return result

    def apply_after_postprocessing(self, session):
        if not self._before_applied or self._after_applied:
            raise RuntimeError("Postprocessing patches must run exactly once after construction.")
        session._release_edge_policy = bool(getattr(self.task, 'release_v1', False)
            or getattr(session, '_capture_release_edge_policy', False))
        for item in self._realized:
            if item['operation'] == 'prepare_category_interaction':
                from .interaction_setup import initialize_interaction
                item['initialization_evidence'] = initialize_interaction(session, self.bindings, item)
                self.applied.append(copy.deepcopy(item))
                for initialized in self.applied:
                    if (initialized['operation'] == 'category_initialization'
                            and initialized.get('phase') == 'after_postprocessing'):
                        from wia.native.esi_category_evaluation import owned_category_module
                        callback = getattr(owned_category_module(initialized['module']), 'on_interaction_initialized', None)
                        if callable(callback):
                            callback(session, copy.deepcopy(item['initialization_evidence']))
                continue
            if item['operation'] == 'category_initialization':
                if item['phase'] == 'after_postprocessing':
                    from wia.native.esi_category_evaluation import owned_category_module
                    module = owned_category_module(item['module'])
                    hook = getattr(module, 'initialize_scene', None)
                    if not callable(hook):
                        raise ValueError('Owned category initialization requires initialize_scene(session, parameters).')
                    session._category_initialization_active = True
                    try:
                        item['initialization_evidence'] = hook(session, copy.deepcopy(item['parameters']))
                    finally:
                        session._category_initialization_active = False
                    self.applied.append(copy.deepcopy(item))
                continue
            if item['operation'] == 'restore_robot_head_mount':
                mounts = getattr(session, '_category_sensor_mounts', {})
                if not mounts or not session.env.robots:
                    raise ValueError('Original native robot head mount was not captured before source postprocessing.')
                for name, (position, quaternion) in mounts.items():
                    session.env.robots[0].sensors[name].set_position_orientation(
                        position=position, orientation=quaternion, frame='parent')
                self.applied.append({**copy.deepcopy(item), 'restored_sensors': sorted(mounts)})
                continue
            if item["operation"] == "add_usd_object":
                from isaacsim.core.api.materials import OmniGlass
                from omnigibson.utils.physx_utils import bind_material
                import omnigibson.lazy as lazy
                import torch
                obj = session.env.scene.object_registry("name", item["name"])
                meshes = {mesh.prim.GetName(): mesh for link in obj.links.values()
                          for mesh in link.visual_meshes.values()}
                if item["glass_mesh_names"]:
                    material = item["glass_material"]
                    material_path = obj.prim_path + "/Looks/OmniGlassClear"
                    OmniGlass(material_path, color=np.array(material["color"]), ior=material["ior"],
                              depth=material["depth_m"], thin_walled=material["thin_walled"])
                for name in item["glass_mesh_names"]:
                    if name not in meshes:
                        raise ValueError("Declared glass mesh is missing from the native asset.")
                    bind_material(meshes[name].prim_path, material_path)
                    meshes[name].prim.CreateAttribute("primvars:doNotCastShadows",
                        lazy.pxr.Sdf.ValueTypeNames.Bool).Set(not material["cast_shadows"])
                from omnigibson.utils.constants import JointType
                for unit, kind in (("rad", JointType.JOINT_REVOLUTE), ("m", JointType.JOINT_PRISMATIC)):
                    for name, position in item[f"initial_joint_positions_{unit}"].items():
                        joint = obj.joints[name]
                        if joint.joint_type != kind:
                            raise ValueError("Initial joint position unit does not match the native joint type.")
                        if not joint.lower_limit <= position <= joint.upper_limit:
                            raise ValueError("Initial joint position exceeds the native limits.")
                        joint.set_pos(torch.tensor([position]), drive=False)
                        joint.set_vel(torch.zeros(1), drive=False)
            if item["phase"] != "after_postprocessing":
                continue
            binding = self._binding(item["binding"])
            obj = session.env.scene.object_registry("name", binding.name)
            if obj is None:
                raise ValueError("Postprocessing patch requires an existing object.")
            if item['operation'] == 'object_physics':
                if any(getattr(binding, field) is not None and getattr(obj, field, None) != getattr(binding, field)
                       for field in ('category', 'model')):
                    raise ValueError('Native physics restoration must match the original object binding.')
                # The installed native setter enables collisions and gravity on
                # every owned link. Source postprocessing can otherwise reset it.
                # The native setter asserts stopped GPU physics even for a
                # redundant assignment. Keep already physical source objects
                # intact; an actual restoration still uses the native setter.
                if obj.visual_only:
                    obj.visual_only = False
                self.applied.append(copy.deepcopy(item))
                continue
            import torch
            values = {"position": torch.tensor(item["position"], dtype=torch.float32)}
            if item["quaternion_xyzw"] is not None:
                values["orientation"] = torch.tensor(item["quaternion_xyzw"], dtype=torch.float32)
            obj.set_position_orientation(**values)
            self.applied.append(copy.deepcopy(item))
        if session._release_edge_policy:
            from .pickup_edges import apply_edge_policy, refresh_edge_references, settle_edge_policy
            names = list(dict.fromkeys([binding.name for binding in self.bindings.values()]
                + list(getattr(session, 'task_object_names', ()))
                + list(getattr(session, 'task_state', {}).get('dynamic_object_names', ()))
                + list(getattr(session, 'postprocess', {}).get('dynamic_object_names', ()))))
            objects = [session.env.scene.object_registry('name', name) for name in names]
            session._category_initialization_active = True
            try:
                placements = apply_edge_policy(session, objects)
                settle_edge_policy(session, placements)
                session._edge_placement_bounds = {}
                refresh_edge_references(getattr(session, 'category_interaction_setup', {}), placements, session)
                for applied in self.applied:
                    refresh_edge_references(applied.get('initialization_evidence'), placements, session)
            finally:
                session._category_initialization_active = False
            self.applied.append({'operation': 'current_support_edge_policy',
                'initialization_evidence': {'schema': 'pickup_edge_policy_v2',
                    'margin_m': .03, 'placements': copy.deepcopy(session.pickup_edge_placements),
                    'physical_grasp_verified': False}})
        self._after_applied = True


class DeclarativeScene(ScenePatches):
    def __init__(self, task):
        super().__init__(task, task.definition.private.bindings)

    def qualify(self, session, executor, output, *, observation=None):
        if self.task.release_v1:
            result = QualificationResult('loaded_release_v1', None, {
                'physical_checks_run': False, 'policy': 'fixed_v1_definition',
                'physical_safety_inferred': False})
            result.save(output, patches=self.applied)
            return result
        from wia.native.utils import _array
        from wia.native.esi_visibility import PrivateVisibilityCheck
        checks = {}
        reason = None
        definition = self.task.definition
        try:
            self.task.verify_unchanged()
            placement = [item['initialization_evidence'] for item in self.applied
                if item['operation'] == 'prepare_category_interaction']
            if placement:
                checks['interaction_initialization'] = placement[-1]
                if not placement[-1].get('accepted'):
                    reason = 'interaction_layout_unqualified'
                    raise ValueError('The measured interaction layout does not have a safe qualified start.')
            if (session.scene_name, session.room_name) != (self.task.scene_name, self.task.room_name):
                reason = "source_scene_mismatch"
                raise ValueError("Loaded scene/room differs from the source.")
            missing = [name for name in definition.requirements.capabilities
                       if not getattr(executor.capabilities, name)]
            checks["required_capabilities"] = not missing
            if missing:
                reason = "unsupported_robot_capabilities"
                raise ValueError("Required capabilities are unavailable.")
            composite = (definition.private.evaluation.type == 'category_operation'
                and module_identity(definition.private.evaluation.module) == 'wia.authoring.converters.composite_task'
                and definition.private.evaluation.operation == 'composite_goals')
            bound, unavailable = {}, []
            for key, binding in definition.private.bindings.items():
                obj = session.env.scene.object_registry("name", binding.name)
                if obj is None or any(getattr(binding, field) is not None and
                    getattr(obj, field, None) != getattr(binding, field) for field in ("category", "model")):
                    if composite:
                        unavailable.append(key)
                        continue
                    reason = "private_binding_mismatch"
                    raise ValueError("A private binding does not match the actual object.")
                bound[key] = obj
            checks["object_bindings"] = not unavailable
            if composite:
                checks['unavailable_composite_bindings'] = unavailable
                checks['partial_goal_qualification'] = True
            operation_goal = definition.private.evaluation.type == 'category_operation'
            target = None
            if operation_goal:
                from wia.native.esi_category_evaluation import qualify_category_operation
                config = definition.private.evaluation.model_dump()
                config.update(scene=self.task.scene_name, room=self.task.room_name,
                    bindings={key: obj.name for key, obj in bound.items()})
                checks['operation_preconditions'] = qualify_category_operation(session, executor, config, bound)
                if not checks['operation_preconditions']['accepted']:
                    reason = checks['operation_preconditions'].get('reason') or 'operation_precondition_failed'
                    raise ValueError('The native operation preconditions are not satisfied.')
            else:
                target = bound[definition.private.evaluation.target]
                movable = not target.visual_only and not target.fixed_base
                checks["target_movable_collidable"] = movable
                if not movable:
                    reason = "target_not_movable_collidable"
                    raise ValueError("Physical target cannot be grasped.")
            if definition.qualification.support:
                if target is None:
                    raise ValueError('Category operations must measure their own support preconditions.')
                low, high = (_array(value) for value in target.aabb)
                support_low, support_high = (_array(value) for value in bound[definition.qualification.support].aabb)
                supported = bool(np.all(low[:2] >= support_low[:2]) and np.all(high[:2] <= support_high[:2])
                    and abs(low[2] - support_high[2]) <= definition.qualification.support_tolerance_m)
                from .interaction_setup import support_contact_paths, touching_task_support, on_top_task_support
                repaired_paths = support_contact_paths(session, target.name)
                if repaired_paths:
                    checks['original_support_aabb_screen'] = supported
                    values = session.env.scene.objects
                    values = values.values() if isinstance(values, dict) else values
                    actual_support = next((obj for obj in values if obj.prim_path in repaired_paths), None)
                    if actual_support is None:
                        supported = False
                    else:
                        adapted_low, adapted_high = (_array(value) for value in actual_support.aabb)
                        surface_z = session.category_interaction_setup['support_surface_z_m']
                        supported = bool(np.all(low[:2] >= adapted_low[:2]) and np.all(high[:2] <= adapted_high[:2])
                            and abs(float(low[2]) - surface_z) <= definition.qualification.support_tolerance_m
                            and touching_task_support(session, target, None)
                            and on_top_task_support(session, target, None))
                        checks['repaired_support_name'] = actual_support.name
                checks["support_aabb_screen"] = supported
                if not supported:
                    reason = "initial_support_failed"
                    raise ValueError("Target failed its physical support screen.")
            for item in self.applied:
                if item["operation"] == "add_primitive":
                    obj = session.env.scene.object_registry("name", item["name"])
                    if obj is None or obj.visual_only or not obj.fixed_base:
                        reason = "primitive_geometry_failed"
                        raise ValueError("Added primitive must be fixed collidable geometry.")
            if definition.private.evaluation.type in ("open_extract_lift", "slide_extract_lift"):
                from wia.native.esi_glass_box import inspect_closed_box, inspect_closed_sliding_box
                inspect = (inspect_closed_sliding_box if definition.private.evaluation.type == "slide_extract_lift"
                           else inspect_closed_box)
                checks["closed_box"] = inspect(session, definition.private.evaluation,
                    self._binding(definition.private.evaluation.container).name, target)
                if not checks["closed_box"]["passed"]:
                    reason = "closed_box_precondition_failed"
                    raise ValueError("The apple must start inside a closed, collidable box.")
            condition = self.variant.initial_visibility or definition.qualification.initial_visibility
            if condition != "none":
                if target is None:
                    raise ValueError('Category operations must declare their own native visibility preconditions.')
                checker = PrivateVisibilityCheck(session, {"scene": self.task.scene_name,
                    "room": self.task.room_name, "target": target.name, "camera_scope": "enabled_agent",
                    "initial_condition": condition, "minimum_pixels": definition.qualification.minimum_pixels,
                    "minimum_extent_px": definition.qualification.minimum_extent_px}, Path(output) / "private_visibility")
                reason = "initial_visibility_failed"
                checks["visibility"] = checker.check_initial(observation)
                session.private_visibility_checker = checker
            result = QualificationResult("accepted_minimal_patch" if self.applied else "accepted_original", None, checks)
        except (ValueError, RuntimeError) as error:
            checks["private_error"] = str(error)
            result = QualificationResult("skipped", reason or "scene_precondition_failed", checks)
        result.save(output, patches=self.applied)
        return result

    def create_evaluator(self, session, controller, output):
        from wia.native.esi_task_evaluation import GraspLiftEvaluator
        from wia.native.esi_glass_box import OpenExtractLiftEvaluator, SlideExtractLiftEvaluator
        # Evaluator selection depends on supported evaluator types, never task IDs.
        evaluators = {"grasp_lift": GraspLiftEvaluator, "open_extract_lift": OpenExtractLiftEvaluator,
                      "slide_extract_lift": SlideExtractLiftEvaluator}
        if self.task.definition.private.evaluation.type == 'stack_relations':
            from wia.native.esi_category_evaluation import StackRelationsEvaluator
            evaluators['stack_relations'] = StackRelationsEvaluator
        if self.task.definition.private.evaluation.type == 'extract_lift':
            from wia.native.esi_category_evaluation import ExtractLiftEvaluator
            evaluators['extract_lift'] = ExtractLiftEvaluator
        if self.task.definition.private.evaluation.type == 'category_operation':
            from wia.native.esi_category_evaluation import create_category_operation_evaluator
            evaluators['category_operation'] = create_category_operation_evaluator
        config = self.task.definition.private.evaluation.model_dump()
        config.update(scene=self.task.scene_name, room=self.task.room_name)
        if config['type'] == 'category_operation':
            config['bindings'] = {key: self._binding(key).name for key in config['binding_keys']}
        else:
            config['target'] = self._binding(config['target']).name
        if config["type"] in ('extract_lift', "open_extract_lift", "slide_extract_lift"):
            config["container"] = self._binding(config["container"]).name
        if config['type'] == 'stack_relations':
            config['relations'] = [{'upper': self._binding(r['upper']).name,
                                    'lower': self._binding(r['lower']).name} for r in config['relations']]
        from wia.tasks.background_policy import install_background_policy
        try:
            install_background_policy(session, self.task)
        except Exception as error:
            if self.task.run_mode != 'diagnostic':
                raise
            session.background_policy_error = f'{type(error).__name__}: {error}'
            session.background_movement_policy = None
        try:
            evaluator = evaluators[config["type"]](session, controller, config, output)
        except Exception as error:
            if self.task.run_mode != 'diagnostic':
                raise
            from wia.native.esi_category_evaluation import UnavailableEvaluator
            evaluator = UnavailableEvaluator(config, 'evaluator_initialization', error)
        if self.task.definition.kind == 'category_robot_task':
            from wia.native.esi_category_evaluation import CategoryEvaluationEnvelope
            evaluator = CategoryEvaluationEnvelope(evaluator, self.task, output)
        if hasattr(evaluator, "sample_step"):
            previous_on_step = controller.on_step
            def sample_evaluation():
                if previous_on_step:
                    previous_on_step()
                evaluator.sample_step()
            controller.on_step = sample_evaluation
        return evaluator
