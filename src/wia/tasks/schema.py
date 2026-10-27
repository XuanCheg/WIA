"""Closed task contracts. Validate through wia.tasks.loader.load_task(path)."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Positive = Annotated[Number, Field(gt=0)]
Nonnegative = Annotated[Number, Field(ge=0)]
Vector3 = tuple[Number, Number, Number]
Vector2 = tuple[Nonnegative, Nonnegative]
Quaternion = tuple[Number, Number, Number, Number]
Text = Annotated[StrictStr, Field(min_length=1)]
Capability = Literal["base_translation", "base_rotation", "ee_translation", "gripper", "ee_rotation", "trunk_height"]


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)



class Source(ClosedModel):
    provider: Literal["esi"]
    json_path: Text = Field(alias="json")

    @model_validator(mode="after")
    def relative_source(self):
        path = PurePosixPath(self.json_path)
        if (path.is_absolute() or ".." in path.parts or "\\" in self.json_path
                or len(path.parts) != 5 or path.suffix != ".json"):
            raise ValueError("ESI source must be Big Task/Small Task/Scene/Room/sample.json.")
        return self


class PublicTask(ClosedModel):
    objective: Text
    success_criteria: Text
    environment: Text
    constraints: tuple[Text, ...] = ()

    def task_text(self):
        parts = [self.objective.strip(), "Physical success criteria: " + self.success_criteria.strip()]
        if self.constraints:
            parts.append("Task constraints: " + "; ".join(self.constraints))
        return "\n\n".join(parts)


class Requirements(ClosedModel):
    capabilities: tuple[Capability, ...]


class Binding(ClosedModel):
    name: Text
    category: Text | None = None
    model: Text | None = None


class GraspLiftEvaluation(ClosedModel):
    type: Literal["grasp_lift"]
    target: Text
    minimum_lift_m: Positive


class ExtractLiftEvaluation(ClosedModel):
    type: Literal['extract_lift']
    target: Text
    container: Text
    minimum_lift_m: Positive = 0.05


class CategoryOperationEvaluation(ClosedModel):
    """An owned category adapter supplies private, read-only native evidence."""
    type: Literal['category_operation']
    module: Text
    operation: Text
    binding_keys: tuple[Text, ...] = ()
    parameters: dict = Field(default_factory=dict)

    @model_validator(mode='after')
    def owned_module(self):
        from wia.authoring.modules import is_owned_module
        if not is_owned_module(self.module):
            raise ValueError('Category operation adapters must be owned converter modules.')
        if len(set(self.binding_keys)) != len(self.binding_keys):
            raise ValueError('Category operation bindings must be unique.')
        return self


class StackRelation(ClosedModel):
    upper: Text
    lower: Text


class StackRelationsEvaluation(ClosedModel):
    type: Literal['stack_relations']
    target: Text
    relations: tuple[StackRelation, ...] = Field(min_length=1)
    stability_steps: Annotated[StrictInt, Field(ge=1)] = 15
    minimum_lift_m: Positive = 0.05


class OpenExtractLiftEvaluation(ClosedModel):
    type: Literal["open_extract_lift"]
    target: Text
    container: Text
    hinge_joint: Text = "lid_hinge"
    minimum_open_angle_deg: Annotated[Number, Field(gt=0, le=110)] = 60.
    rim_height_m: Positive
    extraction_clearance_m: Positive = 0.005
    minimum_lift_m: Positive = 0.05


class SlideExtractLiftEvaluation(ClosedModel):
    type: Literal["slide_extract_lift"]
    target: Text
    container: Text
    slide_joint: Text = "lid_slide"
    minimum_open_displacement_m: Positive = 0.20
    rim_height_m: Positive
    extraction_clearance_m: Positive = 0.005
    minimum_lift_m: Positive = 0.05


class PrivateTask(ClosedModel):
    bindings: dict[str, Binding]
    evaluation: Annotated[GraspLiftEvaluation | ExtractLiftEvaluation | CategoryOperationEvaluation | OpenExtractLiftEvaluation | SlideExtractLiftEvaluation | StackRelationsEvaluation,
                          Field(discriminator="type")]


class RobotPosePatch(ClosedModel):
    operation: Literal["robot_pose"]
    phase: Literal["before_construction"] = "before_construction"
    position: Vector3
    yaw_deg: Number
    heading_offsets_deg: tuple[Number, ...] = (0.,)
    jitter_xy_m: Vector2 = (0., 0.)
    head_pitch_deg: Annotated[Number, Field(ge=-90, le=90,
        description="Additional local downward camera rotation relative to the native head mount; not an absolute optical pitch.")] = 45.


class Placement(ClosedModel):
    region: Text
    position: Vector3


class ObjectPosePatch(ClosedModel):
    operation: Literal["object_pose"]
    phase: Literal["before_construction", "after_postprocessing"] = "before_construction"
    binding: Text
    position: Vector3 | None = None
    candidates: tuple[Placement, ...] = ()
    quaternion_xyzw: Quaternion | None = None
    jitter_xy_m: Vector2 = (0., 0.)

    @model_validator(mode="after")
    def one_placement(self):
        if (self.position is None) == (not self.candidates):
            raise ValueError("Declare one position or nonempty placement candidates.")
        if len({item.region for item in self.candidates}) != len(self.candidates):
            raise ValueError("Placement region names must be unique.")
        if self.quaternion_xyzw is not None and not any(self.quaternion_xyzw):
            raise ValueError("Quaternion must be nonzero.")
        return self


class RobotHeadMountPatch(ClosedModel):
    operation: Literal['restore_robot_head_mount']
    phase: Literal['after_postprocessing'] = 'after_postprocessing'


class ExistingRobotPatch(ClosedModel):
    """Add the installed R1 configuration to an originally robot-free builder."""
    operation: Literal['add_existing_r1']
    phase: Literal['before_construction'] = 'before_construction'


class CategoryInitializationPatch(ClosedModel):
    """An owned converter may restore original mechanisms during initialization."""
    operation: Literal['category_initialization']
    phase: Literal['before_construction', 'after_postprocessing'] = 'after_postprocessing'
    module: Text
    parameters: dict = Field(default_factory=dict)

    @model_validator(mode='after')
    def owned_module(self):
        from wia.authoring.modules import is_owned_module
        if not is_owned_module(self.module):
            raise ValueError('Category initialization must use an owned converter module.')
        return self


class CategoryInteractionPatch(ClosedModel):
    """Screen native starts and declare reversible support placement rules."""
    operation: Literal['prepare_category_interaction']
    phase: Literal['after_postprocessing'] = 'after_postprocessing'
    mode: Literal['navigation', 'manipulation', 'articulation', 'inspection']
    group_bindings: tuple[Text, ...] = ()
    manipulation_bindings: tuple[Text, ...] = ()
    relocation_policy: Literal['preserve', 'rigid_group', 'comparison_pairs', 'relation_group', 'vertical_only', 'independent_groups', 'shared_tabletop', 'seeded_geometry'] = 'preserve'
    layout_parameters: dict = Field(default_factory=dict)
    controller_initial_trunk_joint_positions_rad: tuple[Number, Number, Number, Number] | None = None
    placement_groups: tuple[tuple[Text, ...], ...] = ()
    support_bindings: tuple[Text, ...] = ()
    support_preferences: dict[Text, Text] = Field(default_factory=dict)
    support_search: Literal['bounded_group', 'measured_surfaces'] = 'bounded_group'
    allow_fallback_table: StrictBool = False
    anchor_policy: Literal['none', 'illumination_common_translation', 'liquid_volume_common_translation'] = 'none'
    source_camera_policy: Literal['translate_with_group', 'preserve'] = 'translate_with_group'
    fixed_reference_evidence: dict = Field(default_factory=dict)
    source_region: Text | None = None
    minimum_grasp_height_m: Annotated[Number, Field(ge=0, le=2)] = .35
    maximum_grasp_height_m: Annotated[Number, Field(gt=0, le=2)] = 1.10
    maximum_stance_distance_m: Annotated[Number, Field(gt=0, le=2)] = .85
    warning_codes: tuple[Text, ...] = ()

    @model_validator(mode='after')
    def coherent_policy(self):
        if self.minimum_grasp_height_m >= self.maximum_grasp_height_m:
            raise ValueError('Interaction height window must be increasing.')
        if not set(self.manipulation_bindings) <= set(self.group_bindings):
            raise ValueError('Every manipulation binding must belong to the preserved group.')
        if not set(self.support_bindings) <= set(self.group_bindings):
            raise ValueError('Load-bearing roots must belong to the movable group.')
        if self.relocation_policy == 'rigid_group' and (not self.group_bindings or self.mode == 'navigation'):
            raise ValueError('Only an explicit physical task group may be relocated.')
        if self.relocation_policy == 'vertical_only' and (not self.group_bindings or self.mode == 'navigation'):
            raise ValueError('Vertical support placement requires an explicit physical group.')
        if self.anchor_policy != 'none' and self.relocation_policy != 'vertical_only':
            raise ValueError('Lighting anchors require a common vertical-only translation.')
        if self.fixed_reference_evidence and self.relocation_policy != 'vertical_only':
            raise ValueError('A fixed scene-native reference requires vertical-only placement.')
        if self.relocation_policy == 'relation_group' and (len(self.group_bindings) < 2 or self.mode != 'navigation'):
            raise ValueError('Relation support placement requires a surveyed object formation.')
        if self.relocation_policy == 'comparison_pairs':
            members = [key for group in self.placement_groups for key in group]
            if (self.mode != 'manipulation' or len(self.placement_groups) != 2
                    or any(len(group) != 2 for group in self.placement_groups)
                    or len(set(members)) != 4 or set(members) != set(self.group_bindings)
                    or set(self.manipulation_bindings) != {group[0] for group in self.placement_groups}):
                raise ValueError('Comparison placement requires two distinct target/reference pairs.')
        elif self.relocation_policy in ('independent_groups', 'vertical_only', 'shared_tabletop') and self.placement_groups:
            members = [key for group in self.placement_groups for key in group]
            if (any(not group for group in self.placement_groups) or len(members) != len(set(members))
                    or set(members) != set(self.group_bindings) or self.mode == 'navigation'):
                raise ValueError('Physical placement groups must partition the declared movable bindings.')
        elif self.relocation_policy == 'independent_groups' or self.placement_groups:
            raise ValueError('Independent placement requires explicit physical groups.')
        if self.relocation_policy in ('shared_tabletop', 'seeded_geometry'):
            if not self.group_bindings or self.mode == 'navigation':
                raise ValueError('Revised tabletop placement requires an explicit physical inventory.')
            if self.relocation_policy == 'seeded_geometry' and not self.layout_parameters:
                raise ValueError('Seeded geometry requires its explicit versioned layout contract.')
        if set(self.support_preferences) - set(self.group_bindings):
            raise ValueError('Support preferences must refer to movable group members.')
        return self


class ObjectPhysicsPatch(ClosedModel):
    """Enable an existing task object for real manipulation without resizing it."""
    operation: Literal['object_physics']
    phase: Literal['before_construction', 'after_postprocessing'] = 'before_construction'
    binding: Text
    visual_only: Literal[False] = False
    fixed_base: Literal[False] | None = None

    @model_validator(mode='after')
    def construction_only_base(self):
        if self.phase == 'after_postprocessing' and self.fixed_base is not None:
            raise ValueError('Fixed-base changes must be declared before native construction.')
        return self


class ObjectScalePatch(ClosedModel):
    operation: Literal["object_scale"]
    phase: Literal["before_construction"] = "before_construction"
    binding: Text
    multiplier: Positive


class PrimitivePatch(ClosedModel):
    operation: Literal["add_primitive"]
    phase: Literal["before_construction"] = "before_construction"
    name: Text
    primitive_type: Literal["Cube"] = "Cube"
    position: Vector3
    dimensions_m: tuple[Positive, Positive, Positive]
    quaternion_xyzw: Quaternion = (0., 0., 0., 1.)
    rgba: tuple[Number, Number, Number, Number] = (0.25, 0.32, 0.40, 1.)
    # The existing Milestone 8 screens share one bounded X displacement.
    shared_jitter_x_m: Nonnegative = 0.


class ObjectPerturbationPatch(ClosedModel):
    operation: Literal["object_perturbation"]
    phase: Literal["before_construction"] = "before_construction"
    bindings: tuple[Text, ...]
    jitter_xy_m: Vector2


class DatasetObjectPatch(ClosedModel):
    operation: Literal["add_dataset_object"]
    phase: Literal["before_construction"] = "before_construction"
    name: Text
    category: Text
    model: Text
    position: Vector3
    scale: Positive | tuple[Positive, Positive, Positive] = 1.
    quaternion_xyzw: Quaternion = (0., 0., 0., 1.)


class GlassMaterial(ClosedModel):
    color: tuple[Number, Number, Number] = (1., 1., 1.)
    ior: Positive = 1.45
    depth_m: Positive = 0.1
    thin_walled: StrictBool = True
    cast_shadows: StrictBool = False


class USDObjectPatch(ClosedModel):
    operation: Literal["add_usd_object"]
    phase: Literal["before_construction"] = "before_construction"
    name: Text
    asset: Text
    position: Vector3
    quaternion_xyzw: Quaternion = (0., 0., 0., 1.)
    glass_mesh_names: tuple[Text, ...] = ()
    glass_material: GlassMaterial = Field(default_factory=GlassMaterial)
    initial_joint_positions_rad: dict[Text, Number] = Field(default_factory=dict)
    initial_joint_positions_m: dict[Text, Number] = Field(default_factory=dict)

    @model_validator(mode="after")
    def local_asset(self):
        path = PurePosixPath(self.asset)
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "assets" or path.suffix not in (".usd", ".usda"):
            raise ValueError("USD patches require a repository asset under assets/.")
        if self.initial_joint_positions_rad.keys() & self.initial_joint_positions_m.keys():
            raise ValueError("Each joint must declare one position unit.")
        return self


Patch = Annotated[RobotPosePatch | RobotHeadMountPatch | ExistingRobotPatch | CategoryInitializationPatch | CategoryInteractionPatch | ObjectPhysicsPatch | ObjectPosePatch | ObjectScalePatch | PrimitivePatch | ObjectPerturbationPatch
                  | DatasetObjectPatch | USDObjectPatch,
                  Field(discriminator="operation")]
Visibility = Literal["none", "visible", "hidden_zero", "hidden_unrecognizable"]


class SceneVariant(ClosedModel):
    patches: tuple[Patch, ...]
    initial_visibility: Visibility | None = None

    @model_validator(mode="after")
    def patch_invariants(self):
        robot = [patch for patch in self.patches if isinstance(patch, RobotPosePatch)]
        primitives = [patch for patch in self.patches if isinstance(patch, PrimitivePatch)]
        added = [patch for patch in self.patches if isinstance(patch, (PrimitivePatch, DatasetObjectPatch, USDObjectPatch))]
        if len(robot) > 1 or any(not patch.heading_offsets_deg for patch in robot):
            raise ValueError("Use at most one robot pose with nonempty heading offsets.")
        if len({patch.name for patch in added}) != len(added):
            raise ValueError("Added object names must be unique.")
        if len({patch.shared_jitter_x_m for patch in primitives}) > 1:
            raise ValueError("Added screens must share the same bounded X perturbation.")
        for patch in primitives:
            if not any(patch.quaternion_xyzw) or any(not 0 <= value <= 1 for value in patch.rgba):
                raise ValueError("Primitive quaternion and RGBA must be valid.")
        return self


class Scene(ClosedModel):
    policy: Literal["native_then_minimal_patch_else_skip"]
    variants: dict[str, SceneVariant]

    @model_validator(mode="after")
    def original_is_native(self):
        if "original" not in self.variants or self.variants["original"].patches:
            raise ValueError("The original variant must exist and contain no patches.")
        return self


class Qualification(ClosedModel):
    on_failure: Literal["skip"]
    target_movable_collidable: StrictBool = True
    support: Text | None = None
    support_tolerance_m: Positive = 0.005
    initial_visibility: Visibility = "none"
    minimum_pixels: Annotated[StrictInt, Field(gt=0)] = 64
    minimum_extent_px: Annotated[StrictInt, Field(gt=0)] = 8


class Defaults(ClosedModel):
    seed: Annotated[StrictInt, Field(ge=0)] = 7
    max_steps: Annotated[StrictInt, Field(gt=0)] = 60
    wall_timeout_s: Positive = 5400.
    world_advances_during_inference: StrictBool = True
    observation_condition: Literal["current", "initial_view_only"] = "current"
    action_mode: Literal["fixed", "parameterized"] = "parameterized"


def resolve_episode_defaults(value):
    """Resolve explicit budgets and schema defaults identically at every entry."""
    defaults = value.get('defaults', {}) if isinstance(value, dict) else value.defaults
    return defaults if isinstance(defaults, Defaults) else Defaults.model_validate(defaults)


class RigidContainmentTaskDefinition(ClosedModel):
    """Original containment question paired with a separate approved layout."""
    kind: Literal["rigid_containment_task"]
    id: Text
    version: Annotated[StrictInt, Field(gt=0)] = 1
    status: Literal["pending", "ready", "discarded"] = "pending"
    source: Source
    scene_file: Text
    targets: tuple[Text, ...]
    goal: Literal["answer_and_place"]
    allow_remove_after_verification: Literal[True] = True
    public: PublicTask
    defaults: Defaults = Field(default_factory=lambda: Defaults(seed=0))

    @model_validator(mode="after")
    def scene_and_targets(self):
        path = PurePosixPath(self.scene_file)
        if path.is_absolute() or ".." in path.parts or path.suffix not in (".yaml", ".yml"):
            raise ValueError("Task scene_file must be a relative scene YAML path.")
        if len(self.targets) not in (1, 2, 3) or len(set(self.targets)) != len(self.targets):
            raise ValueError("A containment task requires one to three distinct object bindings.")
        return self


class TaskDefinition(ClosedModel):
    id: Text
    version: Annotated[StrictInt, Field(gt=0)]
    source: Source
    public: PublicTask
    requirements: Requirements
    private: PrivateTask
    scene: Scene
    qualification: Qualification
    defaults: Defaults

    @model_validator(mode="after")
    def references(self):
        bindings = self.private.bindings
        references = (set(self.private.evaluation.binding_keys) if isinstance(self.private.evaluation, CategoryOperationEvaluation)
                      else {self.private.evaluation.target})
        if isinstance(self.private.evaluation, StackRelationsEvaluation):
            for relation in self.private.evaluation.relations:
                references.update((relation.upper, relation.lower))
        if isinstance(self.private.evaluation, (ExtractLiftEvaluation, OpenExtractLiftEvaluation, SlideExtractLiftEvaluation)):
            references.add(self.private.evaluation.container)
        if self.qualification.support:
            references.add(self.qualification.support)
        for variant in self.scene.variants.values():
            for patch in variant.patches:
                if isinstance(patch, (ObjectPosePatch, ObjectScalePatch, ObjectPhysicsPatch)):
                    references.add(patch.binding)
                if isinstance(patch, ObjectPerturbationPatch):
                    references.update(patch.bindings)
        if references - bindings.keys():
            raise ValueError("All private object references must have explicit bindings.")
        if any(not key.strip() for key in bindings) or not self.id.strip():
            raise ValueError("Task and binding identifiers cannot be empty.")
        return self


class CategoryTaskDefinition(TaskDefinition):
    """Pending category-generated physical goals retain the native task contracts."""
    kind: Literal['category_robot_task']
    repair: dict = Field(default_factory=dict)
    status: Literal['pending', 'ready', 'discarded'] = 'pending'


class SceneRuntime(ClosedModel):
    reflection_sensor_profile: dict = Field(default_factory=dict)
    requires_native_visual_discovery: StrictBool = False
    post_controller_settle_steps: Annotated[StrictInt, Field(ge=0, le=5000)] = 0
    trunk_config: Text = "configs/esi_trunk.json"
    robot_camera_config: Text = "configs/esi_robot_cameras.json"
    viewer_config: Text = "configs/esi_viewer.json"
    camera_size: tuple[Annotated[StrictInt, Field(ge=64, le=2048)],
                       Annotated[StrictInt, Field(ge=64, le=2048)]] = (512, 512)
    viewer_main_view: Text = "global_diagonal"
    debug_cameras: StrictBool = True
    human_depth_visualizations: StrictBool = True
    preview_fps: Nonnegative = 2.

    @model_validator(mode="after")
    def local_config_paths(self):
        for value in (self.trunk_config, self.robot_camera_config, self.viewer_config):
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
                raise ValueError("Scene runtime configs must be repository-relative JSON files.")
        return self


class SceneQualification(ClosedModel):
    on_failure: Literal["skip"] = "skip"
    support: Text
    supported_objects: tuple[Text, ...]
    object_supports: dict[Text, Text] = Field(default_factory=dict)
    support_surface_mode: Literal["aabb_top", "local_raycast"] = "aabb_top"
    support_tolerance_m: Positive = 0.015
    maximum_linear_speed_m_s: Positive = 0.02
    maximum_angular_speed_rad_s: Positive = 0.1
    require_support_contact: StrictBool = True
    reject_robot_contacts: StrictBool = True
    screen_furniture_overlap: StrictBool = True
    settle_steps: Annotated[StrictInt, Field(ge=1)] = 30


class SceneDefinition(ClosedModel):
    """Initialization patches only; the ESI question remains the public task."""
    kind: Literal["scene_patch"] = "scene_patch"
    id: Text
    version: Annotated[StrictInt, Field(gt=0)] = 1
    status: Literal["pending", "ready", "discarded"] = "ready"
    source: Source
    target: Text | None = None
    target_bindings: tuple[Text, ...] = ()
    bindings: dict[str, Binding] = Field(default_factory=dict)
    scene: Scene
    qualification: SceneQualification | None = None
    runtime: SceneRuntime = Field(default_factory=SceneRuntime)
    seed: Annotated[StrictInt, Field(ge=0)] = 0

    @model_validator(mode="after")
    def references(self):
        if self.qualification is None:
            if self.target is not None or self.target_bindings:
                raise ValueError("A physical target requires its existing qualification contract.")
            references = set()
            for variant in self.scene.variants.values():
                for patch in variant.patches:
                    if isinstance(patch, (ObjectPosePatch, ObjectScalePatch, ObjectPhysicsPatch)):
                        references.add(patch.binding)
                    elif isinstance(patch, ObjectPerturbationPatch):
                        references.update(patch.bindings)
            if references - self.bindings.keys():
                raise ValueError("All scene object references must have explicit bindings.")
            return self
        if self.target is None:
            raise ValueError("The existing support qualification requires a target.")
        references = {self.target, *self.target_bindings, self.qualification.support, *self.qualification.supported_objects}
        references.update(self.qualification.object_supports.values())
        if self.qualification.object_supports.keys() - set(self.qualification.supported_objects):
            raise ValueError("Support mappings must refer to supported object bindings.")
        for variant in self.scene.variants.values():
            for patch in variant.patches:
                if isinstance(patch, (ObjectPosePatch, ObjectScalePatch, ObjectPhysicsPatch)):
                    references.add(patch.binding)
                elif isinstance(patch, ObjectPerturbationPatch):
                    references.update(patch.bindings)
        if references - self.bindings.keys():
            raise ValueError("All scene object references must have explicit bindings.")
        if self.target not in self.qualification.supported_objects:
            raise ValueError("The scene target must be included in supported_objects.")
        if len(set(self.qualification.supported_objects)) != len(self.qualification.supported_objects):
            raise ValueError("Supported object bindings must be distinct.")
        return self
