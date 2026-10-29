"""Private diagnostics of authorized sensor frames. Run preflight_esi_tasks.py --mode full."""
from __future__ import annotations

import numpy as np

from wia.runtime.preflight.models import Check


class PerceptionProbe:
    """Keep oracle segmentation inside the worker and outside public observations."""

    def __init__(self, context, session, output, emit):
        self.context, self.session, self.output, self.emit = context, session, output, emit
        self.observed = {}
        self.config_before = session.agent_observation_config.to_dict()
        self.modalities_before = {key: set(sensor.modalities) for key, sensor in session.robot.sensors.items()}

    def inspect(self, phase="initial", keys=None):
        from wia.native.esi_visibility import PrivateVisibilityCheck

        observation = self.session.observation_source.agent_observation()
        rows = {}
        for name, rgb in observation.rgb_images().items():
            if rgb is None:
                continue
            pixels = np.asarray(rgb)
            valid = pixels.ndim == 3 and pixels.shape[-1] == 3 and pixels.size > 0 and np.isfinite(pixels).all()
            varying = valid and float(np.std(pixels)) > .5
            rows[name] = {"shape": list(pixels.shape), "valid": bool(valid),
                          "standard_deviation": float(np.std(pixels)) if pixels.size else None}
            self.emit(Check("perception", "rgb." + name, "FAIL" if not valid else "PASS" if varying else "INCONCLUSIVE",
                "Authorized RGB frame is valid and nonconstant." if varying else "Authorized RGB frame is invalid or visually degenerate.",
                evidence=rows[name], verification_level="native_initialization"))
        if not rows:
            self.emit(Check("perception", "rgb_availability", "INCONCLUSIVE", "No RGB input is enabled and available in the Agent observation."))
        for name, enabled in (("head_depth_m", self.config_before["head_depth"]), ("wrist_depth_m", self.config_before["wrist_depth"])):
            if not enabled:
                continue
            depth = getattr(observation.sensors, name)
            if depth is None:
                self.emit(Check("perception", "depth." + name, "INCONCLUSIVE", "Enabled depth input is unavailable.", verification_level="native_initialization"))
                continue
            values = np.asarray(depth)
            valid = values.ndim == 2 and values.size > 0 and not np.isnan(values).any() and not (values < 0).any()
            finite = np.isfinite(values) & (values > 0)
            useful = valid and bool(finite.any())
            self.emit(Check("perception", "depth." + name, "FAIL" if not valid else "PASS" if useful else "INCONCLUSIVE",
                "Authorized metric depth has valid finite returns." if useful else "Depth contains invalid samples or no finite returns.",
                evidence={"shape": list(values.shape), "finite_positive_fraction": float(finite.mean())},
                verification_level="native_initialization"))
        config = self.context.scene.definition.qualification
        variant = self.context.scene.definition.scene.variants[self.context.loaded.variant]
        condition = variant.initial_visibility or getattr(config, "initial_visibility", "none")
        keys = tuple(keys) if keys is not None else tuple(dict.fromkeys(self.context.targets + self.context.containers))
        for key in keys:
            name = self.context.bindings[key].name
            if self.session.env.scene.object_registry("name", name) is None:
                continue
            checker = PrivateVisibilityCheck(self.session, {
                "scene": self.session.scene_name, "room": self.session.room_name, "target": name,
                "camera_scope": "enabled_agent", "initial_condition": "visible",
                "minimum_pixels": getattr(config, "minimum_pixels", 64),
                "minimum_extent_px": getattr(config, "minimum_extent_px", 8)}, self.output / key)
            record = checker.record(phase=phase, observation=observation)
            rows = self.observed.setdefault(key, [])
            rows.append({"phase": phase, "views": record["views"], "visible": record["any_recognizably_visible"]})
            visible = any(row["visible"] for row in rows)
            self.emit(Check("perception", "observability." + key, "PASS" if visible else "INCONCLUSIVE",
                "Required object was distinguishable by the private pixel gate in authorized RGB." if visible
                else "Object has not been observed from sampled states; occlusion does not prove unobservability.",
                [key], {"observations": list(rows), "scope": "pixel evidence, not semantic recognition"}, verification_level="native_initialization"))
            if phase == "initial" and key in self.context.targets and condition != "none":
                accepted = (record["any_recognizably_visible"] if condition == "visible" else
                            record["all_views_zero_pixels"] if condition == "hidden_zero" else not record["any_recognizably_visible"])
                self.emit(Check("perception", "initial_visibility." + key, "PASS" if accepted else "FAIL",
                    "Declared initial visibility constraint holds." if accepted else "Declared initial visibility constraint is violated.",
                    [key], {"condition": condition, "views": record["views"]},
                    None if accepted else "inspect_initial_occlusion", "native_initialization"))
        restored = self.session.agent_observation_config.to_dict() == self.config_before and all(
            set(self.session.robot.sensors[key].modalities) == modalities for key, modalities in self.modalities_before.items())
        self.emit(Check("perception", "observation_contract", "PASS" if restored else "FAIL",
            "Authorized observation switches and sensor modalities are unchanged after private segmentation." if restored
            else "Private diagnostic modalities or observation switches leaked into the observation contract.",
            evidence={"config": self.config_before, "private_reports_only": True}, verification_level="native_initialization"))
        private_keys = {"evaluator_state", "task_success", "answer_labels", "oracle_visibility", "object_poses", "hidden_relationships"}
        def leaked_keys(value):
            if isinstance(value, dict):
                return (set(value) & private_keys) | set().union(*(leaked_keys(item) for item in value.values()))
            if isinstance(value, (list, tuple)):
                return set().union(*(leaked_keys(item) for item in value))
            return set()
        leaks = sorted(leaked_keys(observation.metadata()))
        self.emit(Check("perception", "privileged_metadata_leak", "FAIL" if leaks else "PASS",
            "Privileged evaluator fields appear in the Agent observation." if leaks else
            "No privileged evaluator fields occur in the typed Agent observation metadata.",
            evidence={"leaked_keys": leaks, "scope": "structural field audit; no semantic visibility claims"},
            verification_level="native_initialization"))
