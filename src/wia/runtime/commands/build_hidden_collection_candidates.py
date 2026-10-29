"""Generate pending hidden-collection candidates from explicit layout proposals.

Run: python -B src/wia/runtime/commands/build_hidden_collection_candidates.py --output DIR
Generation does not publish a candidate or certify native visibility/access.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
import json
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from wia.authoring.conversion import SourceInstance, ConversionContext
from wia.authoring.converters.hidden_object_collection import build_episode, open_shelf_objects
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition


def shelf_inventory(stations):
    supports, inventory = [], []
    count = 0
    for index, station in enumerate(stations):
        name = f"collection_shelf_{index + 1}"
        supports.extend(open_shelf_objects(name, station["position"], station["yaw"],
            width=station.get("width", .8), surface_height=station.get("height", .8)))
        rotation = Rotation.from_euler("z", station["yaw"], degrees=True)
        for item in station.get("items", []):
            category, model = item.get("asset", ("hamburger", "fyswch"))
            role = item.get("role", "target")
            count += 1
            # Asset origins differ; use measured native bottom-to-origin offsets.
            offset = {"hamburger": .0307, "hamburger_bun": .013, "club_sandwich": .0405}[category]
            local = [item.get("x", 0.), -.1, station.get("height", .8) + offset + .003]
            inventory.append({"name": f"collection_item_{count}", "category": category, "model": model,
                "role": role, "position": (np.array(station["position"]) + rotation.apply(local)).tolist(),
                "orientation": rotation.as_quat().tolist(), "support": name + "_deck"})
    return supports, inventory


def proposals():
    single = [
        {"position": [10.5, 3.25, 0.], "yaw": -20., "items": [{}]},
        {"position": [14.4, 3.1, 0.], "yaw": -90., "items": [{}]},
        {"position": [16., 1.3, 0.], "yaw": 90., "items": [{}]},
        {"position": [15.6, 2.9, 0.], "yaw": 0., "items": []},
    ]
    similar = [
        {"position": [-4.6, -.3, 0.], "yaw": -30., "width": 1.,
         "items": [{"x": -.23}, {"x": .23}]},
        {"position": [-3.3, -.65, 0.], "yaw": 0., "width": 1.,
         "items": [{"x": -.23}, {"x": .23}]},
        {"position": [-1.2, -.5, 0.], "yaw": 0., "width": 1.2,
         "items": [{"x": -.38}, {"x": 0., "role": "distractor", "asset": ("hamburger_bun", "fqxumh")},
                   {"x": .38, "role": "distractor", "asset": ("club_sandwich", "ablfyd")}]},
    ]
    # Multi-room support proposals use the open side of the dining area. Native
    # route/access validation decides whether this placement is retained.
    multi = single[:3] + [
        {"position": [5.25, 4.15, 0.], "yaw": 0., "width": 1.,
         "items": [{"x": -.23}]},
    ]
    house_source = "Enumerative Perception/Counting w Occlusion/house_single_floor/living_room_0/q_003.json"
    beech_source = "Enumerative Perception/Spatial Segmentation/Beechwood_0_int/living_room_1/q_001.json"
    rows = [
        ("single_room", house_source, "house_single_floor", ["living_room_0"], single,
         [14., 2., .02], 180., "this living room", "A spacious living room with several open shelves."),
        ("multi_room", house_source, "house_single_floor",
         ["living_room_0", "kitchen_0", "dining_room_0"], multi,
         [14., 2., .02], 180., "the main living room, kitchen and dining room",
         "Connected living, kitchen and dining areas. The distribution of targets across rooms is unknown."),
        ("similar_repeated_views", beech_source, "Beechwood_0_int", ["living_room_1"], similar,
         [-.2, 0., .02], 180., "this living room",
         "A living room with a piano, tables and open shelves. Some food items are distractors."),
    ]
    for variant, source, scene, rooms, stations, position, yaw, area, environment in rows:
        supports, inventory = shelf_inventory(stations)
        yield source, {"variant": variant, "scene_model": scene, "loaded_rooms": rooms,
            "target_count": sum(x["role"] == "target" for x in inventory),
            "public_search_area": area, "public_environment": environment,
            "robot_start": {"position": position,
                "orientation": Rotation.from_euler("z", yaw, degrees=True).as_quat().tolist()},
            "inventory": inventory, "support_objects": supports,
            "plate": {"name": "collection_plate", "category": "tray", "model": "gsxbym",
                "right_eef_position_base_m": [.4, -.25, .8], "rim_axis": "y",
                "rim_penetration_m": .025,
                "usable_bounds_local_m": [[-.147, -.21, -.0049], [.147, .21, .17]]},
            "validation_status": "basic_reset_review_pending_full_execution_unverified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = ConversionContext(args.output, seed=0)
    for relative, layout in proposals():
        source = SourceInstance(WIA_ESI_ROOT / 'dataset/json_clean' / relative, relative,
            "Enumerative Perception", "Hidden Object Search")
        result = build_episode(source, context, layout)
        task_definition(result["task_definition"])
        SceneDefinition.model_validate(result["scene_definition"])
        directory = args.output / layout["variant"]
        directory.mkdir(parents=True, exist_ok=True)
        for key, filename in (("task_definition", "task.yaml"), ("scene_definition", "scene.yaml")):
            (directory / filename).write_text(yaml.safe_dump(result[key], sort_keys=False))
        (directory / "proposal.json").write_text(json.dumps(result | {"layout": layout}, indent=2) + "\n")
        print(directory)


if __name__ == "__main__":
    main()
