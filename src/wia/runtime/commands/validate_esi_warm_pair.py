#!/usr/bin/env python3
"""Measure one original A-to-B warm replacement against an independent cold B.

Run only with Main's explicit shared-GPU ownership metadata and immutable code:
python src/wia/runtime/commands/validate_esi_warm_pair.py --mode warm --source-a A.json \
    --source-b B.json --output-dir RUN/warm --gpu-info GPU.json --seed 0 --retain
Run --mode cold with the same sources, GPU/camera/physics settings and a separate
output directory. A/B are restaurant_asian corridor_1 original samples 0 and 3.
Set OMNIGIBSON_APPDATA_PATH to a private cache; otherwise RUN/appdata is used.
Successful retained runs and failures with an existing world stay resident using
renders only. This diagnostic performs no robot episode, qualification or approval.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from wia.runtime.scene_capture_queue import write_json

PAIR_NAMES = ("restaurant_asian__corridor_1_0.json", "restaurant_asian__corridor_1_3.json")
REVIEW_VIEWS = {"head", "scene", "global_overhead"}


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision(root):
    marker = root / ".capture_revision.json"
    if marker.is_file():
        return json.loads(marker.read_text())["commit"]
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True, stderr=subprocess.PIPE,
    ).strip()


def seed_rng(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_sources(sources):
    for path, name in zip(sources, PAIR_NAMES):
        raw = json.loads(path.read_text())
        identity = tuple(raw.get(key) for key in ("big_task", "small_task", "scene", "room"))
        if path.name != name or identity != (
            "Perceptual Grounding", "Partial Occlusion", "restaurant_asian", "corridor_1",
        ):
            raise ValueError("Use only the two selected original restaurant corridor sources.")


def capture(session, gpu, directory):
    from wia.runtime.commands.m3_pair_evidence import capture_review, collect_evidence

    directory.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    images = capture_review(session, directory)
    capture_s = time.monotonic() - began
    if set(images) != REVIEW_VIEWS or not all(Path(path).is_file() for path in images.values()):
        raise RuntimeError("Review capture must produce the three requested diagnostic images.")
    evidence = collect_evidence(session, gpu)
    write_json(directory / "evidence.json", evidence)
    return evidence, images, capture_s


def failure(output, mode, stage, error):
    path = output / "failure.json"
    record = {
        "mode": mode, "stage": stage, "native_pid": os.getpid(),
        "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
        "timestamp": time.time(),
    }
    print(record["error"], file=sys.stderr, flush=True)
    try:
        if path.is_file():
            previous = json.loads(path.read_text())
            previous["retention_error"] = record
            record = previous
        write_json(path, record)
    except Exception as log_error:
        print("Failure receipt could not be published: " + str(log_error), file=sys.stderr, flush=True)


def retain(session, output, mode, state):
    completed_retries = 0
    while True:
        request = output / "retry_request.json"
        if request.is_file():
            claimed = output / f"retry_request_{time.time_ns()}.json"
            request.replace(claimed)
            try:
                instruction = json.loads(claimed.read_text())
                if mode != "warm" or instruction["operation"] != "repeat_pair":
                    raise ValueError("Only the same original A-to-B pair can be repeated.")
                code = Path(instruction["code_root"]).resolve()
                if not code.is_relative_to(output.parent / "revisions"):
                    raise ValueError("A retry requires this diagnostic's immutable archive.")
                if revision(code) != instruction["code_commit"]:
                    raise ValueError("Retry archive revision does not match its request.")
                spec = importlib.util.spec_from_file_location(
                    "m3_pair_retry_preview", code / "src/wia/runtime/scene_warm_preview.py")
                preview = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(preview)
                original = json.loads((output / "result.json").read_text())
                sources = tuple(Path(original["sources"][label]) for label in ("A", "B"))
                validate_sources(sources)
                if any(source_hash(path) != original["source_sha256"][label]
                       for label, path in zip(("A", "B"), sources)):
                    raise ValueError("An original pair source changed before retry.")
                destination = output / ("retry_" + claimed.stem.removeprefix("retry_request_"))
                seed_rng(0)
                preview.switch_original_pair(session, sources[0])
                capture(session, original["gpu"], destination / "initial_A")
                seed_rng(0)
                swap = preview.switch_original_pair(session, sources[1])
                _, images, capture_s = capture(session, original["gpu"], destination / "final_B")
                write_json(destination / "result.json", {
                    "native_pid": os.getpid(), "code_commit": instruction["code_commit"],
                    "original_runtime_commit": revision(ROOT), "swap": swap,
                    "capture_s": capture_s, "images": images,
                    "source_sha256": original["source_sha256"], "seed": 0,
                    "qualification_run": False, "robot_actions": 0, "model_calls": 0,
                })
                completed_retries += 1
                state = "RETAINED"
            except Exception as error:
                state = "FAILED_RETAINED"
                failure(output, mode, "same_pair_retry", error)
        try:
            write_json(output / "retained_world.json", {
                "pid": os.getpid(), "heartbeat": time.time(), "state": state, "mode": mode,
                "source": str(session.metadata_path), "physics_steps_during_passive_retention": 0,
                "completed_pair_retries": completed_retries,
            })
        except Exception as error:
            state = "FAILED_RETAINED"
            failure(output, mode, "retained_heartbeat", error)
        try:
            session.pipeline.og.sim.render()
        except Exception as error:
            state = "FAILED_RETAINED"
            failure(output, mode, "retained_render", error)
        time.sleep(2.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("warm", "cold"), required=True)
    parser.add_argument("--source-a", type=Path, required=True)
    parser.add_argument("--source-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-info", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    parser.add_argument("--retain", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("result.json", "failure.json", "retained_world.json")):
        parser.error("Use a fresh output directory to preserve previous diagnostics.")
    sources = tuple(path.expanduser().resolve() for path in (args.source_a, args.source_b))
    session = None
    stage = "input_validation"
    try:
        validate_sources(sources)
        hashes = {label: source_hash(path) for label, path in zip(("A", "B"), sources)}
        gpu = json.loads(args.gpu_info.expanduser().resolve().read_text())
        appdata = Path(os.environ.get("OMNIGIBSON_APPDATA_PATH")
                       or gpu.get("appdata_path") or output / "appdata").resolve()
        appdata.mkdir(parents=True, exist_ok=True)
        os.environ.update(
            CUDA_VISIBLE_DEVICES=gpu["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
            OMNIGIBSON_GPU_ID="0", OMNIGIBSON_HEADLESS="1",
            OMNIGIBSON_APPDATA_PATH=str(appdata), XDG_CACHE_HOME=str(appdata / "xdg_cache"),
            CUDA_CACHE_PATH=str(appdata / "cuda_cache"),
            __GL_SHADER_DISK_CACHE_PATH=str(appdata / "nvidia_shader_cache"),
        )
        stage = "gpu_bootstrap"
        from wia.runtime.scene_capture_adapters import install_gpu_selection, isolate_texture_cache

        install_gpu_selection(gpu)
        from omnigibson import simulator

        isolate_texture_cache(simulator, os.environ["OMNIGIBSON_APPDATA_PATH"])
        from wia.runtime.scene_warm_preview import (
            compatibility, remember_initial_background, switch_original_pair, warm_contract,
        )
        from wia.runtime.sim.esi_omnigibson import ESISession

        initial_source = sources[0] if args.mode == "warm" else sources[1]
        session = ESISession(initial_source, camera_size=(512, 512), require_robot=True)
        session._initialization_observer = remember_initial_background
        stage = "native_initialization"
        seed_rng(args.seed)
        began = time.monotonic()
        session.reset()
        initialization_s = time.monotonic() - began

        stage = "compatibility"
        previous_contract = warm_contract(session, sources[0])
        next_contract = warm_contract(session, sources[1])
        decision = compatibility(previous_contract, next_contract)
        incompatible_contract = deepcopy(next_contract)
        incompatible_contract["scene_model"] += "__incompatible_probe"
        rejection = compatibility(previous_contract, incompatible_contract)
        if rejection["compatible"]:
            raise RuntimeError("The CPU compatibility check accepted a changed scene model.")
        write_json(output / "compatibility.json", {
            "previous": previous_contract, "next": next_contract,
            "source_pair": decision, "changed_scene_model_rejection": rejection,
        })

        images = {}
        captures = {}
        swap = {"background_reused": False, "swap_s": 0.0}
        if args.mode == "warm":
            stage = "initial_A_capture"
            _, images["initial_A"], captures["initial_A_s"] = capture(
                session, gpu, output / "initial_A",
            )
            if not decision["compatible"]:
                raise RuntimeError("Warm replacement rejected: " + decision["reason"])
            stage = "original_pair_replacement"
            seed_rng(args.seed)
            swap = switch_original_pair(session, sources[1])
            if not swap["background_reused"]:
                raise RuntimeError("The warm replacement did not preserve its existing background.")

        stage = "final_B_capture"
        evidence, images["final_B"], captures["final_B_s"] = capture(
            session, gpu, output / "final_B",
        )
        if evidence["source_sha256"] != hashes["B"]:
            raise RuntimeError("The final evidence does not identify the selected original source B.")
        if any(source_hash(path) != hashes[label] for label, path in zip(("A", "B"), sources)):
            raise RuntimeError("An original source changed during the diagnostic.")
        stage = "result_publication"
        result = {
            "mode": args.mode, "seed": args.seed,
            "sources": {label: str(path) for label, path in zip(("A", "B"), sources)},
            "source_sha256": hashes, "code_commit": revision(ROOT),
            "esi_commit": revision(WIA_ESI_ROOT), "native_pid": os.getpid(),
            "gpu": evidence["gpu"], "camera_size": [512, 512], "require_robot": True,
            "initialization_s": initialization_s, "swap_s": swap["swap_s"],
            "capture_s": sum(captures.values()), "capture_timings": captures,
            "background_reused": swap["background_reused"], "swap": swap,
            "compatibility": decision, "incompatible_contract_rejection": rejection,
            "qualification_run": False, "physical_grasp_verified": False,
            "model_calls": 0, "robot_actions": 0, "images": images,
            "retain_requested": args.retain,
        }
        write_json(output / "result.json", result)
        if args.retain:
            retain(session, output, args.mode, "RETAINED")
        session.close()
        return 0
    except Exception as error:
        failure(output, args.mode, stage, error)
        if session is not None and session.env is not None:
            retain(session, output, args.mode, "FAILED_RETAINED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
