#!/usr/bin/env python3
"""Render native Isaac RGB primitives to diagnose one assigned capture GPU.

Run in the simulation environment with explicit CUDA visibility and Vulkan ICD:
python src/wia/runtime/commands/check_esi_gpu_rendering.py --gpu-info GPU.json --output-dir OUTPUT
This uses installed Isaac Sim without ESI assets, robot actions or qualification.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.runtime.scene_capture_queue import write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-info", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    gpu = json.loads(args.gpu_info.read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    app = None
    stage_name = "native_initialization"
    write_json(output / "process.json", {"pid": os.getpid(), "pgid": os.getpgrp(), "gpu": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vulkan_icd": os.environ.get("VK_ICD_FILENAMES"), "started_at": time.time()})
    try:
        from wia.runtime.scene_capture_adapters import install_gpu_selection

        install_gpu_selection(gpu)
        from isaacsim import SimulationApp

        cache = Path("/tmp/show-harness-zhouxinyang/render_probe") / gpu["uuid"]
        cache.mkdir(parents=True, exist_ok=True)
        app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
            "width": 512, "height": 512, "multi_gpu": False,
            "extra_args": ["--portable", f"--portable-root={cache}",
                "--/plugins/carb.tasking.plugin/threadCount=8",
                "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8",
                "--/exts/omni.physx/numThreads=4"]})
        stage_name = "primitive_scene"
        import carb
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        import torch
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdLux

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        for path, position, scale, color in (
            ("/World/Ground", (0., 0., -.05), (5., 5., .1), (.23, .23, .23)),
            ("/World/RedCube", (-.5, 0., .4), (.8, .8, .8), (.8, .06, .03)),
        ):
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.)
            cube.AddTranslateOp().Set(Gf.Vec3d(*position))
            cube.AddScaleOp().Set(Gf.Vec3f(*scale))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        sphere = UsdGeom.Sphere.Define(stage, "/World/BlueSphere")
        sphere.CreateRadiusAttr(.35)
        sphere.AddTranslateOp().Set(Gf.Vec3d(.5, 0., .35))
        sphere.CreateDisplayColorAttr([Gf.Vec3f(.03, .15, .8)])
        light = UsdLux.DomeLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(1000.)
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateFocalLengthAttr(24.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(2.5, 2.5, 2.), Gf.Vec3d(0., 0., .3), Gf.Vec3d(0., 0., 1.)).GetInverse())
        product = rep.create.render_product(str(camera.GetPath()), (512, 512))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([product])
        omni.timeline.get_timeline_interface().play()
        stage_name = "rgb_capture"
        pixels = np.empty((0,), dtype=np.uint8)
        for _ in range(12):
            rep.orchestrator.step(rt_subframes=4)
            pixels = np.asarray(rgb.get_data())
            if pixels.shape == (512, 512, 4) and float(pixels[..., :3].std()) > 10:
                break
        if pixels.shape != (512, 512, 4):
            raise RuntimeError(f"Native RGB has invalid dimensions: {pixels.shape}")
        colors = pixels[..., :3].astype(float)
        red = (colors[..., 0] > 1.5 * colors[..., 1]) & (colors[..., 0] > 1.5 * colors[..., 2])
        blue = (colors[..., 2] > 1.5 * colors[..., 0]) & (colors[..., 2] > 1.5 * colors[..., 1])
        if colors.std() <= 10 or red.sum() < 100 or blue.sum() < 100:
            raise RuntimeError("Native RGB does not contain both expected colored primitives.")
        Image.fromarray(pixels).save(output / "native_rgb.png")
        settings = carb.settings.get_settings()
        if settings.get("/renderer/activeGpu") != gpu["renderer_index"] or settings.get("/renderer/multiGpu/enabled"):
            raise RuntimeError("Native renderer changed the assigned single-GPU configuration.")
        if torch.cuda.device_count() != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu["uuid"]:
            raise RuntimeError("CUDA visibility does not match the assigned GPU.")
        torch.empty(1, device="cuda:0")
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits"], text=True)
        used = {line.split(",")[1].strip() for line in processes.splitlines()
                if line.split(",")[0].strip() == str(os.getpid())}
        if used != {gpu["uuid"]}:
            raise RuntimeError(f"Native process GPU UUID mismatch: {used}")
        receipt = {"schema": "esi_native_render_probe_v1", "success": True,
            "gpu": gpu, "pid": os.getpid(), "verified_process_gpu_uuids": sorted(used),
            "renderer_active_gpu": settings.get("/renderer/activeGpu"), "renderer": "RaytracedLighting",
            "image": str(output / "native_rgb.png"), "image_shape": list(pixels.shape),
            "rgb_std": float(colors.std()), "red_pixels": int(red.sum()), "blue_pixels": int(blue.sum()),
            "elapsed_s": time.monotonic() - started, "model_calls": 0, "robot_actions": 0,
            "esi_scene_loaded": False, "physical_qualification_run": False}
        write_json(output / "render_receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
        return 0
    except Exception as error:
        write_json(output / "render_failure.json", {"success": False, "gpu": gpu,
            "error": str(error), "error_type": type(error).__name__, "stage": stage_name,
            "elapsed_s": time.monotonic() - started})
        traceback.print_exc()
        return 1
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
