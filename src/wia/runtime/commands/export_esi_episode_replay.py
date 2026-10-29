#!/usr/bin/env python3
"""Export saved observation frames. Run: python src/wia/runtime/commands/export_esi_episode_replay.py RUN_DIR.

This CPU-only replay holds each archived frame until the next capture timestamp.
It never reruns the simulator or represents intermediate motion as captured video.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--playback-speed", type=float, default=20.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.playback_speed) or args.playback_speed < 1:
        parser.error("Playback speed must be finite and at least one.")
    run = args.run.expanduser().resolve()
    episode = run / "episode"
    summary = json.loads((episode / "summary.json").read_text())
    steps = [json.loads(line) for line in (episode / "steps.jsonl").read_text().splitlines()]
    paths = [episode / steps[0]["observation"]]
    paths.extend(episode / step["next_observation"] for step in steps if step.get("next_observation"))
    observations = [json.loads(path.read_text()) for path in paths]
    times = [datetime.fromisoformat(item["captured_at"]).timestamp() for item in observations]
    end = datetime.fromisoformat(summary["finished_at"]).timestamp()
    duration = max(.001, end - times[0])
    destination = run / "recordings" / f"observation_replay_{args.playback_speed:g}x"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "metadata.json").exists():
        print("Replay already exported:", destination)
        return 0
    samples = [{"snapshot_id": item["snapshot_id"], "action_step": item["action_step"],
                "captured_at": item["captured_at"], "source": str(path.relative_to(run)),
                "hold_s": max(.001, (times[index + 1] if index + 1 < len(times) else end) - times[index])}
               for index, (item, path) in enumerate(zip(observations, paths))]
    videos = []
    for view, image_name in (("head", "agentview_rgb"), ("wrist", "wrist_rgb")):
        lines = []
        for path, sample, observation in zip(paths, samples, observations):
            image = path.parent / observation["images"][image_name]["path"]
            if not image.is_file():
                raise RuntimeError(f"Missing archived camera image: {image}")
            lines.extend(["file '" + str(image).replace("'", "'\\''") + "'",
                          "duration " + str(sample["hold_s"] / args.playback_speed)])
        image = paths[-1].parent / observations[-1]["images"][image_name]["path"]
        lines.append("file '" + str(image).replace("'", "'\\''") + "'")
        listing = destination / (view + ".concat.txt")
        listing.write_text("\n".join(lines) + "\n")
        output = destination / (view + ".mp4")
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "concat", "-safe", "0", "-i", str(listing), "-vf", "fps=30",
                   "-t", str(duration / args.playback_speed), "-an", "-c:v", "libx264",
                   "-preset", "fast", "-crf", "26", "-threads", "2", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(output)]
        started = time.perf_counter()
        with (destination / (view + ".ffmpeg.log")).open("w") as log:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=log, check=True)
        reader = imageio_ffmpeg.read_frames(str(output))
        try:
            metadata = next(reader)
            next(reader)  # Verify the exported video contains a decodable frame.
        finally:
            reader.close()
        videos.append({"view": view, "filename": output.name, "size_bytes": output.stat().st_size,
                       "probe": metadata, "encoding_wall_time_s": time.perf_counter() - started})
        print("Exported", output, flush=True)
    report = {"schema": "archived_observation_replay_v1", "status": "saved",
              "exported_at_utc": datetime.now(timezone.utc).isoformat(), "run": str(run),
              "source": "saved canonical observation images and capture timestamps",
              "source_snapshot_count": len(samples), "samples": samples, "videos": videos,
              "source_duration_s": duration, "playback_speed": args.playback_speed,
              "output_fps": 30, "codec": "H.264", "recorded_continuous_video": False,
              "frame_policy": "Hold each observed frame until the next captured timestamp.",
              "simulator_reruns": 0, "model_reruns": 0}
    (destination / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved replay metadata:", destination / "metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
