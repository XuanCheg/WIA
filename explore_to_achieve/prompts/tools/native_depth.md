Only the registered read-only perception tools are authorized. They read the immutable snapshot authorized for the active decision, including deliberately frozen initial depth. None performs motion, segmentation, target judgement or private evaluation. There is no tool for changing snapshots or querying arbitrary snapshot IDs or filesystem paths.

The native transport registers `read_rgb_camera`, plus `read_depth_pixels` and `read_depth_region` when depth is available; they route only to the `depth_snapshot` MCP server described below. Use the exact function names exposed by the native tool interface.

`depth_snapshot.read_depth_pixels` takes `sensor` (`head_depth_m` or `wrist_depth_m`) and `pixels_uv` (1–256 integer [u,v] pairs within native image bounds). Origin is top-left, u rightward, v downward. Results are float32-derived optical-Z metric depths in metres. Invalid readings are null, with no resampling.

`depth_snapshot.read_depth_region` takes `sensor`, integer `u`, `v`, `width`, `height`. Width and height must be positive, the entire rectangle must lie within the native image, and its area must be at most 1024 pixels. Results are matrices ordered by v rows and u columns, using the same depth units and invalid-value rules.

Returned snapshot ID and capture time identify the authorized sensor data. A previous tool result is historical evidence and does not imply permission to query that previous snapshot now. Python alone owns bindings and robot execution. No shell, filesystem, browser, simulator, robot-motion, private evaluator or multi-agent tools are authorized.

`read_rgb_camera` takes `camera`: `chassis_front_left_rgb`, `chassis_front_right_rgb`, `chassis_left_rgb` or `chassis_right_rgb`. The MCP server returns one unmodified PNG from the current authorized sensor snapshot and its snapshot identity. These four views are attached automatically on the first turn and are optional afterward. Head, both wrists and rear chassis RGB remain attached on subsequent turns. Request an optional view whenever useful before returning the next decision. This tool does not advance the simulation or capture a newer image.
