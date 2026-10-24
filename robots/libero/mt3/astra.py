# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Object identification for MT3 perception by a vision-language model through Codex.

One ``codex exec`` call shows the model the agentview and wrist images (with a
labelled pixel grid) and asks, per object category, for a pixel on every
visible instance in each image. Only identification comes from the model; the
pose is then found from the depth around those pixels (SAM3 point masks and
registration, see ``perception.py``).
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Image names shown to the model, by camera.
VIEW_NAMES = {"agentview": "front", "robot0_eye_in_hand": "wrist"}
#: Codex features that only add instructions or tool definitions to each request.
DISABLED_FEATURES = (
    "multi_agent",
    "plugins",
    "apps",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "image_generation",
    "goals",
    "shell_tool",
    "unified_exec",
    "view_image",
    "sleep_tool",
    "skill_search",
    "tool_suggest",
    "hooks",
    "remote_plugin",
    "workspace_dependencies",
    "in_app_browser",
    "code_mode_host",
    "shell_snapshot",
    "guardian_approval",
    "mentions_v2",
    "incremental_tools",
)
INSTRUCTIONS = (
    "You locate objects in robot camera images. Answer with the requested JSON only.\n"
)


def _codex_home() -> str:
    """A Codex home holding only the login, so no user instructions, skills or
    plugins are added to the request."""
    home = Path(
        os.environ.get("RPENT_MT3_CODEX_HOME", "~/.config/rpent/codex_min_home")
    ).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    auth = home / "auth.json"
    if not auth.exists():
        auth.symlink_to(Path("~/.codex/auth.json").expanduser())
    return str(home)


def _codex_binary() -> str:
    path = os.environ.get("RPENT_MT3_CODEX", "")
    if path:
        return path
    found = sorted(
        glob.glob(
            os.path.expanduser(
                "~/.vscode-server/extensions/openai.chatgpt-*/bin/linux-x86_64/codex"
            )
        )
    )
    return found[-1] if found else "codex"


def grid_image(rgb: np.ndarray, scale: int = 1, step: int = 64) -> np.ndarray:
    """``rgb`` upscaled with yellow grid lines labelled in original pixels."""
    from PIL import Image, ImageDraw

    h, w = rgb.shape[:2]
    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize(
        (w * scale, h * scale), Image.BICUBIC
    )
    draw = ImageDraw.Draw(im)
    for k in range(0, max(h, w), step):
        draw.line([(k * scale, 0), (k * scale, h * scale)], fill=(255, 255, 0))
        draw.line([(0, k * scale), (w * scale, k * scale)], fill=(255, 255, 0))
        draw.text((k * scale + 2, 2), str(k), fill=(255, 255, 0))
        draw.text((2, k * scale + 2), str(k), fill=(255, 255, 0))
    return np.asarray(im)


def _usage_limit_delay(stderr: str, now: float | None = None) -> float | None:
    """Seconds until the reset time of a Codex usage-limit error ("... try again
    at 2:57 PM"), plus a margin; None for other errors."""
    import datetime

    if "usage limit" not in stderr:
        return None
    match = re.search(r"try again at (\d{1,2}):(\d{2})\s*([AP]M)", stderr)
    if not match:
        return 600.0
    hour, minute = int(match.group(1)) % 12, int(match.group(2))
    hour += 12 if match.group(3) == "PM" else 0
    current = datetime.datetime.fromtimestamp(time.time() if now is None else now)
    reset = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset <= current:
        reset += datetime.timedelta(days=1)
    return (reset - current).total_seconds() + 30.0


def _parse(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON in answer: {text[:200]!r}")
    return json.loads(match.group(0))


class AstraLocator:
    """Pixels of object instances from one model call per query."""

    def __init__(
        self,
        model: str = "gpt-6-astra",
        effort: str = "low",
        timeout: float = 300.0,
        task: str = "",
        scale: int = 1,
        limit_retries: int = 3,
        max_wait: float = 2700.0,
    ):
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.task = task
        #: Upscaling of the gridded images sent to the model.
        self.scale = scale
        #: On the account's usage limit, wait until its reset time and retry,
        #: at most ``limit_retries`` times and ``max_wait`` seconds in total.
        self.limit_retries = limit_retries
        self.max_wait = max_wait
        #: One entry per call: categories, answer, tokens, seconds.
        self.calls: list[dict[str, Any]] = []

    def prompt(self, size: int, cameras: list[str], categories: dict[str, str]) -> str:
        views = ", ".join(
            f'image {i + 1} = "{VIEW_NAMES.get(c, c)}"'
            + (
                " (fixed camera in front of the table)"
                if c == "agentview"
                else " (camera on the gripper, looking down)"
            )
            for i, c in enumerate(cameras)
        )
        objects = "\n".join(f'- "{k}": {v}' for k, v in categories.items())
        shape = ", ".join(
            f'"{VIEW_NAMES.get(c, c)}": [[row, col], ...]' for c in cameras
        )
        return (
            f"Robot camera images of a tabletop scene: {views}. Yellow grid lines are "
            f"labelled with pixel coordinates of a {size}x{size} image (row from the top, "
            f"col from the left).\n"
            + (f'The robot\'s task: "{self.task}".\n' if self.task else "")
            + "For each object below, give a pixel (in the "
            f"{size}x{size} frame) on the visible surface of EVERY instance of it in EACH "
            "image, near the middle of the visible part. Use an empty list if it is not "
            "visible in an image. Do not guess hidden objects.\n"
            f"Objects:\n{objects}\n"
            "Reply with JSON only, keys exactly as given: "
            f'{{"<object>": {{{shape}}}, ...}}'
        )

    def locate(
        self, images: dict[str, np.ndarray], categories: dict[str, str]
    ) -> dict[str, dict[str, list[list[int]]]]:
        """``{category: {camera: [[row, col], ...]}}`` in the input image frame."""
        from PIL import Image

        cameras = list(images)
        size = next(iter(images.values())).shape[0]
        tic = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="mt3_astra_") as tmp:
            Path(tmp, "instructions.txt").write_text(INSTRUCTIONS)
            args = [
                _codex_binary(),
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-rules",
                "--json",
                "-s",
                "read-only",
                "-m",
                self.model,
            ]
            for key, value in (
                ("model_reasoning_effort", self.effort),
                ("model_instructions_file", str(Path(tmp, "instructions.txt"))),
                ("include_environment_context", "false"),
                ("project_doc_max_bytes", "0"),
                ("web_search", '"disabled"'),
                ("include_apply_patch_tool", "false"),
                ("include_plan_tool", "false"),
            ):
                args += ["-c", f"{key}={value}"]
            for feature in DISABLED_FEATURES:
                args += ["--disable", feature]
            args += ["-o", str(Path(tmp, "answer.txt"))]
            for cam in cameras:
                path = Path(tmp, f"{VIEW_NAMES.get(cam, cam)}.png")
                Image.fromarray(grid_image(images[cam], scale=self.scale)).save(path)
                args += ["-i", str(path)]
            args += ["--", self.prompt(size, cameras, categories)]
            waited = 0.0
            for _ in range(self.limit_retries + 1):
                proc = subprocess.run(
                    args,
                    cwd=tmp,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env={**os.environ, "CODEX_HOME": _codex_home()},
                )
                delay = (
                    _usage_limit_delay(proc.stderr + proc.stdout)
                    if proc.returncode
                    else None
                )
                if delay is None or waited + delay > self.max_wait:
                    break
                logger.warning("codex usage limit; retrying in %.0f s", delay)
                time.sleep(delay)
                waited += delay
            answer_path = Path(tmp, "answer.txt")
            answer = answer_path.read_text() if answer_path.exists() else ""
        usage, error = None, ""
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("usage"):
                usage = event["usage"]
            if isinstance(event, dict) and event.get("type") == "error":
                error = str(event.get("message", ""))
        record: dict[str, Any] = {
            "categories": list(categories),
            "seconds": round(time.monotonic() - tic, 1),
            "usage": usage,
            "returncode": proc.returncode,
            "waited_s": round(waited),
        }
        if proc.returncode:
            # With --json the model error (e.g. the usage limit) is an event on stdout.
            record["stderr"] = (error or proc.stderr)[-300:]
        try:
            raw = _parse(answer)
        except (ValueError, json.JSONDecodeError) as exc:
            record["error"] = str(exc)[:200]
            self.calls.append(record)
            logger.warning("astra locate failed: %s", exc)
            return {}
        names = {VIEW_NAMES.get(c, c): c for c in cameras}
        out: dict[str, dict[str, list[list[int]]]] = {}
        for key in categories:
            per_view = raw.get(key) or {}
            out[key] = {}
            for view, cam in names.items():
                pts = per_view.get(view) or []
                if pts and isinstance(pts[0], (int, float)):
                    pts = [pts]
                out[key][cam] = [
                    [int(p[0]), int(p[1])]
                    for p in pts
                    if isinstance(p, (list, tuple))
                    and len(p) == 2
                    and 0 <= p[0] < size
                    and 0 <= p[1] < size
                ]
        record["answer"] = out
        self.calls.append(record)
        return out


def instances(
    pixels: dict[str, list[list[int]]],
    worlds: dict[str, np.ndarray],
    merge: float = 0.08,
) -> list[dict[str, Any]]:
    """Group a category's pixels (all cameras) into instances by their world points.

    A pixel joins the nearest instance not yet seen by its camera whose first
    point lies within ``merge`` [m] in the table plane (cameras see different
    faces: a side from the front, the top from the wrist), else starts one.
    Returns ``[{"xyz": centroid, "pixels": {camera: [r, c]}}]``.
    """
    found: list[dict[str, Any]] = []
    for cam, pts in pixels.items():
        for r, c in pts:
            xyz = np.asarray(worlds[cam][r, c], dtype=float)
            if not np.isfinite(xyz).all():
                continue
            dist = [
                np.linalg.norm(inst["points"][0][:2] - xyz[:2])
                if cam not in inst["pixels"]
                else np.inf
                for inst in found
            ]
            if dist and min(dist) < merge:
                inst = found[int(np.argmin(dist))]
                inst["pixels"][cam] = [r, c]
                inst["points"].append(xyz)
            else:
                found.append({"pixels": {cam: [r, c]}, "points": [xyz]})
    for inst in found:
        inst["xyz"] = np.mean(inst.pop("points"), axis=0)
    return found


def assign(
    bodies: list[str], found: list[dict[str, Any]], expect: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Instances for ``bodies`` of one category, matched to their expected positions
    (table plane) by minimum total distance; bodies left over get none."""
    from scipy.optimize import linear_sum_assignment

    if not found:
        return {}
    cost = np.array(
        [
            [
                np.linalg.norm(inst["xyz"][:2] - np.asarray(expect[b][:2]))
                if expect.get(b) is not None
                else 0.0
                for inst in found
            ]
            for b in bodies
        ]
    )
    rows, cols = linear_sum_assignment(cost)
    return {bodies[r]: found[c] for r, c in zip(rows, cols)}
