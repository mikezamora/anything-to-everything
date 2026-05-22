"""Frame-sequence -> MP4 render driver.

`render_layer(layer, frames, out_dir, quality)` picks the matching Manim
`Scene` for `layer`, renders it at the requested quality, and returns the
`Path` to the produced `.mp4`.

`manim` is imported lazily inside `render_layer`, so this module is importable
even when manim is absent (the test suite `importorskip`s before exercising a
real render).
"""

from __future__ import annotations

from pathlib import Path

from .scenes import scene_class_for

# Manim quality presets: (resolution, frame_rate). "low" keeps renders fast.
_QUALITY: dict[str, tuple[int, int, int]] = {
    "low": (480, 854, 15),
    "medium": (720, 1280, 30),
    "high": (1080, 1920, 60),
}


def render_layer(layer: str, frames: list, out_dir, quality: str = "low") -> Path:
    """Render the recorded `frames` for `layer` to an MP4 in `out_dir`.

    Args:
        layer: layer key (e.g. ``"qpcn"``). Falls back to a generic scene.
        frames: recorded frame sequence (``Frame`` objects or plain dicts).
        out_dir: directory the MP4 is written into (created if absent).
        quality: ``"low"`` | ``"medium"`` | ``"high"`` -- controls
            resolution/fps. ``"low"`` is fastest and used by tests.

    Returns:
        Path to the produced ``.mp4`` file.

    Raises:
        ImportError: if `manim` is not installed.
    """
    import manim as M

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    height, width, fps = _QUALITY.get(quality, _QUALITY["low"])
    scene_cls = scene_class_for(layer)
    out_name = f"{layer}_export"

    with M.tempconfig({
        "pixel_height": height,
        "pixel_width": width,
        "frame_rate": fps,
        "media_dir": str(out_dir),
        "video_dir": str(out_dir),
        "output_file": out_name,
        "disable_caching": True,
        "verbosity": "ERROR",
        "format": "mp4",
    }):
        scene = scene_cls()
        # Inject the recorded data the scene's `construct` consumes.
        scene.frames = list(frames)
        scene.render()
        produced = Path(scene.renderer.file_writer.movie_file_path)

    # Normalise the result next to `out_dir` with a predictable name so the
    # server can serve it directly regardless of Manim's nested media layout.
    final = out_dir / f"{out_name}.mp4"
    if produced.resolve() != final.resolve():
        final.write_bytes(produced.read_bytes())
    return final
