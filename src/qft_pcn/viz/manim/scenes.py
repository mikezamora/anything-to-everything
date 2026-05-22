"""Manim `Scene` subclasses, one per QFT-PCN layer.

`manim` is imported LAZILY: the scene classes are built inside
`build_scene_classes()`, which is only called once a render is actually
requested. This keeps `import scenes` (and transitively the FastAPI server)
working when `manim` is not installed.

Each scene is constructed with a recorded frame sequence (`frames`, a list of
`Frame` objects or plain dicts) and animates the recorded data on `construct`.
"""

from __future__ import annotations

from functools import lru_cache


def _as_dict(frame) -> dict:
    """Normalise a `Frame` object or plain dict to a `{step, layer_states}`."""
    if hasattr(frame, "to_dict"):
        return frame.to_dict()
    if isinstance(frame, dict):
        return frame
    raise TypeError(f"unsupported frame type: {type(frame)!r}")


def _layer_values(frames, layer: str, key: str) -> list[float]:
    """Extract a numeric series `layer_states[layer][key]` across frames.

    Missing entries are filled with the previous value (or 0.0) so the curve
    stays continuous even if a layer is absent from some frame.
    """
    series: list[float] = []
    last = 0.0
    for f in frames:
        d = _as_dict(f)
        state = d.get("layer_states", {}).get(layer, {})
        val = state.get(key, last)
        try:
            last = float(val)
        except (TypeError, ValueError):
            last = last
        series.append(last)
    return series


@lru_cache(maxsize=1)
def build_scene_classes() -> dict:
    """Import manim and build the per-layer `Scene` subclasses.

    Returns a mapping ``{layer_name: SceneClass}`` plus a ``"_generic"`` entry
    used as a fallback for layers without a dedicated scene. Raises
    `ImportError` (via the `import manim`) when manim is not installed.
    """
    import manim as M

    class LayerScene(M.Scene):
        """Base scene: animates one recorded numeric series as a line plot.

        Subclasses set `layer`, `value_key`, `title` and `value_label`.
        `frames` is injected by `render.py` before `render()` is called.
        """

        layer: str = "generic"
        value_key: str = "value"
        title: str = "QFT-PCN Layer"
        value_label: str = "value"

        # Populated by render.py via attribute injection on the instance.
        frames: list = []

        def _series(self) -> list[float]:
            return _layer_values(self.frames, self.layer, self.value_key)

        def construct(self) -> None:  # noqa: D102
            series = self._series()
            title = M.Text(self.title, font_size=36).to_edge(M.UP)
            self.play(M.Write(title), run_time=0.5)

            if not series:
                self.play(M.Write(
                    M.Text("no frames", font_size=28)), run_time=0.5)
                self.wait(0.5)
                return

            lo, hi = min(series), max(series)
            if hi - lo < 1e-9:
                hi = lo + 1.0
            n = len(series)

            axes = M.Axes(
                x_range=[0, max(n - 1, 1), max(1, (n - 1) // 4 or 1)],
                y_range=[lo, hi, (hi - lo) / 4],
                x_length=9,
                y_length=4.5,
                tips=False,
            )
            x_lbl = axes.get_x_axis_label(M.Text("step", font_size=22))
            y_lbl = axes.get_y_axis_label(
                M.Text(self.value_label, font_size=22))
            self.play(M.Create(axes), M.FadeIn(x_lbl), M.FadeIn(y_lbl),
                      run_time=0.8)

            points = [axes.c2p(i, v) for i, v in enumerate(series)]
            dot = M.Dot(points[0], color=M.YELLOW)
            self.play(M.FadeIn(dot), run_time=0.3)

            # Animate the curve growing point-by-point so the recorded
            # evolution is visible step by step.
            drawn = M.VGroup()
            self.add(drawn)
            for i in range(1, n):
                seg = M.Line(points[i - 1], points[i], color=M.BLUE)
                drawn.add(seg)
                self.play(M.Create(seg), dot.animate.move_to(points[i]),
                          run_time=0.4)
            self.wait(0.6)

    class QpcnScene(LayerScene):
        """Animate the recorded QPCN energy curve over the frame sequence."""

        layer = "qpcn"
        value_key = "energy"
        title = "QPCN — energy descent"
        value_label = "energy"

    class ManifoldScene(LayerScene):
        """Animate the recorded manifold mean-absolute-curvature curve."""

        layer = "manifold"
        value_key = "mean_abs_ricci"
        title = "Manifold — mean |Ricci|"
        value_label = "mean |R|"

    class MultifieldScene(LayerScene):
        """Animate the recorded multifield mean-absolute-coupling curve."""

        layer = "multifield"
        value_key = "mean_abs_coupling"
        title = "Multifield — mean |coupling|"
        value_label = "mean |g|"

    class GenericScene(LayerScene):
        """Fallback scene for layers without a dedicated visualization."""

        layer = "generic"
        value_key = "value"
        title = "QFT-PCN layer"
        value_label = "value"

    return {
        "qpcn": QpcnScene,
        "manifold": ManifoldScene,
        "multifield": MultifieldScene,
        "_generic": GenericScene,
        "_base": LayerScene,
    }


def scene_class_for(layer: str):
    """Return the `Scene` subclass for `layer`, falling back to the generic one.

    Imports manim lazily via `build_scene_classes()`.
    """
    classes = build_scene_classes()
    return classes.get(layer, classes["_generic"])
