"""Manim export pipeline for QFT-PCN layer visualizations.

`render.py` turns a recorded frame sequence into an MP4; `scenes.py` holds one
`Scene` subclass per layer. Both modules import `manim` lazily so importing
this package (and the FastAPI server) never hard-fails when `manim` is absent.
"""
