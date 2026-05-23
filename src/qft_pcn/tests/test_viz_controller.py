"""RunController unit tests: pause/resume/step semantics."""

import asyncio
import pytest

from src.qft_pcn.viz.controller import RunController
from src.qft_pcn.viz.runs import RunSpec


@pytest.mark.asyncio
async def test_resume_streams_all_frames():
    spec = RunSpec(layers=["manifold"], steps=3, grid=8)
    ctrl = RunController(spec)
    frames = []
    async for f in ctrl.frames():
        frames.append(f)
    assert len(frames) == 3


@pytest.mark.asyncio
async def test_pause_blocks_next_frame_until_resume():
    spec = RunSpec(layers=["manifold"], steps=5, grid=8)
    ctrl = RunController(spec)
    received = []

    async def consume():
        async for f in ctrl.frames():
            received.append(f)
            if len(received) == 2:
                ctrl.pause()

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    assert len(received) == 2
    ctrl.resume()
    await task
    assert len(received) == 5


@pytest.mark.asyncio
async def test_step_advances_exactly_one_frame_while_paused():
    spec = RunSpec(layers=["manifold"], steps=5, grid=8)
    ctrl = RunController(spec)
    ctrl.pause()
    received = []

    async def consume():
        async for f in ctrl.frames():
            received.append(f)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.02)
    assert received == []
    ctrl.step()
    await asyncio.sleep(0.05)
    assert len(received) == 1
    ctrl.step()
    await asyncio.sleep(0.05)
    assert len(received) == 2
    ctrl.resume()
    await task
