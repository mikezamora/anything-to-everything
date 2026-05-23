"""Async wrapper around `run_simulation` that exposes pause/resume/step.

A `RunController` owns one (synchronous) simulation generator and an
`asyncio.Event` gate. The async `frames()` iterator pulls one frame at a
time, yielding it through the gate. `pause()` clears the gate; `resume()`
sets it; `step()` flips the gate to ready, allows exactly one frame
through, then re-pauses. Clients that never call any lifecycle method see
the original streaming behaviour.

This is the only place where the synchronous generator is bridged to the
async world; the rest of the server stays unaware.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, Iterator, Optional

from .runs import RunSpec, run_simulation
from .schema import Frame


class RunController:
    """Pause/step/resume wrapper around `run_simulation(spec)`.

    `runner` is injectable so the server can hand in the patchable
    `server.run_simulation` symbol; callers that don't pass one (tests of
    the controller in isolation) get the real driver.
    """

    def __init__(self, spec: RunSpec,
                 runner: Optional[Callable[..., Iterator[Frame]]]
                 = None) -> None:
        self._spec = spec
        self._runner = runner or run_simulation
        self._event = asyncio.Event()
        self._event.set()  # default: running
        self._step_request = 0
        # Populated when `frames()` starts iterating. Maps substrate-kind
        # (e.g. "qpcn", "vqc", "mera") to the live substrate instance, so
        # lifecycle endpoints can mutate parameters between frames.
        self._substrates: dict = {}

    async def frames(self) -> AsyncIterator[Frame]:
        """Yield one `Frame` per simulation step, gated by the event."""
        def _capture(subs: dict) -> None:
            self._substrates = subs
        # Older runner stubs may not accept `on_build`; fall back gracefully.
        try:
            gen = self._runner(self._spec, on_build=_capture)
        except TypeError:
            gen = self._runner(self._spec)
        for frame in gen:
            await self._event.wait()
            yield frame
            await asyncio.sleep(0)
            if self._step_request > 0:
                self._step_request -= 1
                self._event.clear()

    def pause(self) -> None:
        """Clear the gate; the next frame will block on `await`."""
        self._event.clear()

    def resume(self) -> None:
        """Set the gate; pending awaits proceed."""
        self._event.set()

    def step(self) -> None:
        """Allow exactly one frame through, then re-pause."""
        self._step_request += 1
        self._event.set()
