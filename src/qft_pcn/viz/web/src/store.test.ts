import { beforeEach, describe, expect, it } from 'vitest';
import { useVizStore, MAX_FRAMES } from './store';
import type { Frame } from './lib/types';

const f = (step: number): Frame => ({ step, layer_states: {} });

beforeEach(() => useVizStore.getState().resetAll());

describe('per-run frame map', () => {
  it('pushes frames into the active run only', () => {
    const s = useVizStore.getState();
    s.openRun('A');
    s.pushFrame('A', f(0));
    s.pushFrame('A', f(1));
    expect(useVizStore.getState().runs.get('A')?.frames.length).toBe(2);
    expect(useVizStore.getState().runs.has('B')).toBe(false);
  });

  it('caps frames per run at MAX_FRAMES', () => {
    const s = useVizStore.getState();
    s.openRun('A');
    for (let i = 0; i < MAX_FRAMES + 50; i++) s.pushFrame('A', f(i));
    expect(useVizStore.getState().runs.get('A')!.frames.length).toBe(MAX_FRAMES);
  });

  it('pinBaseline stores a second run id', () => {
    const s = useVizStore.getState();
    s.openRun('A'); s.openRun('B');
    s.setActiveRun('A'); s.pinBaseline('B');
    expect(useVizStore.getState().activeRunId).toBe('A');
    expect(useVizStore.getState().baselineRunId).toBe('B');
    s.unpinBaseline();
    expect(useVizStore.getState().baselineRunId).toBeNull();
  });

  it('setPaused and setPlaybackSpeed are independent globals', () => {
    const s = useVizStore.getState();
    s.setPaused(true);
    s.setPlaybackSpeed(2);
    expect(useVizStore.getState().paused).toBe(true);
    expect(useVizStore.getState().playbackSpeed).toBe(2);
  });
});
