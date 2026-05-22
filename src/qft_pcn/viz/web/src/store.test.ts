import { describe, it, expect, beforeEach } from 'vitest';
import { useVizStore } from './store';

describe('viz store', () => {
  beforeEach(() => useVizStore.getState().reset());

  it('appends frames and scrubs', () => {
    const s = useVizStore.getState();
    s.pushFrame({ step: 0, layer_states: {} });
    s.pushFrame({ step: 1, layer_states: {} });
    expect(useVizStore.getState().frames.length).toBe(2);
    useVizStore.getState().setCursor(0);
    expect(useVizStore.getState().currentFrame()?.step).toBe(0);
  });
});
