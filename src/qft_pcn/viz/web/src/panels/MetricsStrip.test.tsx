import { describe, it, expect, beforeEach } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { MetricsStrip } from './MetricsStrip';
import { useVizStore } from '../store';

beforeEach(() => useVizStore.getState().resetAll());

describe('MetricsStrip', () => {
  it('shows "collecting…" with fewer than 2 frames', () => {
    let s = useVizStore.getState();
    s.openRun('A');
    s = useVizStore.getState();
    s.setActiveRun('A');
    s = useVizStore.getState();
    s.pushFrame('A', { step: 0, layer_states: { qpcn: { energy: 1 } } });
    render(<MetricsStrip layer="qpcn" metrics={[
      { key: 'e', label: 'E', color: '#fa0',
        select: (ls) => ls.energy as number },
    ]} />);
    expect(screen.getByText('collecting…')).toBeInTheDocument();
  });

  it('renders one path per series once frames > 1', () => {
    let s = useVizStore.getState();
    s.openRun('A');
    s = useVizStore.getState();
    s.setActiveRun('A');
    for (let i = 0; i < 5; i++) {
      s = useVizStore.getState();
      s.pushFrame('A', { step: i, layer_states: { qpcn: { energy: i * 0.1 } } });
    }
    const { container } = render(<MetricsStrip layer="qpcn" metrics={[
      { key: 'e', label: 'E', color: '#fa0',
        select: (ls) => ls.energy as number },
    ]} />);
    expect(container.querySelectorAll('path').length).toBe(1);
  });
});
