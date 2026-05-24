// src/qft_pcn/viz/web/src/components/FrameInterpreter.test.tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FrameInterpreter } from './FrameInterpreter';
import { useVizStore } from '../store';
import { INTERPRETERS } from '../lib/interpreters';

beforeEach(() => useVizStore.getState().resetAll());

describe('FrameInterpreter', () => {
  it('renders nothing when there is no active run', () => {
    const { container } = render(<FrameInterpreter layer="manifold" />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when the interpreter returns null', () => {
    const s = useVizStore.getState();
    s.openRun('A'); s.setActiveRun('A');
    s.pushFrame('A', { step: 0,
                       layer_states: { manifold: { /* no mean_abs_ricci */ } } });
    const { container } = render(<FrameInterpreter layer="manifold" />);
    expect(container.firstChild).toBeNull();
  });

  it('renders the interpretation text when interpreter returns output', () => {
    const s = useVizStore.getState();
    s.openRun('A'); s.setActiveRun('A');
    s.pushFrame('A', { step: 12,
      layer_states: { manifold: { mean_abs_ricci: 0.55 } } });
    render(<FrameInterpreter layer="manifold" />);
    expect(screen.getByText(/curvature is concentrating/i))
      .toBeInTheDocument();
    expect(screen.getByText(/Step 12/)).toBeInTheDocument();
  });

  it('exposes all registered interpreters as callable functions', () => {
    for (const [, fn] of Object.entries(INTERPRETERS)) {
      expect(typeof fn).toBe('function');
      // Each must handle empty state without throwing:
      expect(fn({}, 0)).toBeNull();
    }
  });
});
