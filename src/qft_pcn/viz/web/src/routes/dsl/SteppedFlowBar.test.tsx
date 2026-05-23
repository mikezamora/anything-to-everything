import '@testing-library/jest-dom/vitest';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SteppedFlowBar } from './SteppedFlowBar';
import { useVizStore } from '../../store';

beforeEach(() => {
  useVizStore.getState().resetAll();
  (globalThis as any).fetch = vi.fn();
});

describe('SteppedFlowBar', () => {
  it('Generate disabled when prompt empty', () => {
    render(<SteppedFlowBar />);
    expect(screen.getByRole('button', { name: /Generate DSL/ }))
      .toBeDisabled();
  });
  it('Run disabled when dslText empty', () => {
    render(<SteppedFlowBar />);
    expect(screen.getByRole('button', { name: /Run DSL/ })).toBeDisabled();
  });
  it('Verbalize disabled when no run', () => {
    render(<SteppedFlowBar />);
    expect(screen.getByRole('button', { name: /Verbalize/ })).toBeDisabled();
  });
});
