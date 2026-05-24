// src/qft_pcn/viz/web/src/routes/TrainingRoute.test.tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TrainingRoute } from './TrainingRoute';

describe('TrainingRoute', () => {
  it('mounts and renders the title', () => {
    render(<TrainingRoute />);
    expect(screen.getByText('Training the QPCN end-to-end'))
      .toBeInTheDocument();
  });

  it('renders the per-frame loop prose', () => {
    render(<TrainingRoute />);
    expect(screen.getByText(/per-frame loop/i)).toBeInTheDocument();
  });

  it('renders the pathologies section', () => {
    render(<TrainingRoute />);
    expect(screen.getByText(/F oscillates/)).toBeInTheDocument();
  });
});
