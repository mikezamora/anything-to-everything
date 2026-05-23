// src/qft_pcn/viz/web/src/components/ExplainerPane.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { ExplainerPane } from './ExplainerPane';
import { EXPLAINERS } from '../lib/explainer';

describe('ExplainerPane', () => {
  it('renders title + one-liner for every layer', () => {
    for (const key of Object.keys(EXPLAINERS)) {
      const { unmount } = render(<ExplainerPane layer={key} />);
      expect(screen.getByText(EXPLAINERS[key].title)).toBeInTheDocument();
      expect(screen.getByText(EXPLAINERS[key].oneLine)).toBeInTheDocument();
      unmount();
    }
  });

  it('renders "What", "Elements", "Math", "Watch" sections', () => {
    render(<ExplainerPane layer="manifold" />);
    expect(screen.getByText('What')).toBeInTheDocument();
    expect(screen.getByText('Elements')).toBeInTheDocument();
    expect(screen.getByText('Math')).toBeInTheDocument();
    expect(screen.getByText('Watch')).toBeInTheDocument();
  });

  it('toggles collapsed state via the header button', () => {
    const { container } = render(<ExplainerPane layer="manifold" />);
    const btn = container.querySelector('button.explainer-toggle');
    expect(btn).toBeTruthy();
    fireEvent.click(btn!);
    expect(container.querySelector('.explainer.collapsed')).toBeTruthy();
  });
});
