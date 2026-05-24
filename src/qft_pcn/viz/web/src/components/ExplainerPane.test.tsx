// src/qft_pcn/viz/web/src/components/ExplainerPane.test.tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ExplainerPane } from './ExplainerPane';
import { EXPLAINERS } from '../lib/explainer';

describe('ExplainerPane (tabbed)', () => {
  it('renders title + one-liner for every layer', () => {
    for (const key of Object.keys(EXPLAINERS)) {
      const { unmount } = render(<ExplainerPane layer={key} />);
      expect(screen.getByText(EXPLAINERS[key].title)).toBeInTheDocument();
      expect(screen.getByText(EXPLAINERS[key].oneLine)).toBeInTheDocument();
      unmount();
    }
  });

  it('renders the 5 tab buttons', () => {
    render(<ExplainerPane layer="manifold" />);
    for (const label of ['Overview', 'Math', 'Worked Example', 'Dynamics', 'Watch']) {
      expect(screen.getByRole('button', { name: label })).toBeInTheDocument();
    }
  });

  it('clicking a tab swaps the body content', () => {
    render(<ExplainerPane layer="manifold" />);
    // Default: Overview shows "What" heading.
    expect(screen.getByText('What')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Watch' }));
    // After click the Watch body section is shown; scope to the heading
    // because the tab button itself also reads "Watch".
    expect(screen.getByRole('heading', { name: 'Watch' })).toBeInTheDocument();
  });

  it('toggles collapsed state via the header button', () => {
    const { container } = render(<ExplainerPane layer="manifold" />);
    const btn = container.querySelector('button.explainer-toggle')!;
    fireEvent.click(btn);
    expect(container.querySelector('.explainer.collapsed')).toBeTruthy();
  });
});
