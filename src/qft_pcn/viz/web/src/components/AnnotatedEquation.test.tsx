// src/qft_pcn/viz/web/src/components/AnnotatedEquation.test.tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AnnotatedEquation } from './AnnotatedEquation';
import { ROLE_COLOR } from '../lib/equations';

describe('AnnotatedEquation', () => {
  it('renders KaTeX for a registered equation', () => {
    const { container } = render(
      <AnnotatedEquation id="__test_free_energy_functional__" />,
    );
    expect(container.querySelector('.katex')).toBeInTheDocument();
  });

  it('renders the natural-language gloss', () => {
    render(<AnnotatedEquation id="__test_free_energy_functional__" />);
    expect(screen.getByText(/precision-weighted squared error/i))
      .toBeInTheDocument();
  });

  it('renders one swatch + gloss per symbol with the role colour', () => {
    const { container } = render(
      <AnnotatedEquation id="__test_free_energy_functional__" />,
    );
    const swatches = container.querySelectorAll('.annotated-eq-swatch');
    expect(swatches.length).toBe(3);
    const styles = Array.from(swatches).map(
      (s) => (s as HTMLElement).style.background,
    );
    expect(styles).toContain(_rgb(ROLE_COLOR['output']));
    expect(styles).toContain(_rgb(ROLE_COLOR['param-learn']));
    expect(styles).toContain(_rgb(ROLE_COLOR['state']));
  });

  it('renders the citation footer', () => {
    render(<AnnotatedEquation id="__test_free_energy_functional__" />);
    expect(screen.getByText(/Arch §3\.1/)).toBeInTheDocument();
  });

  it('renders a fallback when the id is missing', () => {
    render(<AnnotatedEquation id="this-does-not-exist" />);
    expect(screen.getByText(/missing equation/i)).toBeInTheDocument();
  });
});

// Browsers compute color as e.g. `rgb(108, 208, 255)`. Convert hex for compare.
function _rgb(hex: string): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgb(${r}, ${g}, ${b})`;
}
