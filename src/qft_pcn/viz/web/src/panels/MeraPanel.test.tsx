import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MeraPanel, buildTree } from './MeraPanel';
import { meraFrame, emptyFrame } from './__fixtures__/frames';
import type { Frame } from '../lib/types';

it('mounts with a frame', () => {
  render(<MeraPanel frame={meraFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<MeraPanel frame={emptyFrame} />);
});

describe('MeraPanel readouts', () => {
  it('renders leaves, layers, and max χ readout cells', () => {
    render(<MeraPanel frame={meraFrame} />);
    expect(screen.getByText('leaves')).toBeInTheDocument();
    expect(screen.getByText('layers')).toBeInTheDocument();
    expect(screen.getByText('max χ')).toBeInTheDocument();
  });

  it('accepts an optional baselineFrame prop (no-op)', () => {
    render(<MeraPanel frame={meraFrame} baselineFrame={meraFrame} />);
    expect(screen.getByText('leaves')).toBeInTheDocument();
  });

  it('renders an iso err (max) readout and a per-layer sparkline', () => {
    const { container } = render(<MeraPanel frame={meraFrame} />);
    expect(screen.getByText('iso err (max)')).toBeInTheDocument();
    expect(
      container.querySelector('[data-testid="mera-iso-sparkline"]'),
    ).toBeTruthy();
  });

  it('renders a side-by-side two-disk layout when baselineFrame has data', () => {
    const { container } = render(
      <MeraPanel frame={meraFrame} baselineFrame={meraFrame} />,
    );
    const layout = container.querySelector('[data-testid="mera-layout"]');
    expect(layout).toBeTruthy();
    expect(layout!.getAttribute('data-compare')).toBe('side-by-side');
    expect(
      container.querySelector('[data-testid="mera-disk-active"]'),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="mera-disk-baseline"]'),
    ).toBeTruthy();
  });

  it('renders a single-disk layout when no baseline is supplied', () => {
    const { container } = render(<MeraPanel frame={meraFrame} />);
    const layout = container.querySelector('[data-testid="mera-layout"]');
    expect(layout!.getAttribute('data-compare')).toBe('single');
    expect(
      container.querySelector('[data-testid="mera-disk-baseline"]'),
    ).toBeNull();
  });
});

describe('MeraPanel — odd-leaf-count handling', () => {
  // Bug 1: `Math.floor(count/2)` silently dropped the trailing unpaired leaf
  // at each coarsening, producing wrong-topology trees for n_leaves ∈ {5, 7, …}.
  // The fix rounds up and treats the last parent as a 1-child pass-through.
  it('keeps every leaf in the tree for n_leaves=5', () => {
    const { nodes } = buildTree(5, [3, 2, 1], [2, 2, 2]);
    const leaves = nodes.filter((n) => n.kind === 'leaf');
    expect(leaves).toHaveLength(5);
    // First coarsening: ceil(5/2)=3 parents (not 2 = floor(5/2)).
    const depth1 = nodes.filter((n) => n.depth === 1);
    expect(depth1).toHaveLength(3);
    // Second coarsening: ceil(3/2)=2 parents.
    const depth2 = nodes.filter((n) => n.depth === 2);
    expect(depth2).toHaveLength(2);
    // Third coarsening: ceil(2/2)=1 parent (root).
    const depth3 = nodes.filter((n) => n.depth === 3);
    expect(depth3).toHaveLength(1);
  });

  it('does not wrap a lone unpaired parent back to leaf 0 with a stray edge', () => {
    // With n_leaves=5, the third parent at depth=1 has only ONE child
    // (leaf 4) — it must produce exactly one edge, not two-with-modulo
    // (which would have looped back to leaf 0 and made a fake cross-disk
    // chord). Total edges at the first coarsening = 2 + 2 + 1 = 5.
    const { edges, nodes } = buildTree(5, [3], [2]);
    const firstLayerEdges = edges.filter(
      (e) => e.a.depth === 1 && e.b.depth === 0,
    );
    expect(firstLayerEdges).toHaveLength(5);
    // Every leaf is touched by exactly one parent-edge — no leaf is double-used.
    const leaves = nodes.filter((n) => n.kind === 'leaf');
    for (const leaf of leaves) {
      const touchingEdges = firstLayerEdges.filter((e) => e.b === leaf);
      expect(touchingEdges).toHaveLength(1);
    }
  });

  it('renders a 5-leaf snapshot without crashing', () => {
    const oddFrame: Frame = {
      step: 1,
      layer_states: {
        mera: {
          n_leaves: 5,
          layer_dims: [3, 2, 1],
          bond_dims: [2, 2, 2],
          entropies: [0.1, 0.2, 0.3, 0.2],
          iso_residuals: [1e-7, 1e-6, 1e-5],
        },
      },
    };
    render(<MeraPanel frame={oddFrame} />);
    expect(screen.getByText('leaves')).toBeInTheDocument();
  });
});

describe('MeraPanel — value-stable memo across frame re-parses', () => {
  // Bug 2: `bondDims` / `layerDims` arrive from JSON.parse'd frames, so a new
  // array identity lands every frame even when values are unchanged. The panel
  // must not crash and must accept successive renders where the dep arrays
  // have fresh identity but identical contents (a smoke check that the
  // value-stable memo key still produces a valid tree).
  it('re-renders cleanly when bond/layer dim arrays have new identity but same values', () => {
    const makeFrame = (): Frame => ({
      step: 2,
      layer_states: {
        mera: {
          n_leaves: 8,
          // Fresh array literals each call → new identities.
          layer_dims: [4, 2, 1],
          bond_dims: [2, 3, 4],
          entropies: [0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1],
          iso_residuals: [1e-7, 1e-6, 1e-5],
        },
      },
    });
    const { rerender, container } = render(<MeraPanel frame={makeFrame()} />);
    rerender(<MeraPanel frame={makeFrame()} />);
    rerender(<MeraPanel frame={makeFrame()} />);
    // Panel still mounted, single disk present.
    expect(
      container.querySelector('[data-testid="mera-disk-active"]'),
    ).toBeTruthy();
  });
});
