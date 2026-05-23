/**
 * Unit tests for the pure MERA tree layout `buildTree`. The r3f `<Canvas>`
 * mock means the panel smoke test never runs this geometry code, so we
 * exercise it directly here.
 */

import { describe, it, expect } from 'vitest';
import { buildTree } from './MeraPanel';

describe('buildTree', () => {
  it('returns empty for non-positive leaf counts', () => {
    expect(buildTree(0, [], [])).toEqual({ nodes: [], edges: [] });
    expect(buildTree(-3, [4], [2])).toEqual({ nodes: [], edges: [] });
  });

  it('places all leaves on the unit boundary circle', () => {
    const { nodes } = buildTree(8, [4, 2], [2, 2]);
    const leaves = nodes.filter((n) => n.kind === 'leaf');
    expect(leaves).toHaveLength(8);
    for (const leaf of leaves) {
      const r = Math.hypot(leaf.x, leaf.y);
      expect(r).toBeCloseTo(1.0, 6);
      expect(leaf.depth).toBe(0);
    }
  });

  it('halves the node count at each coarser layer', () => {
    const { nodes } = buildTree(8, [4, 2, 1], [2, 2, 2]);
    const byDepth = (d: number) => nodes.filter((n) => n.depth === d).length;
    expect(byDepth(0)).toBe(8);
    expect(byDepth(1)).toBe(4);
    expect(byDepth(2)).toBe(2);
    expect(byDepth(3)).toBe(1);
  });

  it('compresses coarser layers inward (strictly shrinking radius)', () => {
    const { nodes } = buildTree(8, [4, 2, 1], [2, 2, 2]);
    const radiusAt = (d: number) => {
      const n = nodes.find((m) => m.depth === d)!;
      return Math.hypot(n.x, n.y);
    };
    expect(radiusAt(1)).toBeLessThan(radiusAt(0));
    expect(radiusAt(2)).toBeLessThan(radiusAt(1));
    expect(radiusAt(3)).toBeLessThan(radiusAt(2));
  });

  it('labels every coarse-graining node as an isometry — D-6', () => {
    // The substrate exposes `isometries` per layer; intra- and inter-pair
    // disentanglers exist on the substrate but are not surfaced as separate
    // panel nodes. The previous "alternating by layer parity" scheme had no
    // basis in the architecture and is gone.
    const { nodes } = buildTree(8, [4, 2, 1], [2, 2, 2]);
    for (const n of nodes.filter((m) => m.depth > 0)) {
      expect(n.kind).toBe('isometry');
    }
  });

  it('connects every coarse node to two finer nodes', () => {
    const { nodes, edges } = buildTree(8, [4, 2, 1], [2, 2, 2]);
    const coarse = nodes.filter((n) => n.depth > 0);
    // each coarse node contributes exactly two edges.
    expect(edges).toHaveLength(coarse.length * 2);
    for (const e of edges) {
      expect(e.width).toBeGreaterThan(0);
    }
  });

  it('is deterministic across repeated calls', () => {
    const a = buildTree(8, [4, 2], [3, 1]);
    const b = buildTree(8, [4, 2], [3, 1]);
    expect(a).toEqual(b);
  });

  it('handles a single leaf with no coarsening', () => {
    const { nodes, edges } = buildTree(1, [], []);
    expect(nodes).toHaveLength(1);
    expect(edges).toHaveLength(0);
  });
});
