import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { annotate } from './mera-step-annotator';
import type { Frame } from './types';

function frame(step: number, rows: Array<[string, number, number]>): Frame {
  return {
    step,
    layer_states: {
      mera_relax: {
        residuals: rows.map(([rule_id, site, value]) => ({
          rule_id,
          site,
          value,
        })),
      },
    },
  };
}

describe('mera-step-annotator.annotate', () => {
  it('emits exactly one label when R-AddZero@site-3 drops 0.4 → 0.05 at frame 2', () => {
    // 4-frame synthetic sequence; the only significant drop is on
    // R-AddZero @ site 3 between frame index 1 (residual 0.4) and frame
    // index 2 (residual 0.05). All other rows hold steady (or drift by
    // less than the 0.05 threshold).
    const frames: Frame[] = [
      frame(0, [
        ['R-AddZero', 3, 0.40],
        ['R-Eq-Refl', 0, 0.10],
        ['R-Beta', 7, 0.02],
      ]),
      frame(1, [
        ['R-AddZero', 3, 0.40],
        ['R-Eq-Refl', 0, 0.09],
        ['R-Beta', 7, 0.02],
      ]),
      frame(2, [
        ['R-AddZero', 3, 0.05],
        ['R-Eq-Refl', 0, 0.08],
        ['R-Beta', 7, 0.02],
      ]),
      frame(3, [
        ['R-AddZero', 3, 0.05],
        ['R-Eq-Refl', 0, 0.07],
        ['R-Beta', 7, 0.02],
      ]),
    ];

    const labels = annotate(frames);

    // Only frame index 2 fired; frames 1 and 3 are sub-threshold.
    expect([...labels.keys()].sort()).toEqual([2]);
    const at2 = labels.get(2)!;
    expect(at2).toHaveLength(1);
    expect(at2[0].term.rule_id).toBe('R-AddZero');
    expect(at2[0].term.site).toBe(3);
    expect(at2[0].magnitude).toBeCloseTo(0.35, 5);
    expect(at2[0].description).toBe('R-AddZero fired at site 3');
  });

  it('returns an empty map on fewer than two frames', () => {
    expect(annotate([]).size).toBe(0);
    expect(annotate([frame(0, [['R-AddZero', 0, 1.0]])]).size).toBe(0);
  });

  it('ignores rows missing from the previous frame', () => {
    const frames: Frame[] = [
      frame(0, []),
      frame(1, [['R-AddZero', 3, 0.0]]),
    ];
    expect(annotate(frames).size).toBe(0);
  });

  it('sorts multiple firings in a frame by descending magnitude', () => {
    const frames: Frame[] = [
      frame(0, [
        ['R-AddZero', 3, 0.40],
        ['R-Eq-Refl', 0, 0.50],
      ]),
      frame(1, [
        ['R-AddZero', 3, 0.10], // drop 0.30
        ['R-Eq-Refl', 0, 0.05], // drop 0.45
      ]),
    ];
    const labels = annotate(frames).get(1)!;
    expect(labels).toHaveLength(2);
    expect(labels[0].term.rule_id).toBe('R-Eq-Refl');
    expect(labels[1].term.rule_id).toBe('R-AddZero');
  });
});
