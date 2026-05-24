/**
 * Unit tests for the pure geometry / colour helpers in `common.ts`. These run
 * the real numeric code that the r3f `<Canvas>`-mocked panel smoke tests
 * never exercise.
 */

import { describe, it, expect } from 'vitest';
import {
  normGrid,
  diverging,
  sequential,
  speciesColor,
  smallNumberFormat,
} from './common';

describe('normGrid', () => {
  it('scales values into [-1, 1] by peak abs value', () => {
    const { norm, peak } = normGrid([
      [0, 2],
      [-4, 1],
    ]);
    expect(peak).toBe(4);
    expect(norm).toEqual([
      [0, 0.5],
      [-1, 0.25],
    ]);
    for (const row of norm) {
      for (const v of row) {
        expect(v).toBeGreaterThanOrEqual(-1);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
  });

  it('returns peak 0 and zeros for an all-zero grid (no divide-by-zero)', () => {
    const { norm, peak } = normGrid([
      [0, 0],
      [0, 0],
    ]);
    expect(peak).toBe(0);
    expect(norm).toEqual([
      [0, 0],
      [0, 0],
    ]);
  });

  it('treats non-finite entries as zero', () => {
    const { norm, peak } = normGrid([[Number.NaN, Infinity, 3]]);
    expect(peak).toBe(3);
    expect(norm[0][0]).toBe(0);
    expect(norm[0][1]).toBe(0);
    expect(norm[0][2]).toBe(1);
  });

  it('is deterministic across repeated calls', () => {
    const grid = [
      [1, -2],
      [3, 0.5],
    ];
    expect(normGrid(grid)).toEqual(normGrid(grid));
  });
});

describe('diverging', () => {
  it('returns a valid rgb() string for in-range input', () => {
    expect(diverging(0)).toMatch(/^rgb\(\d+, \d+, \d+\)$/);
  });

  it('clamps out-of-range input to the endpoints', () => {
    expect(diverging(5)).toBe(diverging(1));
    expect(diverging(-5)).toBe(diverging(-1));
  });

  it('maps 0 to white, negatives toward blue, positives toward red', () => {
    expect(diverging(0)).toBe('rgb(255, 255, 255)');
    expect(diverging(-1)).toBe('rgb(60, 90, 255)');
    expect(diverging(1)).toBe('rgb(255, 90, 60)');
  });

  it('produces channels within 0..255', () => {
    for (const t of [-1, -0.3, 0, 0.7, 1]) {
      const m = diverging(t).match(/\d+/g)!.map(Number);
      for (const c of m) {
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(255);
      }
    }
  });
});

describe('sequential', () => {
  it('clamps to [0, 1] and is deterministic', () => {
    expect(sequential(2)).toBe(sequential(1));
    expect(sequential(-1)).toBe(sequential(0));
    expect(sequential(0.5)).toBe(sequential(0.5));
  });

  it('produces channels within 0..255', () => {
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      const m = sequential(t).match(/\d+/g)!.map(Number);
      for (const c of m) {
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(255);
      }
    }
  });
});

describe('speciesColor', () => {
  it('is consistent regardless of the order names are passed', () => {
    const a = ['phi', 'psi', 'chi'];
    const b = ['chi', 'phi', 'psi'];
    expect(speciesColor('phi', a)).toBe(speciesColor('phi', b));
    expect(speciesColor('psi', a)).toBe(speciesColor('psi', b));
  });

  it('gives distinct species distinct colours when there is a spread', () => {
    const names = ['a', 'b', 'c'];
    expect(speciesColor('a', names)).not.toBe(speciesColor('c', names));
  });

  it('returns a valid rgb() string and handles a single species', () => {
    expect(speciesColor('only', ['only'])).toMatch(/^rgb\(\d+, \d+, \d+\)$/);
  });

  it('falls back to index 0 for an unknown name', () => {
    expect(speciesColor('missing', ['a', 'b'])).toBe(
      speciesColor('a', ['a', 'b']),
    );
  });
});

describe('smallNumberFormat', () => {
  it('returns the em-dash placeholder for null, undefined, NaN, and Infinity', () => {
    expect(smallNumberFormat(null)).toBe('—');
    expect(smallNumberFormat(undefined)).toBe('—');
    expect(smallNumberFormat(Number.NaN)).toBe('—');
    expect(smallNumberFormat(Infinity)).toBe('—');
    expect(smallNumberFormat(-Infinity)).toBe('—');
  });

  it('renders exact zero as "0" (no decimals, no exponent)', () => {
    expect(smallNumberFormat(0)).toBe('0');
  });

  it('uses fixed-point with 3 decimals when |v| >= 1e-3', () => {
    expect(smallNumberFormat(1)).toBe('1.000');
    expect(smallNumberFormat(0.5)).toBe('0.500');
    expect(smallNumberFormat(0.001)).toBe('0.001');
    expect(smallNumberFormat(-0.25)).toBe('-0.250');
  });

  it('falls back to two-digit exponential for sub-millisecond magnitudes', () => {
    expect(smallNumberFormat(2.9e-5)).toBe('2.90e-5');
    expect(smallNumberFormat(1.7e-7)).toBe('1.70e-7');
    expect(smallNumberFormat(-3.14e-4)).toBe('-3.14e-4');
    expect(smallNumberFormat(9.99e-4)).toBe('9.99e-4');
  });
});
