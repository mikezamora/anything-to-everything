import { describe, it, expect } from 'vitest';
import { PANELS, panelFor } from './index';
import { LAYER_KEYS } from '../lib/types';

describe('panel registry', () => {
  it('has a panel for every layer key', () => {
    for (const key of LAYER_KEYS) {
      expect(PANELS[key]).toBeTypeOf('function');
      expect(panelFor(key)).toBe(PANELS[key]);
    }
  });

  it('returns undefined for an unknown layer', () => {
    expect(panelFor('nope')).toBeUndefined();
  });
});
