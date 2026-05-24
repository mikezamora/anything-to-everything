import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { search } from './search-index';

describe('search-index', () => {
  it('finds the free-energy-functional equation for "precision-weighted"', () => {
    const results = search('precision-weighted');
    const eq = results.find(
      (r) => r.target.kind === 'equation' && r.target.id === 'free-energy-functional',
    );
    expect(eq, 'precision-weighted should hit free-energy-functional').toBeTruthy();
  });

  it('returns at least one article and one equation/layer result for "MERA isometry"', () => {
    const results = search('MERA isometry');
    expect(results.length).toBeGreaterThan(0);
    const article = results.find((r) => r.target.kind === 'article');
    const eqOrLayer = results.find(
      (r) => r.target.kind === 'equation' || r.target.kind === 'layer',
    );
    expect(article, 'expected an article hit for MERA isometry').toBeTruthy();
    expect(eqOrLayer, 'expected an equation or layer hit for MERA isometry').toBeTruthy();
  });

  it('finds the multifield-yukawa equation and the pcn-multifield article for "Yukawa"', () => {
    const results = search('Yukawa');
    const eq = results.find(
      (r) => r.target.kind === 'equation' && r.target.id === 'multifield-yukawa',
    );
    const art = results.find(
      (r) => r.target.kind === 'article' && r.target.id === 'pcn-multifield',
    );
    expect(eq, 'expected multifield-yukawa equation hit').toBeTruthy();
    expect(art, 'expected pcn-multifield article hit').toBeTruthy();
  });

  it('returns an empty result list for an empty / sub-3-char query', () => {
    expect(search('')).toEqual([]);
    expect(search('a b')).toEqual([]);
  });
});
