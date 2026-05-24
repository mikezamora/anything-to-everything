// src/qft_pcn/viz/web/src/lib/equations.ts
/**
 * Annotated-equation registry. Every equation in the docs is registered
 * here so the renderer can attach role-coloured symbol glosses + a
 * citation back to the architecture doc or the substrate source. The
 * role palette is the visual vocabulary readers build across the viz.
 */

export type Role =
  | 'input'
  | 'param-learn'
  | 'param-const'
  | 'output'
  | 'state'
  | 'observable';

export const ROLE_COLOR: Record<Role, string> = {
  'input':       '#6cd0ff',
  'param-learn': '#fbc66a',
  'param-const': '#7e8aa3',
  'output':      '#9aedc1',
  'state':       '#d291ff',
  'observable':  '#ef9090',
};

export const ROLE_LABEL: Record<Role, string> = {
  'input':       'input',
  'param-learn': 'param (learn)',
  'param-const': 'param (const)',
  'output':      'output',
  'state':       'state',
  'observable':  'observable',
};

export interface SymbolGloss {
  symbol: string;     // KaTeX-rendered string e.g. '\\Pi'
  gloss: string;      // natural-language one-liner
  role: Role;
}

export interface EquationSpec {
  id: string;
  tex: string;
  gloss: string;
  symbolGlosses: SymbolGloss[];
  sourceCitation: string;       // 'Arch §3.1' or 'src/qft_pcn/layer.py:192'
}

export const EQUATIONS: Record<string, EquationSpec> = {};

// Fixture equation for AnnotatedEquation tests; will be displaced by real T8 entries.
EQUATIONS['__test_free_energy_functional__'] = {
  id: '__test_free_energy_functional__',
  tex: 'F = \\tfrac{1}{2} \\Pi E^2 - \\tfrac{1}{2} \\log \\Pi',
  gloss: 'Free energy is half the precision-weighted squared error minus half the log precision.',
  symbolGlosses: [
    { symbol: 'F',       gloss: 'free energy',       role: 'output' },
    { symbol: '\\Pi',    gloss: 'precision',         role: 'param-learn' },
    { symbol: 'E',       gloss: 'prediction error',  role: 'state' },
  ],
  sourceCitation: 'Arch §3.1',
};
