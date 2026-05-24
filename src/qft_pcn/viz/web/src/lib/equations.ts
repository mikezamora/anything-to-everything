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
