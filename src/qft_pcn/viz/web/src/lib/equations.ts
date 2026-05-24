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

EQUATIONS['free-energy-functional'] = {
  id: 'free-energy-functional',
  tex: 'F[\\Phi, E, \\Pi] = \\int_M \\left[ \\tfrac{1}{2} \\Pi(x) E(x)^2 - \\tfrac{1}{2} \\log \\Pi(x) \\right] \\sqrt{|g|}\\, d^2x',
  gloss: 'The variational free energy: a manifold integral of precision-weighted squared error minus the log-precision (a regulariser).',
  symbolGlosses: [
    { symbol: 'F',         gloss: 'free energy',           role: 'output' },
    { symbol: '\\Phi',     gloss: 'belief field',          role: 'state' },
    { symbol: 'E',         gloss: 'prediction error field', role: 'state' },
    { symbol: '\\Pi',      gloss: 'precision field',       role: 'param-learn' },
    { symbol: 'g',         gloss: 'metric on the manifold M', role: 'state' },
  ],
  sourceCitation: 'Arch §3.1',
};

EQUATIONS['metric-perturbation'] = {
  id: 'metric-perturbation',
  tex: 'g_{\\mu\\nu}(x) = \\eta_{\\mu\\nu} + h_{\\mu\\nu}(x)',
  gloss: 'The metric is a fixed flat background plus a learned perturbation.',
  symbolGlosses: [
    { symbol: 'g_{\\mu\\nu}', gloss: 'full metric',            role: 'state' },
    { symbol: '\\eta_{\\mu\\nu}', gloss: 'flat reference metric', role: 'param-const' },
    { symbol: 'h_{\\mu\\nu}', gloss: 'learnable perturbation',  role: 'param-learn' },
  ],
  sourceCitation: 'Arch §2.1 / src/qft_pcn/manifold.py',
};

EQUATIONS['ricci-scalar'] = {
  id: 'ricci-scalar',
  tex: 'R = g^{\\mu\\nu} R_{\\mu\\nu}',
  gloss: 'The Ricci scalar contracts the Ricci tensor with the inverse metric — a coordinate-invariant measure of intrinsic curvature.',
  symbolGlosses: [
    { symbol: 'R',            gloss: 'Ricci scalar',            role: 'output' },
    { symbol: 'g^{\\mu\\nu}', gloss: 'inverse metric',          role: 'state' },
    { symbol: 'R_{\\mu\\nu}', gloss: 'Ricci tensor',            role: 'state' },
  ],
  sourceCitation: 'Arch §3.1 / src/qft_pcn/manifold.py:ricci_scalar',
};

EQUATIONS['laplace-beltrami'] = {
  id: 'laplace-beltrami',
  tex: '\\Delta_g \\Phi = \\tfrac{1}{\\sqrt{|g|}} \\partial_\\mu \\big( \\sqrt{|g|}\\, g^{\\mu\\nu} \\partial_\\nu \\Phi \\big)',
  gloss: 'Belief diffuses via the Laplace-Beltrami operator built from the current metric — geometry sets the flow of information.',
  symbolGlosses: [
    { symbol: '\\Delta_g',   gloss: 'Laplace-Beltrami operator on the metric g', role: 'state' },
    { symbol: '\\Phi',       gloss: 'belief field',           role: 'state' },
    { symbol: 'g^{\\mu\\nu}', gloss: 'inverse metric',         role: 'state' },
  ],
  sourceCitation: 'Arch §3.1',
};

EQUATIONS['mps-ansatz'] = {
  id: 'mps-ansatz',
  tex: '|\\psi\\rangle = \\sum_{\\{s\\}} A^{s_1} A^{s_2} \\cdots A^{s_N} |s_1 \\ldots s_N\\rangle',
  gloss: 'A many-body quantum state expressed as a contraction of per-site rank-3 tensors A; the bond dimension caps how much entanglement can cross any cut.',
  symbolGlosses: [
    { symbol: '|\\psi\\rangle', gloss: 'many-body state',     role: 'state' },
    { symbol: 'A^{s_k}',        gloss: 'site-k tensor',       role: 'param-learn' },
    { symbol: 's_k',            gloss: 'local basis index',   role: 'input' },
  ],
  sourceCitation: 'Arch §2.3 / src/qft_pcn/qft/mps.py',
};

EQUATIONS['entanglement-entropy'] = {
  id: 'entanglement-entropy',
  tex: 'S(\\rho_A) = - \\mathrm{Tr}\\, \\rho_A \\log \\rho_A',
  gloss: 'Von Neumann entropy of a reduced density matrix — quantifies entanglement across a chosen bipartition.',
  symbolGlosses: [
    { symbol: 'S',         gloss: 'entanglement entropy',   role: 'output' },
    { symbol: '\\rho_A',   gloss: 'reduced density matrix on subsystem A', role: 'state' },
  ],
  sourceCitation: 'Arch §2.3',
};

EQUATIONS['hamiltonian-decomp'] = {
  id: 'hamiltonian-decomp',
  tex: 'H = \\sum_i h_i + \\sum_{\\langle i,j \\rangle} h_{ij}',
  gloss: 'A local many-body Hamiltonian is a sum of on-site terms plus nearest-neighbour two-site terms.',
  symbolGlosses: [
    { symbol: 'H',       gloss: 'full Hamiltonian',     role: 'output' },
    { symbol: 'h_i',     gloss: 'one-site term at site i', role: 'param-learn' },
    { symbol: 'h_{ij}',  gloss: 'two-site bond term',   role: 'param-learn' },
  ],
  sourceCitation: 'Arch §3.3 / src/qft_pcn/qft/hamiltonian.py',
};

EQUATIONS['imag-time-evolution'] = {
  id: 'imag-time-evolution',
  tex: '|\\psi(\\tau + d\\tau)\\rangle = e^{-H\\, d\\tau} \\, |\\psi(\\tau)\\rangle',
  gloss: 'Imaginary-time evolution projects toward the Hamiltonian\'s ground state by exponentially suppressing higher-energy components.',
  symbolGlosses: [
    { symbol: '|\\psi(\\tau)\\rangle', gloss: 'state at imaginary time τ', role: 'state' },
    { symbol: 'H',                       gloss: 'Hamiltonian',             role: 'param-learn' },
    { symbol: '\\tau',                   gloss: 'imaginary time',          role: 'input' },
  ],
  sourceCitation: 'Arch §3.4 / src/qft_pcn/qft/evolution.py',
};

EQUATIONS['parameter-shift-rule'] = {
  id: 'parameter-shift-rule',
  tex: '\\partial_\\theta \\langle O \\rangle = \\tfrac{1}{2} \\big[ \\langle O \\rangle_{\\theta + \\pi/2} - \\langle O \\rangle_{\\theta - \\pi/2} \\big]',
  gloss: 'Exact gradient of a Pauli-rotation expectation value, computable on quantum hardware via two shifted-parameter evaluations.',
  symbolGlosses: [
    { symbol: '\\theta',         gloss: 'circuit angle',     role: 'param-learn' },
    { symbol: '\\langle O \\rangle', gloss: 'observable expectation', role: 'observable' },
  ],
  sourceCitation: 'Arch §2.4',
};

EQUATIONS['multifield-yukawa'] = {
  id: 'multifield-yukawa',
  tex: 'L_{\\text{int}} = \\sum_{i<j} g_{ij}(x)\\, \\Phi_i(x) \\Phi_j(x)',
  gloss: 'Yukawa-style interaction: a learnable coupling field g_ij weights the cross-species product term in the Lagrangian.',
  symbolGlosses: [
    { symbol: 'L_{\\text{int}}', gloss: 'interaction Lagrangian', role: 'output' },
    { symbol: 'g_{ij}',           gloss: 'pairwise coupling field', role: 'param-learn' },
    { symbol: '\\Phi_i',          gloss: 'species-i belief field',  role: 'state' },
  ],
  sourceCitation: 'Arch §2.2 / src/qft_pcn/multifield.py',
};

EQUATIONS['coupling-descent'] = {
  id: 'coupling-descent',
  tex: '\\dot g_{ij} = -\\eta\\, \\partial_{g_{ij}} F',
  gloss: 'Couplings descend the joint free energy: pairs of fields that explain each other\'s errors grow their coupling.',
  symbolGlosses: [
    { symbol: '\\dot g_{ij}',   gloss: 'coupling rate of change', role: 'state' },
    { symbol: '\\eta',           gloss: 'learning rate',          role: 'param-const' },
    { symbol: 'F',               gloss: 'joint free energy',      role: 'output' },
  ],
  sourceCitation: 'Arch §2.2 / src/qft_pcn/multifield.py',
};

EQUATIONS['param-update'] = {
  id: 'param-update',
  tex: '\\Delta \\theta = -\\eta\\, \\partial_\\theta \\sum_o (\\langle O \\rangle - t_o)^2',
  gloss: 'Hamiltonian parameters descend the squared mismatch between current operator expectations and observation targets.',
  symbolGlosses: [
    { symbol: '\\theta',     gloss: 'Hamiltonian parameter',     role: 'param-learn' },
    { symbol: '\\eta',       gloss: 'learning rate',            role: 'param-const' },
    { symbol: '\\langle O \\rangle', gloss: 'current expectation', role: 'observable' },
    { symbol: 't_o',         gloss: 'observation target',       role: 'input' },
  ],
  sourceCitation: 'Arch §2.3 / src/qft_pcn/qft/qpcn.py',
};
