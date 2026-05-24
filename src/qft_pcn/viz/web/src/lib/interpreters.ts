// src/qft_pcn/viz/web/src/lib/interpreters.ts
/**
 * Per-panel live-frame interpreters. Each takes the panel's current
 * layer_state dict + the global frame step, returns a 1-2 sentence
 * interpretation or null when there's nothing meaningful to say.
 *
 * Citations are article ids (Learn route) so the FrameInterpreter
 * overlay can deep-link.
 *
 * All reads MUST be defensive — missing fields yield null, not exceptions.
 */

export interface InterpretationOutput {
  text: string;
  citation?: string;
}

export type Interpreter = (
  layerState: Record<string, unknown>,
  step: number,
) => InterpretationOutput | null;

function _num(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

export const INTERPRETERS: Record<string, Interpreter> = {
  manifold: (st, step) => {
    const mar = _num(st['mean_abs_ricci']);
    if (mar === undefined) return null;
    if (mar > 0.4) {
      return {
        text: `Step ${step}: mean|R| = ${mar.toFixed(3)} — curvature is concentrating, likely tracking an error spike.`,
        citation: 'fusion-manifold',
      };
    }
    if (mar < 0.02) {
      return {
        text: `Step ${step}: mean|R| ≈ 0 — geometry has nearly flattened; the error field is no longer sourcing curvature.`,
        citation: 'fusion-manifold',
      };
    }
    return {
      text: `Step ${step}: mean|R| = ${mar.toFixed(3)} — moderate curvature, the system is mid-relaxation.`,
      citation: 'fusion-manifold',
    };
  },

  multifield: (st, step) => {
    const mac = _num(st['mean_abs_coupling']);
    if (mac === undefined) return null;
    if (mac < 0.02) {
      return { text: `Step ${step}: mean|g| ≈ 0 — fields are essentially uncoupled.`,
               citation: 'pcn-multifield' };
    }
    if (mac > 0.5) {
      return { text: `Step ${step}: mean|g| = ${mac.toFixed(3)} — strong cross-field coupling; expect joint relaxation.`,
               citation: 'pcn-multifield' };
    }
    return { text: `Step ${step}: mean|g| = ${mac.toFixed(3)} — moderate coupling growing under correlated errors.`,
             citation: 'pcn-multifield' };
  },

  mps: (st, step) => {
    const bd = st['bond_dims'];
    const ents = st['entropies'];
    if (!Array.isArray(bd) || !Array.isArray(ents)) return null;
    const total = (ents as (number | null)[])
      .reduce<number>((a, v) => a + (v ?? 0), 0);
    const chiMax = Math.max(...(bd as number[]));
    return { text: `Step ${step}: χ_max = ${chiMax}, total entanglement entropy = ${total.toFixed(3)} nats.`,
             citation: 'qft-mps' };
  },

  hamiltonian: (st, _step) => {
    const n = _num(st['n_sites']);
    const d = _num(st['d_local']);
    const species = st['species'];
    if (n === undefined || d === undefined || !Array.isArray(species)) return null;
    return { text: `Hamiltonian on ${n} sites, local Fock dim ${d}, ${(species as string[]).length} species. Acts as the generative model whose ground state is the QPCN's belief.`,
             citation: 'qft-hamiltonian' };
  },

  qpcn: (st, step) => {
    const e = _num(st['energy']);
    if (e === undefined) return null;
    return { text: `Step ${step}: ⟨H⟩ = ${e.toFixed(4)} — variational energy under imag-time should decrease monotonically.`,
             citation: 'fusion-qpcn' };
  },

  mera: (st, _step) => {
    const leaves = _num(st['n_leaves']);
    const ld = st['layer_dims'];
    if (leaves === undefined || !Array.isArray(ld)) return null;
    return { text: `MERA on ${leaves} leaves, ${(ld as number[]).length} hierarchical layers. Multi-scale entanglement renormalisation.`,
             citation: 'qft-mera' };
  },

  vqc: (st, _step) => {
    const nq = _num(st['n_qubits']);
    const nl = _num(st['n_layers']);
    if (nq === undefined || nl === undefined) return null;
    return { text: `Variational circuit: ${nq} qubits × ${nl} layers. Parameter-shift gradients drive theta toward the target observable.`,
             citation: 'qft-vqc' };
  },

  logic: (st, step) => {
    const te = _num(st['total_energy']);
    if (te === undefined) return null;
    if (te < 0.05) {
      return { text: `Step ${step}: residual = ${te.toFixed(4)} — program has nearly reduced to a normal form.`,
               citation: 'fusion-logic' };
    }
    return { text: `Step ${step}: residual = ${te.toFixed(4)} — relaxation in progress.`,
             citation: 'fusion-logic' };
  },

  mera_relax: (st, step) => {
    const te = _num(st['total_energy']);
    const fp = st['forall_protected_leaves'];
    if (te === undefined) return null;
    const fpStr = Array.isArray(fp) && fp.length
      ? ` ${fp.length} leaves are Forall-protected (held bitwise stable).`
      : '';
    return { text: `Step ${step}: H_eval residual = ${te.toFixed(4)}.${fpStr}`,
             citation: 'fusion-mera-relax' };
  },

  bridge: (st, _step) => {
    const ts = _num(st['trotter_steps']);
    if (ts === undefined) return null;
    return { text: `Bridge ran ${ts} Trotter steps to reach the published ground state.`,
             citation: 'fusion-bridge' };
  },

  'pcn-fields': (st, step) => {
    const layers = st['layers'];
    if (!Array.isArray(layers)) return null;
    return { text: `Step ${step}: ${(layers as unknown[]).length}-layer PCN hierarchy. Top-down predictions, bottom-up errors meet at each layer's Φ.`,
             citation: 'pcn-fields' };
  },

  'pcn-dynamics': (st, step) => {
    const f = _num(st['total_free_energy']);
    if (f === undefined) return null;
    return { text: `Step ${step}: total F = ${f.toFixed(3)} nats. Free energy should monotonically decay if the prior is well-matched.`,
             citation: 'pcn-dynamics' };
  },

  'pcn-coupling': (st, step) => {
    const t = _num(st['mean_abs_stress_energy']);
    const r = _num(st['mean_abs_ricci']);
    if (t === undefined || r === undefined) return null;
    return { text: `Step ${step}: mean|T| = ${t.toExponential(2)} sourcing mean|R| = ${r.toFixed(3)}. PCN error drives QFT geometry; QFT expectations drive PCN targets.`,
             citation: 'fusion-pcn-coupling' };
  },
};
