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
};
