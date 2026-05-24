// src/qft_pcn/viz/web/src/lib/article-types.ts
/**
 * Types backing the Learn route and the tabbed ExplainerPane.
 *
 * An ArticleSpec is a static, structured document describing one
 * Learn-route chapter. ExplainerSpec.tabs (in lib/explainer.ts) re-uses
 * WorkedExample + Pathology shapes so the same examples can power both
 * the rail tabs and the long-form articles without duplication.
 */

export interface WorkedExample {
  title: string;
  setup: string;
  steps: { description: string; equationId?: string; result: string }[];
  takeaway: string;
}

export interface Pathology {
  signal: string;     // "mean |R| oscillating"
  cause: string;      // "κ_R too large; reduce coupling"
}

export type ArticleSection =
  | { kind: 'prose'; body: string[] }
  | { kind: 'equation'; equationId: string; caption?: string }
  | { kind: 'workedExample'; example: WorkedExample }
  | {
      kind: 'trainingDynamics';
      updateRuleId: string;       // equation id
      expect: string[];           // bullets
      pathologies: Pathology[];
    }
  | { kind: 'miniViz'; layer: string; fixtureFrameId: string }
  | { kind: 'callout'; severity: 'note' | 'warn'; body: string };

export interface ArticleSpec {
  id: string;
  title: string;
  sectionPath: string[];        // ['§3 PCN side', '3.1 Manifold']
  prerequisites?: string[];     // article ids
  sections: ArticleSection[];
  citations: { label: string; href: string }[];
}
