/**
 * Shared panel chrome: a dark frame with a title bar (layer name + step) and
 * a body that either shows content or a friendly empty state. Every per-layer
 * panel wraps its visualization in this so they all look consistent.
 *
 * Optional additive props:
 *   - isExtension/extensionAnchor: render an "extension pending" badge
 *     linking into EXTENSIONS.md.
 *   - toolbar / readouts / metricsStrip: slots rendered above / below the
 *     main body, used by uplifted panels to add controls & sparklines.
 */

import type { ReactNode } from 'react';
import './panel.css';

export interface PanelShellProps {
  title: string;
  step?: number;
  meta?: string;
  /** When false, the body is replaced with an empty-state message. */
  hasData: boolean;
  emptyMessage?: string;
  /** When true, renders an "extension pending" badge + tooltip link. */
  isExtension?: boolean;
  extensionAnchor?: string;
  toolbar?: ReactNode;
  readouts?: ReactNode;
  metricsStrip?: ReactNode;
  children: ReactNode;
}

export function PanelShell({
  title,
  step,
  meta,
  hasData,
  emptyMessage = 'No data for this layer at the current step.',
  isExtension,
  extensionAnchor,
  toolbar,
  readouts,
  metricsStrip,
  children,
}: PanelShellProps) {
  return (
    <div className="viz-panel">
      <div className="viz-panel__header">
        <span className="viz-panel__title">{title}</span>
        {step !== undefined && (
          <span className="viz-panel__meta">step {step}</span>
        )}
        {meta && <span className="viz-panel__meta">{meta}</span>}
        {isExtension && (
          <a
            className="viz-panel__badge"
            href={`../../EXTENSIONS.md${extensionAnchor ?? ''}`}
            title="This panel is fixture-only; see EXTENSIONS.md"
          >
            extension pending
          </a>
        )}
      </div>
      {toolbar && <div className="viz-panel__toolbar">{toolbar}</div>}
      <div className="viz-panel__body">
        {hasData ? (
          children
        ) : (
          <div className="viz-panel__empty">{emptyMessage}</div>
        )}
      </div>
      {readouts && <div className="viz-panel__readouts">{readouts}</div>}
      {metricsStrip && (
        <div className="viz-panel__metrics">{metricsStrip}</div>
      )}
    </div>
  );
}
