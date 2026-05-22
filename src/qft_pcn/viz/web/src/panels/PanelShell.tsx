/**
 * Shared panel chrome: a dark frame with a title bar (layer name + step) and
 * a body that either shows content or a friendly empty state. Every per-layer
 * panel wraps its visualization in this so they all look consistent.
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
  children: ReactNode;
}

export function PanelShell({
  title,
  step,
  meta,
  hasData,
  emptyMessage = 'No data for this layer at the current step.',
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
      </div>
      <div className="viz-panel__body">
        {hasData ? (
          children
        ) : (
          <div className="viz-panel__empty">{emptyMessage}</div>
        )}
      </div>
    </div>
  );
}
