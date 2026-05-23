/**
 * Left rail: three collapsible sections (QFT / PCN / QPCN) each containing
 * its layers. Section header is itself clickable: it selects the matching
 * intro pseudo-layer (intro-qft / intro-pcn / intro-qpcn) so the explainer
 * panel area renders the section narrative.
 */

import { useState } from 'react';
import { SECTIONS } from '../lib/sections';
import { useVizStore } from '../store';

export function SectionedLayerSelector() {
  const selected = useVizStore((s) => s.selectedLayer);
  const select = useVizStore((s) => s.selectLayer);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  return (
    <nav className="layer-selector">
      <h2>Layers</h2>
      {SECTIONS.map((sec) => {
        const isCollapsed = collapsed[sec.key] ?? false;
        const introKey = `intro-${sec.key}`;
        return (
          <div key={sec.key} className={
            `layer-section section-${sec.key}${isCollapsed ? ' collapsed' : ''}`
          }>
            <div className="layer-section-header">
              <button
                type="button"
                className={selected === introKey
                  ? 'section-title active' : 'section-title'}
                onClick={() => select(introKey)}
              >
                {sec.title}
              </button>
              <button
                type="button"
                className="section-collapse-toggle"
                aria-label={isCollapsed ? 'Expand section' : 'Collapse section'}
                onClick={() => setCollapsed((c) => ({ ...c, [sec.key]: !isCollapsed }))}
              >
                {isCollapsed ? '▸' : '▾'}
              </button>
            </div>
            {!isCollapsed && (
              <ul>
                {sec.layers.map((layer) => (
                  <li key={layer}>
                    <button
                      type="button"
                      className={layer === selected ? 'active' : ''}
                      onClick={() => select(layer)}
                    >
                      {layer}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        );
      })}
    </nav>
  );
}
