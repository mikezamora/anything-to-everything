/**
 * Panel registry — maps each layer name in `LAYER_KEYS` to the React component
 * that visualizes it. `App.tsx` looks up the selected layer here and renders
 * the matching panel with the current `Frame`.
 */

import type { ComponentType } from 'react';
import type { Frame, LayerKey } from '../lib/types';
import { ManifoldPanel } from './ManifoldPanel';
import { MultifieldPanel } from './MultifieldPanel';
import { MpsPanel } from './MpsPanel';
import { HamiltonianPanel } from './HamiltonianPanel';
import { QpcnPanel } from './QpcnPanel';
import { MeraPanel } from './MeraPanel';
import { VqcPanel } from './VqcPanel';
import { LogicPanel } from './LogicPanel';
import { MeraRelaxPanel } from './MeraRelaxPanel';
import { BridgePanel } from './BridgePanel';
import { PcnFieldsPanel } from './PcnFieldsPanel';

export type PanelComponent = ComponentType<{ frame: Frame; baselineFrame?: Frame }>;

export const PANELS: Record<LayerKey, PanelComponent> = {
  manifold: ManifoldPanel,
  multifield: MultifieldPanel,
  mps: MpsPanel,
  hamiltonian: HamiltonianPanel,
  qpcn: QpcnPanel,
  mera: MeraPanel,
  vqc: VqcPanel,
  logic: LogicPanel,
  mera_relax: MeraRelaxPanel,
  bridge: BridgePanel,
  'pcn-fields': PcnFieldsPanel,
} as Record<LayerKey, PanelComponent>;

/** Look up a panel by layer name; `undefined` for an unknown layer. */
export function panelFor(layer: string): PanelComponent | undefined {
  return PANELS[layer as LayerKey];
}
