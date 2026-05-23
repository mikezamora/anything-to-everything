/**
 * Panel registry — maps each layer name in `LAYER_KEYS` to the React component
 * that visualizes it. `App.tsx` looks up the selected layer here and renders
 * the matching panel with the current `Frame`.
 */

import type { ComponentType } from 'react';
import type { Frame } from '../lib/types';
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
import { PcnDynamicsPanel } from './PcnDynamicsPanel';
import { PcnCouplingPanel } from './PcnCouplingPanel';
import { IntroQftPanel } from './IntroQftPanel';
import { IntroPcnPanel } from './IntroPcnPanel';
import { IntroQpcnPanel } from './IntroQpcnPanel';

export type PanelComponent = ComponentType<{ frame: Frame; baselineFrame?: Frame }>;

export const PANELS: Record<string, PanelComponent> = {
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
  'pcn-dynamics': PcnDynamicsPanel,
  'pcn-coupling': PcnCouplingPanel,
  'intro-qft': IntroQftPanel,
  'intro-pcn': IntroPcnPanel,
  'intro-qpcn': IntroQpcnPanel,
};

/** Look up a panel by layer name; `undefined` for an unknown layer. */
export function panelFor(layer: string): PanelComponent | undefined {
  return PANELS[layer];
}
