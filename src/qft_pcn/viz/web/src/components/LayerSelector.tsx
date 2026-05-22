/** Left-rail list of the eight substrate layers; clicking selects one. */

import { LAYER_KEYS } from '../lib/types';
import { useVizStore } from '../store';

export function LayerSelector() {
  const selectedLayer = useVizStore((s) => s.selectedLayer);
  const selectLayer = useVizStore((s) => s.selectLayer);

  return (
    <nav className="layer-selector">
      <h2>Layers</h2>
      <ul>
        {LAYER_KEYS.map((layer) => (
          <li key={layer}>
            <button
              type="button"
              className={layer === selectedLayer ? 'active' : ''}
              onClick={() => selectLayer(layer)}
            >
              {layer}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
