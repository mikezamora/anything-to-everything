/**
 * QPCN panel — three Plotly charts: the scalar energy (descent), a bar chart
 * of per-observable prediction error, and the learnable-parameter values.
 * A panel renders one Frame; the parent re-renders with new frames as the run
 * streams, so each chart reflects the current step.
 *
 * Reads `frame.layer_states.qpcn` (shape: `snapshot_qpcn`).
 */

import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import type { Data as PlotData, Layout as PlotLayout } from 'plotly.js-dist-min';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';

interface QpcnState {
  energy?: number | null;
  pred_errors?: Record<string, number> | null;
  params?: Record<string, number | null> | null;
  bond_dims?: number[] | null;
  entropies?: Array<number | null> | null;
  occupations?: number[] | null;
  step?: number | null;
}

const DARK: Partial<PlotLayout> = {
  paper_bgcolor: '#0b0e14',
  plot_bgcolor: '#0b0e14',
  font: { color: '#8fa8d8', size: 10 },
  margin: { l: 44, r: 12, t: 24, b: 28 },
};

function Chart({
  data,
  layout,
}: {
  data: PlotData[];
  layout: Partial<PlotLayout>;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    Plotly.react(el, data, { ...DARK, ...layout }, {
      displayModeBar: false,
      responsive: true,
    });
    return () => Plotly.purge(el);
  }, [data, layout]);
  return <div ref={ref} style={{ width: '100%', height: '100%' }} />;
}

export function QpcnPanel({ frame }: { frame: Frame }) {
  const st = (frame.layer_states.qpcn ?? {}) as QpcnState;
  const errors = st.pred_errors ?? {};
  const params = st.params ?? {};
  const occ = st.occupations ?? [];
  const hasData =
    st.energy != null ||
    Object.keys(errors).length > 0 ||
    Object.keys(params).length > 0;

  const errKeys = Object.keys(errors);
  const paramKeys = Object.keys(params);

  return (
    <PanelShell
      title="QPCN — energy descent, errors, parameters"
      step={st.step ?? frame.step}
      meta={st.energy != null ? `E = ${st.energy.toFixed(4)}` : undefined}
      hasData={hasData}
    >
      <div
        style={{
          display: 'grid',
          gridTemplateRows: '1fr 1fr',
          gridTemplateColumns: '1fr 1fr',
          gap: 4,
          height: '100%',
        }}
      >
        <Chart
          data={[
            {
              x: ['E'],
              y: [st.energy ?? 0],
              type: 'bar',
              marker: { color: '#d05f8f' },
            },
          ]}
          layout={{
            title: { text: 'Energy', font: { size: 11 } },
            yaxis: { gridcolor: '#1c2230' },
          }}
        />
        <Chart
          data={[
            {
              x: errKeys,
              y: errKeys.map((k) => errors[k]),
              type: 'bar',
              marker: { color: '#d0a05f' },
            },
          ]}
          layout={{
            title: { text: 'Per-observable error', font: { size: 11 } },
            yaxis: { gridcolor: '#1c2230' },
          }}
        />
        <Chart
          data={[
            {
              x: paramKeys,
              y: paramKeys.map((k) => params[k] ?? 0),
              type: 'bar',
              marker: { color: '#5fd0c8' },
            },
          ]}
          layout={{
            title: { text: 'Learnable parameters', font: { size: 11 } },
            yaxis: { gridcolor: '#1c2230' },
          }}
        />
        <Chart
          data={[
            {
              x: occ.map((_, i) => i),
              y: occ,
              type: 'bar',
              marker: { color: '#5f8fd0' },
            },
          ]}
          layout={{
            title: { text: 'Tensor norms (occupation)', font: { size: 11 } },
            xaxis: { title: { text: 'site' }, gridcolor: '#1c2230' },
            yaxis: { gridcolor: '#1c2230' },
          }}
        />
      </div>
    </PanelShell>
  );
}
