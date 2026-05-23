/**
 * QPCN panel — energy readout, a per-observable prediction-error table, a
 * bar chart of learnable parameter values, and a MetricsStrip tracking the
 * scalar energy + each learnable parameter across the run.
 *
 * Reads `frame.layer_states.qpcn` (shape: `snapshot_qpcn`).
 *
 * Note: writable-param sliders are intentionally NOT exposed — the substrate
 * setter for QPCN params is not surfaced here (see EXTENSIONS.md). The
 * prediction-error "target" column is also omitted because snapshots don't
 * carry target values; we render `(unknown)` in the readout meta instead of
 * inventing one.
 */

import { useEffect, useMemo, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import type { Data as PlotData, Layout as PlotLayout } from 'plotly.js-dist-min';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { MetricsStrip } from './MetricsStrip';

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

export function QpcnPanel({
  frame,
}: {
  frame: Frame;
  baselineFrame?: Frame;
}) {
  const st = (frame.layer_states.qpcn ?? {}) as QpcnState;
  const errors = (st.pred_errors ?? {}) as Record<string, number>;
  const params = (st.params ?? {}) as Record<string, number>;
  const occ = st.occupations ?? [];
  const hasData =
    st.energy != null ||
    Object.keys(errors).length > 0 ||
    Object.keys(params).length > 0;

  const paramKeys = Object.keys(params);

  const paramVals = paramKeys.map((k) => params[k] ?? 0);
  const paramData = useMemo<PlotData[]>(
    () => [
      {
        x: paramKeys,
        y: paramVals,
        type: 'bar',
        marker: { color: '#5fd0c8' },
      },
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [paramKeys.join('|'), paramVals.join('|')],
  );
  const paramLayout = useMemo<Partial<PlotLayout>>(
    () => ({
      title: { text: 'Learnable parameters', font: { size: 11 } },
      yaxis: { gridcolor: '#1c2230' },
    }),
    [],
  );

  const occData = useMemo<PlotData[]>(
    () => [
      {
        x: occ.map((_, i) => i),
        y: occ,
        type: 'bar',
        marker: { color: '#5f8fd0' },
      },
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [occ.join('|')],
  );
  const occLayout = useMemo<Partial<PlotLayout>>(
    () => ({
      title: { text: 'Tensor norms (occupation)', font: { size: 11 } },
      xaxis: { title: { text: 'site' }, gridcolor: '#1c2230' },
      yaxis: { gridcolor: '#1c2230' },
    }),
    [],
  );

  const energyText =
    st.energy != null ? st.energy.toFixed(4) : '(unknown)';

  const readouts = (
    <div className="qpcn-readouts" style={{ display: 'flex', gap: 16 }}>
      <div>
        <span style={{ color: '#7f8bb0', marginRight: 6 }}>energy</span>
        <span style={{ color: '#fbc66a' }}>{energyText}</span>
      </div>
      <table
        className="qpcn-errors"
        style={{ borderCollapse: 'collapse', fontSize: 11 }}
      >
        <thead>
          <tr>
            <th style={{ textAlign: 'left', paddingRight: 12 }}>obs</th>
            <th style={{ textAlign: 'right' }}>error</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(errors).map(([k, v]) => (
            <tr key={k}>
              <td style={{ paddingRight: 12 }}>{k}</td>
              <td
                style={{
                  textAlign: 'right',
                  color: v >= 0 ? '#ef9090' : '#9aedc1',
                }}
              >
                {v.toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  const metricsStrip = (
    <MetricsStrip
      layer="qpcn"
      metrics={[
        {
          key: 'E',
          label: 'energy',
          color: '#fbc66a',
          select: (ls) => ls.energy as number,
        },
        ...Object.keys(
          (st.params as Record<string, number>) ?? {},
        ).map((p, i) => ({
          key: `p:${p}`,
          label: p,
          color: ['#6cd0ff', '#9aedc1', '#d291ff'][i % 3],
          select: (ls) =>
            (ls.params as Record<string, number> | undefined)?.[p] as number,
        })),
      ]}
    />
  );

  return (
    <PanelShell
      title="QPCN — energy descent, errors, parameters"
      step={st.step ?? frame.step}
      meta={st.energy != null ? `E = ${st.energy.toFixed(4)}` : undefined}
      hasData={hasData}
      readouts={readouts}
      metricsStrip={metricsStrip}
    >
      <div
        style={{
          display: 'grid',
          gridTemplateRows: '1fr',
          gridTemplateColumns: '1fr 1fr',
          gap: 4,
          height: '100%',
        }}
      >
        <Chart data={paramData} layout={paramLayout} />
        <Chart data={occData} layout={occLayout} />
      </div>
    </PanelShell>
  );
}
