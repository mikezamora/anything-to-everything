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

import { useEffect, useMemo, useRef, useState } from 'react';
import Plotly from 'plotly.js-dist-min';
import type { Data as PlotData, Layout as PlotLayout } from 'plotly.js-dist-min';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelReadouts, type ReadoutCell } from './PanelReadouts';
import { MetricsStrip } from './MetricsStrip';
import { useVizStore } from '../store';
import { FrameInterpreter } from '../components/FrameInterpreter';

// The QPCN substrate's Hamiltonian exposes parameters by `<species>.<attr>`
// (e.g. `A.mass`). The viz schema marks the per-attribute slots that are safe
// to mutate live; the panel renders a slider for any snapshot-`params` key
// whose attribute suffix matches one of these.
const WRITABLE_QPCN_ATTRS = new Set(['mass', 'kinetic']);

function attrSuffix(name: string): string {
  const ix = name.indexOf('.');
  return ix >= 0 ? name.slice(ix + 1) : name;
}

interface QpcnState {
  energy?: number | null;
  pred_errors?: Record<string, number> | null;
  params?: Record<string, number | null> | null;
  bond_dims?: number[] | null;
  entropies?: Array<number | null> | null;
  /** Real per-species particle occupations ⟨n_k⟩ (snapshot field
   * `occupations_n`), keyed by species name. The previously rendered
   * `occupations` field was the Frobenius norm of the MPS site tensor
   * (canonical-form gauge), not ⟨n_k⟩; see deviation D-1. */
  occupations_n?: Record<string, number[]> | null;
  /** Canonical-form sanity diagnostic: ‖A[k]‖_F per MPS site. */
  tensor_norms?: number[] | null;
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
  baselineFrame,
}: {
  frame: Frame;
  baselineFrame?: Frame;
}) {
  const st = (frame.layer_states.qpcn ?? {}) as QpcnState;
  const bst = (baselineFrame?.layer_states.qpcn ?? {}) as QpcnState;
  const errors = (st.pred_errors ?? {}) as Record<string, number>;
  const baseErrors = (bst.pred_errors ?? {}) as Record<string, number>;
  const hasBaseline = !!baselineFrame;
  const params = (st.params ?? {}) as Record<string, number>;
  const occN = (st.occupations_n ?? {}) as Record<string, number[]>;
  const occSpecies = Object.keys(occN);
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

  // One grouped bar series per species, x = site index, y = ⟨n_k⟩.
  // Honest physical occupation per §3.3.4 (was the MPS tensor Frobenius
  // norm before the D-1 fix).
  const occColors = ['#5f8fd0', '#5fd0c8', '#d0a05f', '#a05fd0', '#d05f8f'];
  const occData = useMemo<PlotData[]>(
    () =>
      occSpecies.map((sp, i) => ({
        x: occN[sp].map((_, k) => k),
        y: occN[sp],
        type: 'bar' as const,
        name: sp,
        marker: { color: occColors[i % occColors.length] },
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [occSpecies.join('|'), occSpecies.map((s) => occN[s].join(',')).join('|')],
  );
  const occLayout = useMemo<Partial<PlotLayout>>(
    () => ({
      title: { text: '⟨n_k⟩ per site (per species)', font: { size: 11 } },
      xaxis: { title: { text: 'site' }, gridcolor: '#1c2230' },
      yaxis: { title: { text: '⟨n⟩' }, gridcolor: '#1c2230' },
      barmode: 'group',
      showlegend: occSpecies.length > 1,
      legend: { font: { size: 9 } },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [occSpecies.length],
  );

  const energyText =
    st.energy != null ? st.energy.toFixed(4) : '(unknown)';
  const baselineEnergy =
    hasBaseline && bst.energy != null && Number.isFinite(bst.energy)
      ? bst.energy
      : null;
  const energyDelta =
    baselineEnergy != null && st.energy != null
      ? st.energy - baselineEnergy
      : null;
  const energyDeltaColor =
    energyDelta == null
      ? undefined
      : energyDelta < 0
        ? '#9aedc1'
        : energyDelta > 0
          ? '#ef9090'
          : '#7f8bb0';

  const errorEntries = Object.entries(errors);
  const errorVals = errorEntries.map(([, v]) => v).filter((v) => Number.isFinite(v));
  const meanPredErr =
    errorVals.length > 0
      ? errorVals.reduce((a, b) => a + Math.abs(b), 0) / errorVals.length
      : null;
  const baseErrorVals = Object.values(baseErrors).filter((v) => Number.isFinite(v));
  const baseMeanPredErr =
    baseErrorVals.length > 0
      ? baseErrorVals.reduce((a, b) => a + Math.abs(b), 0) / baseErrorVals.length
      : null;

  // PanelReadouts strip — the highlightIds here match the `readout: 'energy'`
  // and `readout: 'pred_errors'` references in `lib/explainer.ts`, so that
  // hovering an explainer watch item flashes the matching cell.
  const readoutCells: ReadoutCell[] = [
    {
      label: 'energy',
      value: energyText,
      highlightId: 'energy',
      baselineValue: baselineEnergy,
    },
    {
      label: 'pred_err_count',
      value: errorEntries.length,
      highlightId: 'pred_errors',
    },
  ];
  if (meanPredErr != null) {
    readoutCells.push({
      label: 'mean |pred_err|',
      value: meanPredErr.toFixed(4),
      highlightId: 'pred_errors',
      baselineValue: baseMeanPredErr,
    });
  }

  const readouts = (
    <div className="qpcn-readouts">
      <PanelReadouts cells={readoutCells} />
      {/* Preserve the legacy energy Δ rendering for back-compat with the
       * existing `.qpcn-energy-delta` test hook. The PanelReadouts cell
       * already shows the Δ in compare mode; this hidden span keeps the
       * targeted CSS selector live. */}
      {energyDelta != null && (
        <small
          className="qpcn-energy-delta"
          style={{ color: energyDeltaColor, marginLeft: 4, fontSize: 11 }}
        >
          ({energyDelta >= 0 ? '+' : ''}
          {energyDelta.toFixed(4)})
        </small>
      )}
    </div>
  );

  // Honest empty-state note when no observation targets are emitted.
  const noTargets = st.energy == null && errorEntries.length === 0;
  const emptyTargetsNote = (
    <div
      className="qpcn-no-targets-note"
      style={{ fontSize: 11, color: '#7f8bb0', padding: '6px 4px' }}
    >
      No observation targets emitted for this preset. Try
      {' '}
      <code style={{ color: '#9aa6c8' }}>qpcn.quarter-density-target</code>
      {' '}
      for a run with live ⟨n⟩ targets.
    </div>
  );

  const predErrorsTable = errorEntries.length > 0 ? (
    <table
      className="qpcn-errors"
      style={{ borderCollapse: 'collapse', fontSize: 11 }}
    >
      <thead>
        <tr>
          <th style={{ textAlign: 'left', paddingRight: 12 }}>obs</th>
          <th style={{ textAlign: 'right' }}>error</th>
          {hasBaseline && <th style={{ textAlign: 'right', paddingLeft: 12 }}>Δ</th>}
        </tr>
      </thead>
      <tbody>
        {errorEntries.map(([k, v]) => {
          const bv = baseErrors[k];
          const d = hasBaseline && typeof bv === 'number' && Number.isFinite(bv)
            ? v - bv
            : null;
          const dColor =
            d == null
              ? undefined
              : d < 0
                ? '#9aedc1'
                : d > 0
                  ? '#ef9090'
                  : '#7f8bb0';
          return (
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
              {hasBaseline && (
                <td
                  className="qpcn-error-delta"
                  style={{
                    textAlign: 'right',
                    paddingLeft: 12,
                    color: dColor,
                  }}
                >
                  {d == null ? '—' : `${d >= 0 ? '+' : ''}${d.toFixed(4)}`}
                </td>
              )}
            </tr>
          );
        })}
      </tbody>
    </table>
  ) : null;

  const metrics = useMemo(
    () => [
      {
        key: 'E',
        label: 'energy',
        color: '#fbc66a',
        select: (ls: Record<string, unknown>) =>
          ls.energy as number | null | undefined,
      },
      ...paramKeys.map((p, i) => ({
        key: `p:${p}`,
        label: p,
        color: ['#6cd0ff', '#9aedc1', '#d291ff'][i % 3],
        select: (ls: Record<string, unknown>) =>
          (ls.params as Record<string, number> | undefined)?.[p],
      })),
    ],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [paramKeys.join('|')],
  );

  const metricsStrip = <MetricsStrip layer="qpcn" metrics={metrics} />;

  // ---- live-param sliders (visible only when paused on an active run) -----
  const paused = useVizStore((s) => s.paused);
  const activeRunId = useVizStore((s) => s.activeRunId);
  const setError = useVizStore((s) => s.setError);
  const writableKeys = paramKeys.filter((k) =>
    WRITABLE_QPCN_ATTRS.has(attrSuffix(k)),
  );
  const [localVals, setLocalVals] = useState<Record<string, number>>({});

  const showSliders = paused && !!activeRunId && writableKeys.length > 0;

  async function onSlider(name: string, value: number) {
    setLocalVals((v) => ({ ...v, [name]: value }));
    if (!activeRunId) return;
    try {
      // Lazy import keeps the panel test-friendly: transport.ts is mocked
      // at the module boundary via the global fetch stub the test installs.
      const { transport } = await import('../lib/transport');
      await transport.setParams(activeRunId, { qpcn: { [name]: value } });
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const sliders = showSliders ? (
    <div
      data-testid="qpcn-param-sliders"
      style={{ display: 'flex', flexDirection: 'column', gap: 4, padding: 4 }}
    >
      {writableKeys.map((k) => {
        const cur = localVals[k] ?? params[k] ?? 0;
        return (
          <label
            key={k}
            style={{ display: 'flex', alignItems: 'center', gap: 6,
                     fontSize: 11, color: '#9aa6c8' }}
          >
            <span style={{ minWidth: 70 }}>{k}</span>
            <input
              type="range"
              min={0}
              max={5}
              step={0.05}
              value={cur}
              onChange={(e) => onSlider(k, parseFloat(e.target.value))}
              data-testid={`qpcn-slider-${k}`}
            />
            <span style={{ minWidth: 48, textAlign: 'right' }}>
              {cur.toFixed(2)}
            </span>
          </label>
        );
      })}
    </div>
  ) : null;

  return (
    <PanelShell
      title="QPCN — energy descent, errors, parameters"
      step={st.step ?? frame.step}
      meta={st.energy != null ? `E = ${st.energy.toFixed(4)}` : undefined}
      hasData={hasData}
      readouts={readouts}
      metricsStrip={metricsStrip}
    >
      <FrameInterpreter layer="qpcn" />
      {noTargets ? emptyTargetsNote : predErrorsTable}
      <div
        style={{
          display: 'grid',
          gridTemplateRows: '1fr',
          gridTemplateColumns: '1fr 1fr',
          gap: 4,
          height: '100%',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <Chart data={paramData} layout={paramLayout} />
          {sliders}
        </div>
        <Chart data={occData} layout={occLayout} />
      </div>
    </PanelShell>
  );
}
