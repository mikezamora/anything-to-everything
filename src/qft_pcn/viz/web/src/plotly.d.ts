/**
 * Minimal ambient declaration for `plotly.js-dist-min`, which ships no types.
 * We model only the surface the panels use: `newPlot` / `react` / `purge` /
 * `relayout`, plus loose `Data` / `Layout` aliases so callers stay typed.
 */
declare module 'plotly.js-dist-min' {
  // Loose but non-`any` aliases — panel code casts its trace objects to these.
  export type Data = Record<string, unknown>;
  export type Layout = Record<string, unknown>;
  export type Config = Record<string, unknown>;

  interface PlotlyStatic {
    newPlot(
      root: HTMLElement,
      data: Data[],
      layout?: Partial<Layout>,
      config?: Partial<Config>,
    ): Promise<void>;
    react(
      root: HTMLElement,
      data: Data[],
      layout?: Partial<Layout>,
      config?: Partial<Config>,
    ): Promise<void>;
    relayout(root: HTMLElement, layout: Partial<Layout>): Promise<void>;
    purge(root: HTMLElement): void;
  }

  const Plotly: PlotlyStatic;
  export default Plotly;
}
