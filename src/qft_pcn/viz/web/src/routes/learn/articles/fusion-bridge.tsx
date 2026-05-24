// src/qft_pcn/viz/web/src/routes/learn/articles/fusion-bridge.tsx
/**
 * §4.6 Fusion — the Bridge layer. RunResult inspector, one-shot resolver
 * vs per-frame stepping, how the viz wraps run_problem into a layer of
 * snapshots the panels consume.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'fusion-bridge',
  title: 'Fusion: the Bridge — RunResult, snapshots, and the per-frame loop',
  sectionPath: ['§4 QPCN Fusion', '4.6 Bridge'],
  sections: [
    {
      kind: 'prose',
      body: [
        'The Bridge is the layer that connects the Python substrate (manifold + PCN + QPCN + logic Hamiltonian) to the visualisation. It is the smallest piece of code that has to know about both worlds. On the Python side, the substrate exposes a `run_problem(spec) -> RunResult` entry point that takes a RunSpec (parsed from a DSL or a preset) and produces a stream of per-frame snapshots. On the viz side, the panels consume those snapshots and render them. The Bridge is the glue: it turns the resolver call into an iterable, wraps each snapshot into a typed JSON payload, and streams them over WebSocket to the browser.',
        'There are two modes of operation, and the choice between them is one of the most important configuration knobs. **One-shot resolution** runs `run_problem` to completion, collects every frame into a list, and ships the whole RunResult at once — the panels then scrub through the frames with a timeline slider. This is ideal for completed runs you want to inspect repeatedly. **Per-frame stepping** runs the substrate one frame at a time, shipping each snapshot as soon as it is produced — the panels then update live as the run progresses. This is ideal for long runs where you want to watch convergence happen.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'Both modes use the same Bridge code; only the streaming policy differs. A `RunResult` is just a typed list of `Snapshot` objects, each carrying the full per-layer public state for one frame: PCN beliefs and errors, QFT MPS tensors (or their summaries), Hamiltonian parameters, manifold metric, Ricci field, all the per-rule residuals, and metadata (frame number, run id, wallclock). The schema is large but flat; the panels each pick out the fields they care about.',
        'The most expensive part of producing a Snapshot is not the substrate computation — it is the *extraction*. The MPS is internally a list of `(2, chi, chi, 2)` complex tensors; the panel doesn\'t need them all, just summaries (entanglement entropy per cut, leading basis amplitudes, expectation values). Similarly the manifold metric is a `(Nx, Ny, 2, 2)` array; the panel needs the Ricci scalar (`(Nx, Ny)`) and a downsampled `h`. The Bridge runs the extractor functions and packs the results, dropping precision where the visualisation can afford it.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'free-energy-functional',
      caption: 'F is one of the headline numbers shipped in every Snapshot — the Bridge\'s "is this run healthy?" indicator.',
    },
    {
      kind: 'prose',
      body: [
        'The per-frame loop on the Python side runs in sequence: (1) PCN steps, (2) stress-energy update of the metric, (3) Laplace-Beltrami diffusion, (4) one Trotter step of QPCN imag-time, (5) operator readouts and parameter update, (6) snapshot extraction. The §T Training article walks through all six steps in detail with the actual frame numbers from a default preset; this article focuses on what the Bridge does *after* step 6 — packing, streaming, and timeline assembly on the viz side.',
        'A subtle but important wrinkle: `trotter_steps` is reported in the Snapshot only at frame 0. It is a property of the RunSpec (how many Trotter steps per frame the QPCN is taking), not a per-frame value, so the Bridge omits it from later snapshots to save bandwidth. The panels read it from frame 0 and cache it. If you join a stream late and miss frame 0, the panels fall back to a "trotter_steps unknown" badge.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'imag-time-evolution',
      caption: 'Step (4) of the per-frame loop — the QPCN takes trotter_steps of imag-time per frame; this is what the Bridge reports once at frame 0.',
    },
    {
      kind: 'prose',
      body: [
        'The RunResult inspector panel (§4.6) lets you scrub a one-shot RunResult frame by frame, with every layer\'s extracted snapshot exposed as a JSON tree on the right and a layer-aware visualisation on the left. It is the canonical "what does the system look like right now?" tool. The same panel can run in live mode (per-frame stepping) — the JSON tree updates as snapshots arrive, and there is a play/pause/scrub control that lets you freeze the stream and inspect any past frame from the buffer.',
        'Reading the inspector is mostly mechanical: navigate the JSON tree to find the field you want, hover for the type annotation, and use the time slider to compare across frames. The most useful workflow is to pin a field in one frame and a different field in another, then scrub the slider — the panel highlights the deltas.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'coupling-descent',
      caption: 'The coupling-descent step (5) — the parameter trajectory is one of the most-watched fields in the inspector, since it shows whether learning is happening.',
    },
    {
      kind: 'prose',
      body: [
        '**The bridge\'s role in the overall pipeline.** The Bridge occupies a very specific seat between the upstream DSL/LLM emission layer and the downstream substrate execution. Upstream, an LLM (or a preset, or a hand-written file) produces a DSL payload describing a run — what fields, what Hamiltonian, what observables, what coupling constants. That payload is parsed and validated against the DSL schema by code that does not know anything about QFTs or PCNs. Downstream, the substrate executes the validated RunSpec and produces a stream of Snapshots; that code does not know anything about WebSockets, JSON serialisation, or browsers. The Bridge sits exactly between these two and is the only layer that knows about both.',
        'This sandwich position has architectural consequences. The Bridge is the *contract surface* of the whole system: every panel, every external client, every preset, every LLM-emitted DSL run flows through it. If the contract is well-defined here, the upstream and downstream can evolve independently — new panels can be added without touching the substrate, and new substrate features (a new field, a new Hamiltonian term) only ripple outward if the Bridge\'s Snapshot schema needs to grow. The pipeline diagram for any run is therefore: `LLM/preset -> DSL string -> RunSpec -> [Bridge resolves] -> Snapshot stream -> WebSocket -> panels`. The Bridge is the only `[ ... ]` step — everything else is mechanical translation.',
      ],
    },
    {
      kind: 'prose',
      body: [
        '**One-shot vs streamed resolution: when each is appropriate.** The two streaming modes are not interchangeable; they target different workflows. **One-shot** is appropriate when (i) the run is short enough to fit comfortably in memory (default presets are O(200) frames at O(10 KB) each, so ~2 MB — fine); (ii) the user wants to scrub *backwards* as well as forwards through the timeline, which only the buffered-all-frames mode supports cheaply; (iii) the run is reproducible and the user is comparing two completed runs side-by-side. **Streamed** (per-frame) is appropriate when (i) the run is long enough that buffering everything is wasteful or impossible; (ii) the user wants to watch convergence happen *live*, which is genuinely informative for diagnosing pathologies (oscillating `<H>`, runaway parameters, stuck residuals — these show up better in motion than in a static plot); (iii) the run is exploratory and the user may want to abort early.',
        'A deliberate v1 choice: **bridge runs from emitted DSL are always one-shot.** When an LLM emits a DSL fragment via the §5 DSL channel and the Bridge resolves it, the resulting RunResult is delivered as a single payload, not streamed. The reason is contractual: a streamed run is a *live process* that can be inspected, paused, and reasoned about by the LLM only with significant coordination machinery (cursors, ack protocols, partial-result schemas) that v1 does not have. A one-shot RunResult is a *value* — completed, immutable, addressable — and that is exactly the kind of thing an LLM can pass around, cite, and reason about in a single turn. The streamed mode remains available for interactive panel use, where a human is the consumer; the one-shot mode is what the agent-facing path uses, and the asymmetry is intentional.',
      ],
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Trace `physics-relax` preset through bridge -> builders -> snapshot',
        setup:
          'Select the `physics-relax` preset and start a run. Trace the path from preset name to the first frame\'s snapshot, identifying each layer the data flows through.',
        steps: [
          {
            description:
              'User clicks "physics-relax" in the preset picker. The viz sends a WebSocket message `{cmd: "start-run", preset: "physics-relax"}` to the Bridge.',
            result: 'Bridge receives start-run message with preset = "physics-relax".',
          },
          {
            description:
              'The Bridge looks up the preset in its preset registry, retrieves the canonical DSL string for physics-relax, and calls dsl_to_runspec(dsl_string) to produce a typed RunSpec. The RunSpec includes fields, hamiltonian terms, observables, and run-config (frames, trotter_steps, kappa_R, chi_max).',
            result: 'RunSpec produced; e.g. for physics-relax: fields = [phi], hamiltonian = [kinetic, mass, lambda_phi4], observables = [<phi>, <phi^2>], frames = 200, trotter_steps = 4.',
          },
          {
            description:
              'The Bridge calls run_problem(spec) — but in per-frame mode, this is a generator that yields one Snapshot at a time. The Python substrate builders fire in order: _build_manifold(spec) constructs the (Nx, Ny, 2, 2) metric grid, _build_network(spec) wires up the PCN layers, _build_qpcn(spec) builds the MPS and Hamiltonian. All three are now live in-memory.',
            result: 'Substrate built: manifold (flat), PCN (zero beliefs), QPCN (random MPS).',
            equationId: 'free-energy-functional',
          },
          {
            description:
              'Frame 0 runs the per-frame loop once. PCN beliefs initialise, error field is computed, stress-energy is zero (no errors yet on a fresh prior). Manifold stays flat. QPCN takes 4 Trotter steps of imag-time, MPS relaxes a bit. Operator expectations are read out. The snapshot extractor packs: PCN state, manifold (flat, with metadata), QPCN summaries, Hamiltonian params, frame metadata. Also packs trotter_steps = 4 (frame-0-only field).',
            result: 'Frame 0 snapshot built; first WebSocket payload shipped.',
            equationId: 'imag-time-evolution',
          },
          {
            description:
              'Viz receives the WebSocket frame, decodes the JSON, dispatches to all subscribed panels. Each panel (manifold, PCN, QPCN, parameters, free-energy trace) picks its own fields from the snapshot and renders. The user sees the initial visualisation.',
            result: 'First-frame visualisation rendered. Bridge continues per-frame stepping for frames 1..199.',
            equationId: 'coupling-descent',
          },
        ],
        takeaway:
          'A preset selection becomes a DSL string, becomes a RunSpec, becomes a substrate, becomes a stream of Snapshots, becomes a stream of WebSocket payloads, becomes panel renders. The Bridge is the single layer that orchestrates this — preset -> snapshot is its job. The §4.6 panel exposes every step of this trace in its developer tools tab, including the raw RunSpec, the latest snapshot JSON, and the WebSocket framing.',
      },
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'free-energy-functional',
      expect: [
        'trotter_steps reported once at frame 0; absent from later frames.',
        'Snapshot extraction wallclock is sub-frame: the bridge does not become the bottleneck.',
        'WebSocket payload sizes shrink after frame 0 (no trotter_steps, no preset-init metadata).',
        'Late joiners (subscribing after frame 0) get a "trotter_steps unknown" badge — expected.',
      ],
      pathologies: [
        { signal: 'Frames arrive at irregular intervals', cause: 'Snapshot extraction is dominating; the substrate is faster than the bridge. Reduce extracted field set or downsample.' },
        { signal: 'trotter_steps missing on frame 0', cause: 'Bridge frame-0-only metadata logic broken. Check the snapshot packer\'s frame-number branch.' },
        { signal: 'Run never starts after preset select', cause: 'dsl_to_runspec failure or builder exception. Check the bridge log for tracebacks.' },
        { signal: 'JSON tree shows null fields', cause: 'Extractor function returned None for some layer (often the QPCN if chi blew up). Inspect the Python-side log.' },
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §11 (Viz bridge)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/viz/bridge.py', href: '../../bridge.py' },
  ],
};
