/**
 * Zustand store: per-run frame ring buffers + scrub + baseline + transport.
 *
 * `runs` is a Map keyed by opaque run id. Each `RunState` is a capped frame
 * ring buffer + a cursor. `activeRunId` is what panels render against; if
 * `baselineRunId` is set, panels also receive its current frame for diff.
 * `paused` and `playbackSpeed` are global transport state — applied
 * client-side to the live frame pump.
 */

import { create } from 'zustand';
import type { ChatTurn, Frame, Route } from './lib/types';

export const MAX_FRAMES = 2000;

export interface RunState {
  frames: Frame[];
  cursor: number;
  live: boolean;
}

interface VizState {
  runs: Map<string, RunState>;
  activeRunId: string | null;
  baselineRunId: string | null;
  selectedLayer: string;
  paused: boolean;
  playbackSpeed: number;
  error: string | null;
  route: Route;
  chat: ChatTurn[];
  dslText: string;
  llmModel: string | null;
  steppedMode: boolean;

  openRun: (id: string) => void;
  setActiveRun: (id: string) => void;
  pinBaseline: (id: string) => void;
  unpinBaseline: () => void;

  pushFrame: (runId: string, frame: Frame) => void;
  setCursor: (runId: string, cursor: number) => void;
  setLive: (runId: string, live: boolean) => void;

  selectLayer: (layer: string) => void;
  setPaused: (paused: boolean) => void;
  setPlaybackSpeed: (speed: number) => void;
  setError: (err: string | null) => void;

  setRoute: (r: Route) => void;
  appendChat: (turn: ChatTurn) => void;
  setDslText: (t: string) => void;
  setLlmModel: (m: string | null) => void;
  setSteppedMode: (v: boolean) => void;

  currentFrame: (runId?: string | null) => Frame | undefined;
  baselineFrame: () => Frame | undefined;
  resetAll: () => void;
}

const initRun = (): RunState => ({ frames: [], cursor: 0, live: true });

export const useVizStore = create<VizState>((set, get) => ({
  runs: new Map(),
  activeRunId: null,
  baselineRunId: null,
  selectedLayer: 'manifold',
  paused: false,
  playbackSpeed: 1,
  error: null,
  route: 'viz',
  chat: [],
  dslText: '',
  llmModel: null,
  steppedMode: false,

  openRun: (id) => set((s) => {
    if (s.runs.has(id)) return {};
    const next = new Map(s.runs);
    next.set(id, initRun());
    return { runs: next, activeRunId: s.activeRunId ?? id };
  }),

  setActiveRun: (id) => set({ activeRunId: id }),
  pinBaseline: (id) => set({ baselineRunId: id }),
  unpinBaseline: () => set({ baselineRunId: null }),

  pushFrame: (runId, frame) => set((s) => {
    const cur = s.runs.get(runId);
    if (!cur) return {};
    let frames = [...cur.frames, frame];
    if (frames.length > MAX_FRAMES) {
      frames = frames.slice(frames.length - MAX_FRAMES);
    }
    const cursor = cur.live
      ? frames.length - 1
      : Math.min(cur.cursor, frames.length - 1);
    const next = new Map(s.runs);
    next.set(runId, { ...cur, frames, cursor });
    return { runs: next };
  }),

  setCursor: (runId, cursor) => set((s) => {
    const cur = s.runs.get(runId);
    if (!cur) return {};
    const max = Math.max(0, cur.frames.length - 1);
    const next = new Map(s.runs);
    next.set(runId, {
      ...cur,
      cursor: Math.max(0, Math.min(cursor, max)),
      live: false,
    });
    return { runs: next };
  }),

  setLive: (runId, live) => set((s) => {
    const cur = s.runs.get(runId);
    if (!cur) return {};
    const next = new Map(s.runs);
    next.set(runId, { ...cur, live });
    return { runs: next };
  }),

  selectLayer: (layer) => set({ selectedLayer: layer }),
  setPaused: (paused) => set({ paused }),
  setPlaybackSpeed: (speed) => set({ playbackSpeed: Math.max(0.25, speed) }),
  setError: (error) => set({ error }),

  setRoute: (route) => set({ route }),
  appendChat: (turn) => set((s) => ({ chat: [...s.chat, turn] })),
  setDslText: (dslText) => set({ dslText }),
  setLlmModel: (llmModel) => set({ llmModel }),
  setSteppedMode: (steppedMode) => set({ steppedMode }),

  currentFrame: (runId) => {
    const id = runId ?? get().activeRunId;
    if (!id) return undefined;
    const r = get().runs.get(id);
    return r?.frames[r.cursor];
  },

  baselineFrame: () => {
    const id = get().baselineRunId;
    if (!id) return undefined;
    const r = get().runs.get(id);
    return r?.frames[r.cursor];
  },

  resetAll: () => set({
    runs: new Map(),
    activeRunId: null,
    baselineRunId: null,
    paused: false,
    playbackSpeed: 1,
    error: null,
    route: 'viz',
    chat: [],
    dslText: '',
    // llmModel and steppedMode are user preferences — preserved across resets.
  }),
}));
