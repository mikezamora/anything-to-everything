/**
 * Zustand store holding the streamed `Frame` history and scrub state.
 *
 * `frames` is a ring buffer capped at `MAX_FRAMES`; once full, the oldest
 * frame is dropped as a new one arrives. When `live` is true, `pushFrame`
 * keeps the cursor pinned to the newest frame so the UI follows the stream.
 */

import { create } from 'zustand';
import type { Frame } from './lib/types';

/** Ring-buffer capacity for the streamed frame history. */
export const MAX_FRAMES = 2000;

interface VizState {
  frames: Frame[];
  cursor: number;
  live: boolean;
  selectedLayer: string;
  error: string | null;

  pushFrame: (frame: Frame) => void;
  setCursor: (cursor: number) => void;
  reset: () => void;
  selectLayer: (layer: string) => void;
  setLive: (live: boolean) => void;
  setError: (error: string | null) => void;
  currentFrame: () => Frame | undefined;
}

export const useVizStore = create<VizState>((set, get) => ({
  frames: [],
  cursor: 0,
  live: true,
  selectedLayer: 'manifold',
  error: null,

  pushFrame: (frame) =>
    set((state) => {
      let frames = [...state.frames, frame];
      if (frames.length > MAX_FRAMES) {
        frames = frames.slice(frames.length - MAX_FRAMES);
      }
      const cursor = state.live
        ? frames.length - 1
        : Math.min(state.cursor, frames.length - 1);
      return { frames, cursor };
    }),

  setCursor: (cursor) =>
    set((state) => {
      const max = Math.max(0, state.frames.length - 1);
      return { cursor: Math.max(0, Math.min(cursor, max)) };
    }),

  reset: () => set({ frames: [], cursor: 0, live: true, error: null }),

  selectLayer: (layer) => set({ selectedLayer: layer }),

  setLive: (live) => set({ live }),

  setError: (error) => set({ error }),

  currentFrame: () => {
    const { frames, cursor } = get();
    return frames[cursor];
  },
}));
