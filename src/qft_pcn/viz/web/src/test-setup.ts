/**
 * Vitest setup: jsdom lacks WebGL and a real layout engine, so we mock the
 * libraries that need them. r3f's `<Canvas>` is replaced with a plain div that
 * renders nothing (its children touch WebGL), and `plotly.js-dist-min` is
 * stubbed so panels can call `newPlot`/`react`/`purge` without crashing.
 *
 * Panels are still real components — these mocks only neutralize the
 * browser-graphics dependencies that jsdom cannot provide.
 */

import { vi } from 'vitest';
import React from 'react';

// --- react-three-fiber: Canvas needs WebGL; render an inert placeholder. -----
vi.mock('@react-three/fiber', async () => {
  const actual =
    await vi.importActual<typeof import('@react-three/fiber')>(
      '@react-three/fiber',
    );
  return {
    ...actual,
    // Render an inert div; drop r3f-only props (children, camera, etc.) that
    // are not valid DOM attributes so React does not warn under jsdom.
    Canvas: (_props: Record<string, unknown>) =>
      React.createElement('div', { 'data-testid': 'r3f-canvas' }),
  };
});

// --- @react-three/drei: components used inside Canvas never render in tests. --
vi.mock('@react-three/drei', () => ({
  OrbitControls: () => null,
  Line: () => null,
  Text: () => null,
  Html: ({ children }: { children?: React.ReactNode }) =>
    React.createElement(React.Fragment, null, children),
}));

// --- plotly: no real DOM measurement under jsdom. ----------------------------
vi.mock('plotly.js-dist-min', () => ({
  default: {
    newPlot: vi.fn().mockResolvedValue(undefined),
    react: vi.fn().mockResolvedValue(undefined),
    purge: vi.fn(),
    relayout: vi.fn().mockResolvedValue(undefined),
  },
}));

// jsdom has no ResizeObserver; provide a no-op so resize-aware panels mount.
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
}
