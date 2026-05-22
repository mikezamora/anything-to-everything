/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev server proxies the FastAPI backend so the SPA can use same-origin paths.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/run': 'http://localhost:8000',
      '/ws': { target: 'http://localhost:8000', ws: true },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
  },
});
