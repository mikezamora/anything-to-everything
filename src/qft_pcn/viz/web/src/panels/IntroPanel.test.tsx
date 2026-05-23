// src/qft_pcn/viz/web/src/panels/IntroPanel.test.tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { IntroQftPanel } from './IntroQftPanel';
import { IntroPcnPanel } from './IntroPcnPanel';
import { IntroQpcnPanel } from './IntroQpcnPanel';
import { SECTIONS } from '../lib/sections';

const frame = { step: 0, layer_states: {} } as any;

describe('Intro panels', () => {
  it('IntroQftPanel renders the QFT section title', () => {
    render(<IntroQftPanel frame={frame} />);
    expect(screen.getByText('QFT Substrate')).toBeInTheDocument();
  });
  it('IntroPcnPanel renders the PCN section title', () => {
    render(<IntroPcnPanel frame={frame} />);
    expect(screen.getByText('PCN Substrate')).toBeInTheDocument();
  });
  it('IntroQpcnPanel renders the QPCN section title', () => {
    render(<IntroQpcnPanel frame={frame} />);
    expect(screen.getByText('QPCN Fusion')).toBeInTheDocument();
  });
  it('each panel lists its layer summaries', () => {
    for (const sec of SECTIONS) {
      const Panel = sec.key === 'qft' ? IntroQftPanel
                  : sec.key === 'pcn' ? IntroPcnPanel
                  : IntroQpcnPanel;
      const { unmount } = render(<Panel frame={frame} />);
      for (const ls of sec.layerSummaries) {
        expect(screen.getByText(new RegExp(ls.oneLine.slice(0, 20), 'i')))
          .toBeInTheDocument();
      }
      unmount();
    }
  });
});
