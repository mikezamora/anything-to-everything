// Temporary DOM-dump probe (delete after diagnosis).
import { render } from '@testing-library/react';
import { describe, it } from 'vitest';
import { MultifieldPanel } from './MultifieldPanel';
import snap from './__fixtures__/mf_live.json';
import type { Frame } from '../lib/types';

describe('probe', () => {
  it('dumps DOM with live shape frame', () => {
    const frame = {
      step: 5,
      layer_states: { multifield: snap },
    } as unknown as Frame;
    const { container } = render(<MultifieldPanel frame={frame} />);
    // eslint-disable-next-line no-console
    console.log('====DOM_START====\n' + container.innerHTML + '\n====DOM_END====');
  });
});
