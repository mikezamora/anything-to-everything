import { render } from '@testing-library/react';
import { it } from 'vitest';
import { VqcPanel } from './VqcPanel';
import { vqcFrame, emptyFrame } from './__fixtures__/frames';

it('mounts with a frame', () => {
  render(<VqcPanel frame={vqcFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<VqcPanel frame={emptyFrame} />);
});
