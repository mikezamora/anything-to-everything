import { render } from '@testing-library/react';
import { it } from 'vitest';
import { MpsPanel } from './MpsPanel';
import { mpsFrame, emptyFrame } from './__fixtures__/frames';

it('mounts with a frame', () => {
  render(<MpsPanel frame={mpsFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<MpsPanel frame={emptyFrame} />);
});
