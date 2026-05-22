import { render } from '@testing-library/react';
import { it } from 'vitest';
import { ManifoldPanel } from './ManifoldPanel';
import { manifoldFrame, emptyFrame } from './__fixtures__/frames';

it('mounts with a frame', () => {
  render(<ManifoldPanel frame={manifoldFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<ManifoldPanel frame={emptyFrame} />);
});
