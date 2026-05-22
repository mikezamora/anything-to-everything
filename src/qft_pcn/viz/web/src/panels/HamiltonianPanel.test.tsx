import { render } from '@testing-library/react';
import { it } from 'vitest';
import { HamiltonianPanel } from './HamiltonianPanel';
import { hamiltonianFrame, emptyFrame } from './__fixtures__/frames';

it('mounts with a frame', () => {
  render(<HamiltonianPanel frame={hamiltonianFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<HamiltonianPanel frame={emptyFrame} />);
});
