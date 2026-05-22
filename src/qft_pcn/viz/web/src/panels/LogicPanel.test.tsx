import { render } from '@testing-library/react';
import { it } from 'vitest';
import { LogicPanel } from './LogicPanel';
import { logicFrame, emptyFrame } from './__fixtures__/frames';

it('mounts with a frame', () => {
  render(<LogicPanel frame={logicFrame} />);
});

it('mounts with an empty layer state', () => {
  render(<LogicPanel frame={emptyFrame} />);
});
